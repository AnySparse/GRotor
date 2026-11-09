#pragma once
// Distributed routing kernels with BFS layout, reusable scratch storage,
// separate demand commits and full-matrix peer indexing at every GPU count.
// Source attribution and license notices are retained in NOTICE.md.
namespace grotor::multi::legacy {
constexpr int block_size = 256, warp_size = 32, blocks = 1056;
using demandType = int; // Doubled units: wire=2, half-via=1.
struct P2PContext {
    float* vcost[8]{};
    double* presum[8]{};
    demandType* demand[8]{};
    bool* wcost_inc[8]{};
    bool* vcost_inc[8]{};
    int* timestamp[8]{};
};
#pragma push_macro("IDX")
#undef IDX
#define IDX(l,x,y) ((l) * X_Size * Y_Size + (x) * Y_Size + (y))


__device__ __forceinline__ void get_target_pe_and_offset(
    int l, int x, int y,
    int X_Size, int Y_Size, int nranks,
    int &target_pe, size_t &local_offset)
{

    int chunk_size = (X_Size + nranks - 1) / nranks;
    target_pe = x / chunk_size;

    if (target_pe >= nranks) target_pe = nranks - 1;

    local_offset = (size_t)l * X_Size * Y_Size + (size_t)x * Y_Size + (size_t)y;
}


template<class T> __device__ __forceinline__ void write_p2p(T** base_ptrs, size_t offset, T val, int target_pe) {

    base_ptrs[target_pe][offset] = val;
}


template<class T> __device__ __forceinline__ void atomic_add_p2p(T** base_ptrs, size_t offset, T val, int target_pe) {

    atomicAdd(&base_ptrs[target_pe][offset], val);
}


__device__ __forceinline__ float read_vcost(
    int l, int x, int y,
    int X_Size, int Y_Size,
    int rank, int nranks, P2PContext p2p)
{
    int target_pe;
    size_t local_offset;

    get_target_pe_and_offset(l, x, y, X_Size, Y_Size, nranks, target_pe, local_offset);

    return p2p.vcost[target_pe][local_offset];
}


__device__ __forceinline__ double wire_segment_cost(
    int layer, int xmin, int xmax, int ymin, int ymax,
    int X_Size, int Y_Size, int nranks,
    P2PContext p2p)
{
    // InstantGR2 defines a zero-length wire query as zero.
    if (xmin == xmax && ymin == ymax) return 0.;

    int chunk_size = (X_Size + nranks - 1) / nranks;

    int dx = (xmax > xmin) ? 1 : 0;
    int dy = (ymax > ymin) ? 1 : 0;

    int pe_start = xmin / chunk_size;
    int pe_end   = (dx ? xmax - 1 : xmin) / chunk_size;

    double total_cost = 0.0;

    for (int pe = pe_start; pe <= pe_end; pe++) {
        int pe_x_start = pe * chunk_size;
        int pe_x_end   = (pe + 1) * chunk_size;

        int cur_xmin = max(xmin, pe_x_start);
        int cur_xmax = min(xmax, pe_x_end);

        int end_x = dx ? cur_xmax - 1 : xmin;
        int end_y = dy ? ymax - 1 : ymin;

        double cost = p2p.presum[pe][IDX(layer, end_x, end_y)];

        int start_x = dx ? cur_xmin : xmin;
        int start_y = dy ? ymin : ymin;

        if ((dx && start_x > pe_x_start) || (dy && start_y > 0)) {
            int read_sx = dx ? start_x - 1 : xmin;
            int read_sy = dy ? start_y - 1 : ymin;

            cost -= p2p.presum[pe][IDX(layer, read_sx, read_sy)];
        }

        total_cost += cost;
    }

    return total_cost;
}


__device__ void add_unit_demand_update_timestamp(
    int l, int minx, int maxx, int miny, int maxy, int stamp,
    int X_Size, int Y_Size, int *pre_demand, int *timestamp,
    int rank, int nranks, int K = 1)
{
    assert(minx == maxx || miny == maxy);

    int target_idx = -1;
    if(minx == maxx) {
        target_idx = IDX(l, minx, maxy);
    }
    else if(miny == maxy) {
        target_idx = IDX(l, maxx, miny);
    }

    int target_x = (minx == maxx) ? minx : maxx;
    int target_y = (minx == maxx) ? maxy : miny;

    int target_pe;
    size_t local_offset;
    get_target_pe_and_offset(l, target_x, target_y, X_Size, Y_Size, nranks, target_pe, local_offset);

    if (target_pe == rank) {

        timestamp[local_offset] = stamp;
    } else {

        nvshmem_int_p(timestamp + local_offset, stamp, target_pe);
    }

}


__global__ void compute_presum_nvshmem(
    int L_Size, int X_Size, int Y_Size, int XY_max, int DIR,
    int *idx2track, float *wcost, double *presum,
    int rank, int nranks)
{
    extern __shared__ double sum[];

    int chunk_size = (X_Size + nranks - 1) / nranks;
    int l = idx2track[blockIdx.x] / XY_max;

    if(l & 1 ^ DIR) {

        int x = idx2track[blockIdx.x] % XY_max;

        int target_pe = x / chunk_size;
        if (target_pe >= nranks) target_pe = nranks - 1;
        if (target_pe != rank) return;

        for(int y = threadIdx.x; y < Y_Size; y += blockDim.x)
            sum[y] = wcost[l * X_Size * Y_Size + x * Y_Size + y];
        __syncthreads();

        for(int d = 0; (1 << d) < Y_Size; d++) {
            for(int idx = threadIdx.x; idx < Y_Size; idx += blockDim.x)
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int y = threadIdx.x; y < Y_Size; y += blockDim.x)
            presum[l * X_Size * Y_Size + x * Y_Size + y] = sum[y];

    } else {

        int y = idx2track[blockIdx.x] % XY_max;

        int start_x = rank * chunk_size;
        int end_x = (rank == nranks - 1) ? X_Size : start_x + chunk_size;
        int local_x_size = end_x - start_x;

        if (local_x_size <= 0) return;

        for(int local_x = threadIdx.x; local_x < local_x_size; local_x += blockDim.x)
            sum[local_x] = wcost[l * X_Size * Y_Size + (start_x + local_x) * Y_Size + y];
        __syncthreads();

        for(int d = 0; (1 << d) < local_x_size; d++) {
            for(int idx = threadIdx.x; idx < local_x_size; idx += blockDim.x)
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int local_x = threadIdx.x; local_x < local_x_size; local_x += blockDim.x)
            presum[l * X_Size * Y_Size + (start_x + local_x) * Y_Size + y] = sum[local_x];
    }
}


__global__ void L_shape_route_inc_warp_centric_nvshmem_p2p(
    int net_cnt,
    int net_offset,
    int *node_cnt_sum,
    int *nodes_,
    int *net_depth_offset,
    int *net_depth_cnt,
    int *par_nodes_,
    double *dist_,
    int *from_,
    int *layer_range_,
    int stamp,
    unsigned long long ROUTE_PER_PIN,
    int *routes,
    int *pin_acc_num,
    int *net_ids,
    int L_Size,
    int X_Size,
    int Y_Size,
    int DIR,
    float *vcost,
    float *wcost,
    double *presum,
    int *pre_demand,
    int *timestamp,
    int max_net_nodes_num,

    int rank,
    int nranks,
    P2PContext p2p
) {
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id     = thread_id   / warp_size;
    int thread_lane = threadIdx.x & (warp_size-1);
    int warp_lane   = threadIdx.x / warp_size;
    int num_warps   = (block_size / warp_size) * gridDim.x;
    int block_id    = blockIdx.x;
    int num_blocks  = gridDim.x;
    int block_thread_id = threadIdx.x;

    for (int net_id = warp_id; net_id < net_cnt; net_id += num_warps) {
        int net_idx = net_id + net_offset;
        int depth_cnt = net_depth_cnt[net_idx + 1] - net_depth_cnt[net_idx];
        int net_node_cnt = node_cnt_sum[net_idx + 1] - node_cnt_sum[net_idx];
        int *net_routes = routes + pin_acc_num[net_ids[net_idx]] * ROUTE_PER_PIN;
        int *nodes = nodes_ + node_cnt_sum[net_idx];
        int *par_nodes = par_nodes_ + node_cnt_sum[net_idx];
        int *layer_range = layer_range_ + node_cnt_sum[net_idx];
        double *dist = dist_ + warp_id * max_net_nodes_num * L_Size * L_Size;
        int *from = from_ + warp_id * max_net_nodes_num * L_Size * L_Size;

        for (int i = thread_lane; i < net_node_cnt; i += warp_size) {
            int l = nodes[i] / X_Size / Y_Size, x = nodes[i] / Y_Size % X_Size, y = nodes[i] % Y_Size;
            double *node_dist = dist + i * L_Size * L_Size;
            for(int minl = 0; minl < L_Size; minl++) {
                node_dist[minl * L_Size + minl] = 0;
                for(int maxl = minl + 1; maxl < L_Size; maxl++)
                    node_dist[minl * L_Size + maxl] = node_dist[minl * L_Size + maxl - 1] + read_vcost(maxl - 1, x, y, X_Size, Y_Size, rank, nranks, p2p);
                for(int maxl = minl; maxl < L_Size; maxl++) {
                    if(l < L_Size && (minl > l || maxl < l)) node_dist[minl * L_Size + maxl] = INF;

                }
            }
        }
        __syncwarp();

        for (int d = depth_cnt - 1; d > 0; d--) {
            int global_depth_id = net_depth_cnt[net_idx] + d;
            int depth_node_cnt = net_depth_offset[global_depth_id + 1] - net_depth_offset[global_depth_id];
            int net_base_i = net_depth_offset[global_depth_id] - net_depth_offset[net_depth_cnt[net_idx]];
            for (int d_i = thread_lane; d_i < depth_node_cnt; d_i += warp_size) {
                int i = net_base_i + d_i;
                int x = nodes[i] / Y_Size % X_Size, y = nodes[i] % Y_Size;
                int px = nodes[par_nodes[i]] / Y_Size % X_Size, py = nodes[par_nodes[i]] % Y_Size;
                int minx = min(x, px), maxx = max(x, px), miny = min(y, py), maxy = max(y, py);
                double *node_dist = dist + i * L_Size * L_Size, *par_dist = dist + par_nodes[i] * L_Size * L_Size;
                assert(par_nodes[i] < i);
                int *prev = from + i * L_Size * L_Size;
                int cur_from[10];
                double min_cost_cur[10];
                double min_cost_par[100];

                for(int l = 0; l < L_Size; l++) {
                    min_cost_cur[l] = min_cost_par[l * L_Size + l] = INF;
                    for(int minl = 0; minl <= l; minl++) {
                        for(int maxl = l; maxl < L_Size; maxl++){
                            if(node_dist[minl * L_Size + maxl] < min_cost_cur[l]) {
                                min_cost_cur[l] = node_dist[minl * L_Size + maxl];
                                cur_from[l] = minl * L_Size + maxl;
                            }
                        }
                    }
                }

                if(x == px || y == py) {
                    for(int l = 0; l < L_Size; l++) {

                        if((l & 1 ^ DIR) == 0 && y != py) continue;
                        if((l & 1 ^ DIR) == 1 && x != px) continue;

                        min_cost_par[l * L_Size + l] = min_cost_cur[l] + wire_segment_cost(l, minx, maxx, miny, maxy, X_Size, Y_Size, nranks, p2p);
                        prev[l * L_Size + l] = cur_from[l] * L_Size * L_Size + l * L_Size + l;
                    }
                } else {

                    for(int curl = 0; curl < L_Size; curl++) {
                        double cost = min_cost_cur[curl];
                        if(curl & 1 ^ DIR)
                            cost += wire_segment_cost(curl, x, x, miny, maxy, X_Size, Y_Size, nranks, p2p);
                        else
                            cost += wire_segment_cost(curl, minx, maxx, y, y, X_Size, Y_Size, nranks, p2p);
                        for(int parl = curl & 1 ^ 1; parl < L_Size; parl += 2) {
                            assert(curl % 2 != parl % 2);
                            double cost2 = 0;

                            for(int l = min(curl, parl) + 1; l < max(curl, parl); l++) {
                                cost2 += (curl & 1 ^ DIR) ? read_vcost(l, x, py, X_Size, Y_Size, rank, nranks, p2p) : read_vcost(l, px, y, X_Size, Y_Size, rank, nranks, p2p);
                            }
                            if(parl & 1 ^ DIR) {
                                cost2 += wire_segment_cost(parl, px, px, miny, maxy, X_Size, Y_Size, nranks, p2p);
                            }
                            else {
                                cost2 += wire_segment_cost(parl, minx, maxx, py, py, X_Size, Y_Size, nranks, p2p);
                            }
                            if(cost + cost2 < min_cost_par[parl * L_Size + parl]) {
                                min_cost_par[parl * L_Size + parl] = cost + cost2;
                                prev[parl * L_Size + parl] = cur_from[curl] * L_Size * L_Size + curl * L_Size + parl;
                            }

                        }
                    }
                }

                for(int minl = 0; minl < L_Size; minl++) {
                    for(int maxl = minl + 1; maxl < L_Size; maxl++) {
                        if(min_cost_par[minl * L_Size + maxl - 1] <= min_cost_par[maxl * L_Size + maxl]) {
                            min_cost_par[minl * L_Size + maxl] = min_cost_par[minl * L_Size + maxl - 1];
                            prev[minl * L_Size + maxl] = prev[minl * L_Size + maxl - 1];
                        } else {
                            min_cost_par[minl * L_Size + maxl] = min_cost_par[maxl * L_Size + maxl];
                            prev[minl * L_Size + maxl] = prev[maxl * L_Size + maxl];
                        }
                    }
                    for(int maxl = minl; maxl < L_Size; maxl++) {
                        atomicAdd(&par_dist[minl * L_Size + maxl], min_cost_par[minl * L_Size + maxl]);

                    }
                }
            }
            __syncwarp();
        }
        if (thread_lane == 0) {
            net_routes[0] = 1;
            layer_range[0] = 0;
            for(int minl = 0; minl < L_Size; minl++)
                for(int maxl = minl; maxl < L_Size; maxl++)
                    if(dist[minl * L_Size + maxl] < dist[layer_range[0]]) {
                        layer_range[0] = minl * L_Size + maxl;
                    }
        }
        __syncwarp();

        for (int d = 0; d < depth_cnt; d++) {
            int global_depth_id = net_depth_cnt[net_idx] + d;
            int depth_node_cnt = net_depth_offset[global_depth_id + 1] - net_depth_offset[global_depth_id];

            int net_base_i = net_depth_offset[global_depth_id] - net_depth_offset[net_depth_cnt[net_idx]];
            for (int d_i = thread_lane; d_i < depth_node_cnt; d_i += warp_size) {

                int i = net_base_i + d_i;
                int *prev = from + i * L_Size * L_Size;

                if(i > 0) layer_range[i] = prev[layer_range[par_nodes[i]]] / L_Size / L_Size;
                assert(dist[i * L_Size * L_Size + layer_range[i]] < INF);

                int minl = layer_range[i] / L_Size, maxl = layer_range[i] % L_Size, x = nodes[i] / Y_Size % X_Size, y = nodes[i] % Y_Size;

                if(minl < maxl) {
                    int idd1 = atomicAdd(net_routes,2);
                    net_routes[idd1] = IDX(minl, x, y);
                    net_routes[idd1 + 1] = IDX(maxl, x, y);

                }

                if(i == 0) continue;
                int px = nodes[par_nodes[i]] / Y_Size % X_Size, py = nodes[par_nodes[i]] % Y_Size;
                int minx = min(x, px), maxx = max(x, px), miny = min(y, py), maxy = max(y, py);
                int curl = prev[layer_range[par_nodes[i]]] / L_Size % L_Size, parl = prev[layer_range[par_nodes[i]]] % L_Size;

                if(px == x || py == y) {
                    assert(curl == parl);
                    int idd1 = atomicAdd(net_routes,2);
                    net_routes[idd1] = IDX(curl, minx, miny);
                    net_routes[idd1 + 1] = IDX(curl, maxx, maxy);

                    add_unit_demand_update_timestamp(curl, minx, maxx, miny, maxy, stamp, X_Size, Y_Size, pre_demand, timestamp, rank, nranks);
                } else {
                    assert(curl % 2 != parl % 2);
                    int minl = min(curl, parl), maxl = max(curl, parl);
                    if(curl & 1 ^ DIR) {

                        int idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(curl, x, y);
                        net_routes[idd1 + 1] = IDX(curl, x, py);

                        add_unit_demand_update_timestamp(curl, x, x, miny, maxy, stamp, X_Size, Y_Size, pre_demand, timestamp, rank, nranks);

                        idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(curl, x, py);
                        net_routes[idd1 + 1] = IDX(parl, x, py);

                        idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(parl, x, py);
                        net_routes[idd1 + 1] = IDX(parl, px, py);

                        add_unit_demand_update_timestamp(parl, minx, maxx, py, py, stamp, X_Size, Y_Size, pre_demand, timestamp, rank, nranks);
                    } else {
                        int idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(curl, x, y);
                        net_routes[idd1 + 1] = IDX(curl, px, y);

                        add_unit_demand_update_timestamp(curl, minx, maxx, y, y, stamp, X_Size, Y_Size, pre_demand, timestamp, rank, nranks);

                        idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(curl, px, y);
                        net_routes[idd1 + 1] = IDX(parl, px, y);

                        idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(parl, px, y);
                        net_routes[idd1 + 1] = IDX(parl, px, py);

                        add_unit_demand_update_timestamp(parl, px, px, miny, maxy, stamp, X_Size, Y_Size, pre_demand, timestamp, rank, nranks);
                    }
                }
            }
            __syncwarp();
        }

    }

}


__global__ void commit_edge_inc_set_update_nvshmem_p2p(
    int net_cnt, int net_offset, int global_timestamp, int X_Size, int Y_Size, int DIR,
    int *net_ids, int *routes, unsigned long long ROUTE_PER_PIN,
    int* total_via_count, int *pin_acc_num,
    P2PContext p2p,
    int nranks,
    int K = 1)
{
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id    = blockIdx.x;
    int num_blocks  = gridDim.x;
    int thread_block_lane   = threadIdx.x;

    int target_pe;
    size_t local_offset;

    for (int net_idx = block_id; net_idx < net_cnt; net_idx += num_blocks) {
        int net_id = net_ids[net_idx + net_offset];
        int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;

        for (int i = 1; i < net_routes[0]; i += 2) {

            if(net_routes[i] / X_Size / Y_Size == net_routes[i + 1] / X_Size / Y_Size) {
                int l = net_routes[i] / X_Size / Y_Size;
                int x0 = net_routes[i] / Y_Size % X_Size, y0 = net_routes[i] % Y_Size;
                int x1 = net_routes[i + 1] / Y_Size % X_Size, y1 = net_routes[i + 1] % Y_Size;

                if(x0 < x1) for(int x = x0 + thread_block_lane; x < x1; x += block_size) {
                    get_target_pe_and_offset(l, x, y0, X_Size, Y_Size, nranks, target_pe, local_offset);

                    atomic_add_p2p(p2p.demand, local_offset, K, target_pe);
                    write_p2p(p2p.timestamp, local_offset, global_timestamp, target_pe);
                    write_p2p(p2p.wcost_inc, local_offset, true, target_pe);
                    write_p2p(p2p.vcost_inc, local_offset, true, target_pe);

                    if (x == x1 - 1) {
                        get_target_pe_and_offset(l, x + 1, y0, X_Size, Y_Size, nranks, target_pe, local_offset);
                        write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                    }
                }

                if(x1 < x0) for(int x = x1 + thread_block_lane; x < x0; x += block_size) {
                    get_target_pe_and_offset(l, x, y0, X_Size, Y_Size, nranks, target_pe, local_offset);

                    atomic_add_p2p(p2p.demand, local_offset, K, target_pe);
                    write_p2p(p2p.timestamp, local_offset, global_timestamp, target_pe);
                    write_p2p(p2p.wcost_inc, local_offset, true, target_pe);
                    write_p2p(p2p.vcost_inc, local_offset, true, target_pe);

                    if (x == x0 - 1) {
                        get_target_pe_and_offset(l, x + 1, y0, X_Size, Y_Size, nranks, target_pe, local_offset);
                        write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                    }
                }

                if(y0 < y1) for(int y = y0 + thread_block_lane; y < y1; y += block_size) {
                    get_target_pe_and_offset(l, x0, y, X_Size, Y_Size, nranks, target_pe, local_offset);

                    atomic_add_p2p(p2p.demand, local_offset, K, target_pe);
                    write_p2p(p2p.timestamp, local_offset, global_timestamp, target_pe);
                    write_p2p(p2p.wcost_inc, local_offset, true, target_pe);
                    write_p2p(p2p.vcost_inc, local_offset, true, target_pe);

                    if (y == y1 - 1){
                        get_target_pe_and_offset(l, x0, y + 1, X_Size, Y_Size, nranks, target_pe, local_offset);
                        write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                    }
                }

                if(y1 < y0) for(int y = y1 + thread_block_lane; y < y0; y += block_size) {
                    get_target_pe_and_offset(l, x0, y, X_Size, Y_Size, nranks, target_pe, local_offset);

                    atomic_add_p2p(p2p.demand, local_offset, K, target_pe);
                    write_p2p(p2p.timestamp, local_offset, global_timestamp, target_pe);
                    write_p2p(p2p.wcost_inc, local_offset, true, target_pe);
                    write_p2p(p2p.vcost_inc, local_offset, true, target_pe);

                    if (y == y0 - 1){
                        get_target_pe_and_offset(l, x0, y + 1, X_Size, Y_Size, nranks, target_pe, local_offset);
                        write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                    }
                }
            }
        }
    }
}


__global__ void commit_via_demand_set_update_nvshmem_p2p(
    int net_cnt, int net_offset, int global_timestamp, int X_Size, int Y_Size, int DIR,
    int *net_ids, int *routes, unsigned long long ROUTE_PER_PIN,
    int* total_via_count, int *pin_acc_num,
    P2PContext p2p,
    int nranks,
    int K = 1)
{
    int thread_id    = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads  = gridDim.x * blockDim.x;
    int target_pe;
    size_t local_offset;

    for (int net_idx = thread_id; net_idx < net_cnt; net_idx += num_threads) {
        int net_id = net_ids[net_idx + net_offset];
        int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;

        for(int i = 1; i < net_routes[0]; i += 2) {
            if(net_routes[i] / X_Size / Y_Size != net_routes[i + 1] / X_Size / Y_Size) {
                int x = net_routes[i] / Y_Size % X_Size, y = net_routes[i] % Y_Size;
                int l0 = net_routes[i] / X_Size / Y_Size, l1 = net_routes[i + 1] / X_Size / Y_Size;
                int minl = min(l0, l1), maxl = max(l0, l1);

                for(int l = minl; l < maxl; l++) {

                    int ts_pe;
                    size_t ts_offset;
                    get_target_pe_and_offset(l, x, y, X_Size, Y_Size, nranks, ts_pe, ts_offset);
                    int current_ts = p2p.timestamp[ts_pe][ts_offset];

                    if(current_ts < global_timestamp) {

                        get_target_pe_and_offset(l, x, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                        write_p2p(p2p.vcost_inc, local_offset, true, target_pe);

                        if(l & 1 ^ DIR) {
                            if(y == 0) {
                                get_target_pe_and_offset(l, x, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                atomic_add_p2p(p2p.demand, local_offset, (demandType)K, target_pe);
                                write_p2p(p2p.wcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x, y + 1, X_Size, Y_Size, nranks, target_pe, local_offset);
                                write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                            }
                            else if(y == Y_Size - 1) {
                                get_target_pe_and_offset(l, x, y - 1, X_Size, Y_Size, nranks, target_pe, local_offset);
                                atomic_add_p2p(p2p.demand, local_offset, (demandType)K, target_pe);
                                write_p2p(p2p.wcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x, y - 1, X_Size, Y_Size, nranks, target_pe, local_offset);
                                write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                            }
                            else {
                                get_target_pe_and_offset(l, x, y - 1, X_Size, Y_Size, nranks, target_pe, local_offset);
                                atomic_add_p2p(p2p.demand, local_offset, (demandType)(0.5 * K), target_pe);
                                write_p2p(p2p.wcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                atomic_add_p2p(p2p.demand, local_offset, (demandType)(0.5 * K), target_pe);
                                write_p2p(p2p.wcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x, y - 1, X_Size, Y_Size, nranks, target_pe, local_offset);
                                write_p2p(p2p.vcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x, y + 1, X_Size, Y_Size, nranks, target_pe, local_offset);
                                write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                            }
                        } else {
                            if(x == 0) {
                                get_target_pe_and_offset(l, x, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                atomic_add_p2p(p2p.demand, local_offset, (demandType)K, target_pe);
                                write_p2p(p2p.wcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x + 1, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                            }
                            else if(x == X_Size - 1) {
                                get_target_pe_and_offset(l, x - 1, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                atomic_add_p2p(p2p.demand, local_offset, (demandType)K, target_pe);
                                write_p2p(p2p.wcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x - 1, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                            }
                            else {
                                get_target_pe_and_offset(l, x - 1, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                atomic_add_p2p(p2p.demand, local_offset, (demandType)(0.5 * K), target_pe);
                                write_p2p(p2p.wcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                atomic_add_p2p(p2p.demand, local_offset, (demandType)(0.5 * K), target_pe);
                                write_p2p(p2p.wcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x - 1, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                write_p2p(p2p.vcost_inc, local_offset, true, target_pe);

                                get_target_pe_and_offset(l, x + 1, y, X_Size, Y_Size, nranks, target_pe, local_offset);
                                write_p2p(p2p.vcost_inc, local_offset, true, target_pe);
                            }
                        }
                    }
                }
            }
        }
    }
}


#pragma pop_macro("IDX")
} // namespace grotor::multi::legacy

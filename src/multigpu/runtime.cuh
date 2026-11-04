#pragma once

// One MPI rank owns one GPU. Only rank zero owns the CPU routing database.
// Small batches use the existing executor; the final batch shares one immutable
// spatially sharded cost snapshot and exchanges disjoint routes and wire contributions through
// NVSHMEM's NVLink mappings. A global barrier separates all wires from all vias.
#include "routing/lshape_warp.cuh"
#include "routing/flt_block.cuh"
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <numeric>
#include <exception>
#ifdef GROTOR_MULTI_LEGACY_LSHAPE
#include <nccl.h>
#include "multigpu/legacy_lshape_kernels.cuh"
#endif

namespace grotor::multi {
using Clock = std::chrono::steady_clock;
inline double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
inline void mpi_check(int status) {
    if (status != MPI_SUCCESS) throw std::runtime_error("MPI operation failed");
}
inline void broadcast(void* pointer, size_t bytes) {
    auto* p = static_cast<unsigned char*>(pointer);
    while (bytes) {
        const int count = int(std::min(bytes, size_t(1) << 28));
        mpi_check(MPI_Bcast(p, count, MPI_BYTE, 0, MPI_COMM_WORLD));
        bytes -= count; p += count;
    }
}
template<class T> void broadcast(T& object) { broadcast(&object, sizeof(T)); }
template<class T> void transfer(std::vector<T>& values, int peer, bool send) {
    uint64_t count = values.size();
    if (send) mpi_check(MPI_Send(&count, sizeof(count), MPI_BYTE, peer, 0, MPI_COMM_WORLD));
    else {
        mpi_check(MPI_Recv(&count, sizeof(count), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        values.resize(count);
    }
    auto* p = reinterpret_cast<unsigned char*>(values.data());
    size_t bytes = values.size() * sizeof(T);
    while (bytes) {
        const int n = int(std::min(bytes, size_t(1) << 28));
        if (send) mpi_check(MPI_Send(p, n, MPI_BYTE, peer, 0, MPI_COMM_WORLD));
        else mpi_check(MPI_Recv(p, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        bytes -= n; p += n;
    }
}

enum class Command : int { stop, lshape, flt, legacy_lshape };
struct Header {
    Command command;
    RoutingDeviceView dimensions; // Only scalar fields survive on worker ranks.
    int nets = 0, stamp = 0, batch_index = 0, batches = 0;
    uint64_t route_ints = 0;
};
struct Payload {
    int begin = 0;
    std::vector<int> offsets, nodes, parents;
    std::vector<flt::NetDescriptor> descriptors;
    std::vector<flt::Node> flt_nodes;
    std::vector<flt::Edge> flt_edges;
    void exchange(int peer, bool send) {
        if (send) mpi_check(MPI_Send(&begin, 1, MPI_INT, peer, 0, MPI_COMM_WORLD));
        else mpi_check(MPI_Recv(&begin, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        transfer(offsets, peer, send); transfer(nodes, peer, send); transfer(parents, peer, send);
        transfer(descriptors, peer, send); transfer(flt_nodes, peer, send); transfer(flt_edges, peer, send);
    }
    int nets(Command command) const {
        return command == Command::lshape ? int(offsets.size()) - 1 : int(descriptors.size());
    }
};

struct SymmetricMemory {
    std::vector<void*> allocations;
    template<class T> T* allocate(size_t count) {
        void* p = nvshmem_malloc(std::max(size_t(1), count) * sizeof(T));
        if (!p) throw std::runtime_error("NVSHMEM allocation failed; increase NVSHMEM_SYMMETRIC_SIZE or reduce the batch");
        allocations.push_back(p);
        return static_cast<T*>(p);
    }
    ~SymmetricMemory() {
        // The outer handler aborts the communicator on failure. A collective
        // free during unwinding could wait forever for a failed peer.
        if (std::uncaught_exceptions()) return;
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) nvshmem_free(*it);
    }
};
template<class T> T* root_pointer(T* local) {
    auto* p = static_cast<T*>(nvshmem_ptr(local, 0));
    if (!p) throw std::runtime_error("Every worker must have an NVLink P2P mapping to PE 0");
    return p;
}
template<class T> void upload(T* device, const std::vector<T>& host) {
    if (!host.empty()) cuda_check(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload multi-GPU metadata");
}
inline void copy(void* target, const void* source, size_t bytes) {
    cuda_check(cudaMemcpyAsync(target, source, bytes, cudaMemcpyDefault), "NVLink snapshot copy");
}

struct TimingEvent {
    cudaEvent_t value = nullptr;
    TimingEvent() { cuda_check(cudaEventCreate(&value), "create batch timing event"); }
    ~TimingEvent() { if (value) cudaEventDestroy(value); }
    void record() { cuda_check(cudaEventRecord(value), "record batch timing event"); }
    float since(const TimingEvent& first) const {
        float value_ms = 0;
        cuda_check(cudaEventElapsedTime(&value_ms, first.value, value), "read batch timing event");
        return value_ms;
    }
};

__global__ void initialize_ids(int* ids, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = i;
}
__global__ void scatter_routes(RoutingDeviceView compact, RoutingDeviceView original,
                               const int* ids, bool second) {
    const int slot = blockIdx.x, id = ids[slot];
    const int* source = compact.routes + size_t(compact.pin_acc_num[slot]) * compact.ROUTE_PER_PIN;
    int* target = second ? original.routes_phase2 + size_t(original.pin_acc_num_phase2[id]) * original.ROUTE_PER_PIN_PHASE2
                         : original.routes + size_t(original.pin_acc_num[id]) * original.ROUTE_PER_PIN;
    for (int i = threadIdx.x; i < source[0]; i += blockDim.x) target[i] = source[i];
}

__global__ void pack_phase_one_routes(RoutingDeviceView compact, RoutingDeviceView original, const int* ids) {
    const int slot = blockIdx.x, id = ids[slot];
    const int* source = original.routes + size_t(original.pin_acc_num[id]) * original.ROUTE_PER_PIN;
    int* target = compact.routes + size_t(compact.pin_acc_num[slot]) * compact.ROUTE_PER_PIN_PHASE2;
    for (int i = threadIdx.x; i < source[0]; i += blockDim.x) target[i] = source[i];
}

#ifdef GROTOR_MULTI_GRID
#include "multigpu/grid.cuh"
#endif
#ifdef GROTOR_MULTI_LEGACY_LSHAPE
#include "multigpu/legacy_lshape.cuh"
#endif

class Session;
inline Session* session = nullptr;

class Session {
public:
    int rank = 0, size = 1;
    Session(int& argc, char**& argv) {
        int provided = 0;
        mpi_check(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &size));
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED unavailable");
        MPI_Comm local;
        mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_size = 0, local_rank = 0, devices = 0;
        MPI_Comm_size(local, &local_size); MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
        cuda_check(cudaGetDeviceCount(&devices), "visible GPU count");
        if (local_size != size || size > devices || size > 8)
            throw std::runtime_error("Use 1-8 ranks on one NVLink host with a distinct visible GPU per rank");
        for (int peer = 0; peer < size; ++peer) if (peer != local_rank) {
            int atomics = 0;
            cuda_check(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported,
                local_rank, peer), "NVLink native peer atomic support");
            if (!atomics) throw std::runtime_error("All participating GPUs must support native peer atomics");
        }
        cuda_check(cudaSetDevice(local_rank), "bind MPI rank to GPU");
        nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
        if (!rank && nvshmemx_get_uniqueid(&uid)) throw std::runtime_error("NVSHMEM UID failed");
        broadcast(uid);
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_set_attr_uniqueid_args(rank, size, &uid, &attr);
        // This executor uses host/on-stream APIs and nvshmem_ptr mappings;
        // no device-side NVSHMEM calls or relocatable CUDA code are required.
#ifdef GROTOR_MULTI_LEGACY_LSHAPE
        if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr)) throw std::runtime_error("NVSHMEM device initialization failed");
#else
        if (nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr)) throw std::runtime_error("NVSHMEM initialization failed");
#endif
        if (nvshmem_my_pe() != rank || nvshmem_n_pes() != size) throw std::runtime_error("MPI/PE rank mismatch");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, local_rank), "GPU properties");
        // Workers have not executed the earlier batches. Load their solver
        // kernels before measurement, matching rank zero's warm CUDA module.
        cudaFuncAttributes attributes{};
        cuda_check(cudaFuncGetAttributes(&attributes, Lshape_route::Lshape_route_warp), "load L-shape solver");
        cuda_check(cudaFuncGetAttributes(&attributes, Lshape_route_detour::precompute_cost_edge), "load FLT edge costs");
        cuda_check(cudaFuncGetAttributes(&attributes, Lshape_route_detour::route_flt_block), "load FLT solver");
        printf("MULTIGPU_DEVICE rank=%d pe=%d device=%d name=%s\n", rank, rank, local_rank, properties.name);
#ifdef GROTOR_MULTI_LEGACY_LSHAPE
        if (!rank) printf("MULTIGPU_LSHAPE_EXECUTOR implementation=legacy-p2p blocks=1056 threads=256 demand=int2 dirty=bool matrix=full single_path=peer all_batches=distributed\n");
#endif
#ifdef GROTOR_MULTI_GRID
        if (!rank) printf("MULTIGPU_CONFIG grid=spatial cost=incremental prefix=local-dense bit=off-all-batches commit=distributed timing=my-gr-cuda-event\n");
#else
        if (!rank) printf("MULTIGPU_CONFIG grid=root cost=incremental prefix=bit commit=root\n");
#endif
        session = this;
    }
    ~Session() {
        if (std::uncaught_exceptions()) return;
        if (!rank) { Command stop = Command::stop; broadcast(stop); }
#ifdef GROTOR_MULTI_LEGACY_LSHAPE
        nvshmem_finalize();
#else
        nvshmemx_hostlib_finalize();
#endif
        MPI_Finalize(); session = nullptr;
    }
    void sync() const {
        // Host barriers alone do not order CUDA kernels and P2P stores.
        nvshmemx_barrier_all_on_stream(nullptr);
        cuda_check(cudaDeviceSynchronize(), "complete NVLink work and collective barrier");
    }
    void serve() {
        for (;;) {
            Command command = Command::stop;
            broadcast(command);
            if (command == Command::stop) break;
#ifdef GROTOR_MULTI_LEGACY_LSHAPE
            if (command == Command::legacy_lshape) { legacy::run(nullptr,rank,size); continue; }
#endif
            run(nullptr, {}, nullptr, nullptr, command);
        }
    }

#ifdef GROTOR_MULTI_LEGACY_LSHAPE
    void lshape_all(RouterState& state,const std::vector<std::vector<int>>& batches,
                    const std::vector<int>& ordered_ids,const int* offsets,const int* nodes,const int* parents) {
        const auto start=Clock::now();
        Command command=Command::legacy_lshape; broadcast(command);
        legacy::Header header{state.device,int(batches.size()),int(ordered_ids.size()),int(batches.back().size()),state.global_timestamp};
        std::vector<legacy::Payload> payloads(size);
        const int chunk=(state.device.X+size-1)/size;
        size_t slot=0;
        for (const auto& batch:batches) {
            for (int id:batch) {
                assert(ordered_ids[slot]==id);
                const int owner=((state.nets[id].minx+state.nets[id].maxx)/2)/chunk;
                const int begin=offsets[slot],count=offsets[slot+1]-begin;
                payloads[owner].append(id,nodes+begin,parents+begin,count);
                ++slot;
            }
            for (auto& payload:payloads) payload.batches.push_back(payload.ids.size());
        }
        legacy::run(&state,rank,size,&header,&payloads,start);
    }
#endif

    // Reordering independent nets preserves relative node IDs within each DAG.
    // Packed cost ties retain the same candidate order after local rebasing.
    std::vector<int> spatial_order(RouterState& state, std::vector<int>& ids) const {
        const int shard_x = (state.device.X + size - 1) / size;
        auto owner = [&](int id) { return ((state.nets[id].minx + state.nets[id].maxx) / 2) / shard_x; };
        // Independent nets within one CPU batch may be permuted. DAG candidate
        // order and all pre-batch costs stay unchanged; wire sums are integers.
        std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return owner(a) < owner(b); });
        std::vector<int> cuts{0};
        int next = 0;
        for (int p = 1; p < size; ++p) {
            while (next < ids.size() && owner(ids[next]) < p) ++next;
            cuts.push_back(next);
        }
        cuts.push_back(ids.size());
        return cuts;
    }

    void run(RouterState* state, const std::vector<int>& ids, const Header* root_header,
             std::vector<Payload>* payloads, Command command,
             Clock::time_point preparation_start = Clock::now()) {
        Header header{};
        if (!rank) header = *root_header;
        broadcast(header);
        Payload payload;
        std::vector<int> partitions(size + 1, header.nets);
        std::vector<int> pins(header.nets + 1);
        if (!rank) {
            for (int i = 0; i < header.nets; ++i)
                pins[i + 1] = pins[i] + state->nets[ids[i]].pins.size();
            payload = std::move((*payloads)[0]);
            for (int p = 0; p < size; ++p) partitions[p] = (*payloads)[p].begin;
            for (int p = 1; p < size; ++p) (*payloads)[p].exchange(p, true);
        } else payload.exchange(rank, false);
        broadcast(pins.data(), pins.size() * sizeof(int));
        const int count = payload.nets(command);
        const size_t cells = size_t(header.dimensions.L) * header.dimensions.X * header.dimensions.Y;
        const int shard_x = (header.dimensions.X + size - 1) / size;
        const size_t shard_cells = size_t(header.dimensions.L) * shard_x * header.dimensions.Y;
        SymmetricMemory symmetric;
#ifdef GROTOR_MULTI_GRID
        float* via = symmetric.allocate<float>(shard_cells);
        double* prefix = symmetric.allocate<double>(shard_cells);
#else
        float* via = symmetric.allocate<float>(size > 1 ? shard_cells : 1);
        double* prefix = symmetric.allocate<double>(size > 1 ? shard_cells : 1);
#endif
        int* wire = nullptr;
        int* timestamps = nullptr;
#ifndef GROTOR_MULTI_GRID
        wire = symmetric.allocate<int>(cells);
        timestamps = symmetric.allocate<int>(cells);
#endif
        int* routes = symmetric.allocate<int>(header.route_ints);
        int* pin_offsets = symmetric.allocate<int>(pins.size());
        int* all_ids = symmetric.allocate<int>(header.nets);
        int* flags = symmetric.allocate<int>(header.nets + 1);
#ifndef GROTOR_MULTI_GRID
        if (!rank) printf("MULTIGPU_STORAGE stage=%s shard_x=%d cost_bytes_per_gpu=%zu exchange_bytes_per_gpu=%zu\n",
            command == Command::lshape ? "lshape" : "flt", size > 1 ? shard_x : 0,
            size > 1 ? shard_cells * (sizeof(float) + sizeof(double)) : size_t(0),
            cells * 2 * sizeof(int) + header.route_ints * sizeof(int) + (size_t(header.nets) * 3 + 2) * sizeof(int));
#else
        if (!rank) printf("MULTIGPU_STORAGE stage=%s shard_x=%d grid_bytes_per_gpu=%zu route_bytes_per_gpu=%zu\n",
            command == Command::lshape ? "lshape" : "flt", shard_x,
            shard_cells * (4 * sizeof(float) + sizeof(double) + sizeof(int)) + size_t(shard_x) * header.dimensions.Y * sizeof(int),
            header.route_ints * sizeof(int));
#endif
        // Every worker checks mappings, even ranks assigned an empty partition.
        RoutingDeviceView view{};
        view.L = header.dimensions.L; view.X = header.dimensions.X; view.Y = header.dimensions.Y;
        view.XY = header.dimensions.XY; view.DIR = header.dimensions.DIR;
        view.NET_NUM = header.nets;
        view.ROUTE_PER_PIN = header.dimensions.ROUTE_PER_PIN;
        view.ROUTE_PER_PIN_PHASE2 = header.dimensions.ROUTE_PER_PIN_PHASE2;
        view.vcost = via; view.presum = prefix;
#ifdef GROTOR_MULTI_GRID
        if (true) {
#else
        if (size > 1) {
#endif
            view.cost_shard_x = shard_x;
            for (int p = 0; p < size; ++p) {
                view.peer_vcost[p] = static_cast<float*>(nvshmem_ptr(via, p));
                view.peer_presum[p] = static_cast<double*>(nvshmem_ptr(prefix, p));
                if (!view.peer_vcost[p] || !view.peer_presum[p])
                    throw std::runtime_error("All cost shards require direct NVLink peer mappings");
            }
        } else {
            view.vcost = state->device.vcost;
            view.presum = state->device.presum;
        }
#ifndef GROTOR_MULTI_GRID
        view.pre_demand = root_pointer(wire); view.timestamp = root_pointer(timestamps);
#endif
#ifdef GROTOR_MULTI_GRID
        view.routes = routes; view.routes_phase2 = routes;
        view.pin_acc_num = pin_offsets; view.pin_acc_num_phase2 = pin_offsets;
        view.ripup_flag = flags; view.net_ids = all_ids + payload.begin;
#else
        view.routes = root_pointer(routes); view.routes_phase2 = view.routes;
        view.pin_acc_num = root_pointer(pin_offsets); view.pin_acc_num_phase2 = view.pin_acc_num;
        view.ripup_flag = root_pointer(flags); view.net_ids = root_pointer(all_ids) + payload.begin;
#endif
#ifdef GROTOR_MULTI_GRID
        DistributedGrid grid(view, header, rank, size);
#endif

        DeviceMemory local;
        auto allocate = [&](auto*& pointer, size_t n) { local.allocate(&pointer, n * sizeof(*pointer)); };
        int *offsets = nullptr, *nodes = nullptr, *parents = nullptr, *from = nullptr, *ranges = nullptr;
        double* dist = nullptr;
        FltDeviceView flt_view{};
        flt::Topology topology;
        flt::Plans plans(topology, {});
        std::unique_ptr<flt::Workspace> workspace;
        if (command == Command::lshape) {
            allocate(offsets, payload.offsets.size()); allocate(nodes, payload.nodes.size());
            allocate(parents, payload.parents.size()); allocate(ranges, payload.nodes.size());
            allocate(dist, payload.nodes.size() * size_t(view.L) * view.L);
            allocate(from, payload.nodes.size() * size_t(view.L) * view.L);
        } else if (count) {
            topology.arenas.resize(1);
            for (auto node : payload.flt_nodes) topology.arenas[0].nodes.push(node);
            for (auto edge : payload.flt_edges) topology.arenas[0].edges.push(edge);
            plans.nets = std::move(payload.descriptors);
            flt::Batch batch;
            batch.nets = count; batch.nodes = payload.flt_nodes.size(); batch.edges = payload.flt_edges.size();
            for (auto node : payload.flt_nodes) batch.children += node.children;
            for (const auto& net : plans.nets) {
                batch.depth_slots += net.depths + 1;
                int children = 0;
                for (int n = 0; n < net.nodes; ++n) children += payload.flt_nodes[net.source_node + n].children;
                batch.slots.push_back(topology.spans.size());
                topology.spans.push_back({net.source_node, net.source_edge, net.nodes, net.edges, children, 0, net.depths});
            }
            plans.batches.push_back(batch);
            plans.max_nodes = plans.total_nodes = batch.nodes;
            plans.max_edges = plans.total_edges = batch.edges;
            plans.max_children = batch.children; plans.max_depth_slots = batch.depth_slots;
            workspace = std::make_unique<flt::Workspace>(flt_view, plans);
            int* net_ids = nullptr; allocate(net_ids, count); view.net_ids = net_ids;
            // Prime CUB/assembly code loading, as on the root's preceding small
            // batches. Actual upload, assembly and initialization run again
            // inside the measured interval; no DP solve is omitted.
            workspace->upload(topology, plans);
            workspace->assemble(topology, plans, plans.batches[0], view);
        }
        int* original_ids = nullptr;
        if (!rank) allocate(original_ids, ids.size());
#ifdef GROTOR_MULTI_GRID
        // The legacy batch timer assumes the distributed grid and routes are
        // already resident. Initialization and final collection are reported
        // separately; rip-up, cost/prefix updates and commits remain timed.
        upload(pin_offsets, pins);
        initialize_ids<<<BLOCK_NUM(header.nets), THREAD_NUM>>>(all_ids, header.nets);
        cuda_check(cudaMemsetAsync(flags, command == Command::flt ? 1 : 0, (header.nets + 1) * sizeof(int)), "set local compact rip-up flags");
        if (!rank) {
            upload(original_ids, ids);
            state->global_timestamp = header.stamp;
            grid.snapshot(*state);
            if (command == Command::flt)
                pack_phase_one_routes<<<header.nets, 128>>>(view, state->device, original_ids);
        }
        sync();
        grid.prepare_tables();
        if (command == Command::flt) grid.distribute_routes(routes, pins, partitions);
        if (command == Command::lshape) {
            // The legacy L-shape loop uploads topology and initializes costs
            // before its start event; each batch refreshes costs after commit.
            upload(offsets, payload.offsets); upload(nodes, payload.nodes); upload(parents, payload.parents);
            grid.costs(); grid.prefix();
        }
        TimingEvent begin_event, cost_event, solve_event, commit_event, end_event;
#endif
        sync();
        const double preparation_ms = ms(preparation_start);
        const auto start = Clock::now();
#ifdef GROTOR_MULTI_GRID
        begin_event.record();
        if (command == Command::flt) {
            if (count) {
                workspace->upload(topology, plans);
                workspace->assemble(topology, plans, plans.batches[0], view);
            }
            if (count) graph::commit_partition_wires<<<std::min(count, 1056), 128>>>(view, count, header.stamp - 1, -1);
            sync();
            if (count) graph::commit_via_demand<<<BLOCK_NUM(count), THREAD_NUM>>>(view, count, 0, header.stamp - 1, -1);
            sync();
            grid.costs();
            grid.prefix();
            sync();
        }
        cost_event.record();
#else
        if (!rank) {
            if (command == Command::flt) {
                // Rip-up reads the original phase-one routes and must finish
                // before the common pre-batch cost snapshot is refreshed.
                upload(original_ids, ids);
                auto original = state->device; original.net_ids = original_ids;
                graph::commit_wire_demand<<<BLOCK_NUM(header.nets), THREAD_NUM>>>(original, header.nets, 0, ++state->global_timestamp, -1, 0);
                graph::commit_via_demand<<<BLOCK_NUM(header.nets), THREAD_NUM>>>(original, header.nets, 0, state->global_timestamp, -1, 0);
            } else upload(original_ids, ids);
            ++state->global_timestamp;
            graph::refresh_cost(*state);
            // Distribute each cost cell once. Nearby DAGs read their local X
            // shard; boundary-crossing queries use the peer's NVSHMEM mapping.
            if (size > 1) for (int p = 0; p < size; ++p) {
                const int begin = p * shard_x, width = std::min(shard_x, view.X - begin);
                if (width <= 0) continue;
                cuda_check(cudaMemcpy2DAsync(view.peer_vcost[p], size_t(shard_x) * view.Y * sizeof(float),
                    state->device.vcost + size_t(begin) * view.Y, size_t(view.X) * view.Y * sizeof(float),
                    size_t(width) * view.Y * sizeof(float), view.L, cudaMemcpyDefault), "distribute via-cost shard");
                cuda_check(cudaMemcpy2DAsync(view.peer_presum[p], size_t(shard_x) * view.Y * sizeof(double),
                    state->device.presum + size_t(begin) * view.Y, size_t(view.X) * view.Y * sizeof(double),
                    size_t(width) * view.Y * sizeof(double), view.L, cudaMemcpyDefault), "distribute prefix shard");
            }
            copy(timestamps, state->device.timestamp, cells * sizeof(int));
            cuda_check(cudaMemsetAsync(wire, 0, cells * sizeof(int)), "clear final-batch wire differences");
            upload(pin_offsets, pins);
            initialize_ids<<<BLOCK_NUM(header.nets), THREAD_NUM>>>(all_ids, header.nets);
            // cudaMemset(1) is 0x01010101; kernels only distinguish zero/nonzero.
            cuda_check(cudaMemsetAsync(flags, command == Command::flt ? 1 : 0, (header.nets + 1) * sizeof(int)), "set compact rip-up flags");
        }
        sync();
#endif
        const double snapshot_ms = ms(start);
        if (count && command == Command::lshape) {
#ifndef GROTOR_MULTI_GRID
            upload(offsets, payload.offsets); upload(nodes, payload.nodes); upload(parents, payload.parents);
#endif
            Lshape_route::Lshape_route_warp<<<(count + 3) / 4, 128>>>(view, count, 0, offsets, nodes, parents, dist, from, ranges, header.stamp);
        } else if (count) {
#ifndef GROTOR_MULTI_GRID
            workspace->upload(topology, plans);
            workspace->assemble(topology, plans, plans.batches[0], view);
#endif
            Lshape_route_detour::precompute_cost_edge<<<BLOCK_NUM(plans.max_edges * 9), THREAD_NUM>>>(view, flt_view, plans.max_edges);
            Lshape_route_detour::route_flt_block<<<count, 128>>>(view, flt_view, workspace->descriptors, 0, workspace->depth_offsets, header.stamp);
        }
        cuda_check(cudaGetLastError(), "multi-GPU solver launch");
        sync();
#ifdef GROTOR_MULTI_GRID
        solve_event.record();
#endif
        const double solve_ms = ms(start) - snapshot_ms;
#ifdef GROTOR_MULTI_GRID
        // Cooperative wire commits were fused into each net's traceback. The
        // preceding collective completed every timestamp shard before vias.
        if (count) graph::commit_via_demand<<<BLOCK_NUM(count), THREAD_NUM>>>(view, count, 0, header.stamp);
        sync();
        commit_event.record();
        if (command == Command::lshape) {
            grid.costs(); grid.prefix(); sync();
        }
        end_event.record();
        cuda_check(cudaEventSynchronize(end_event.value), "complete measured batch");
        const auto finalize_start = Clock::now();
        const double local_timings[] = {end_event.since(begin_event),
            command == Command::lshape ? end_event.since(commit_event) : cost_event.since(begin_event),
            command == Command::lshape ? solve_event.since(begin_event) : solve_event.since(cost_event),
            command == Command::lshape ? commit_event.since(solve_event) : end_event.since(solve_event)};
        double rank_timings[8 * 4]{};
        mpi_check(MPI_Allgather(local_timings, 4, MPI_DOUBLE, rank_timings, 4, MPI_DOUBLE, MPI_COMM_WORLD));
        int critical = 0;
        for (int p = 1; p < size; ++p) if (rank_timings[4 * p] > rank_timings[4 * critical]) critical = p;
        const double* measured = rank_timings + 4 * critical;
        printf("MULTIGPU_RANK_TIMING stage=%s rank=%d elapsed_ms=%.6f\n",
            command == Command::lshape ? "lshape" : "flt", rank, local_timings[0]);
        if (!rank) {
            grid.gather(*state, routes, pins, partitions, command == Command::flt);
            auto compact = view;
            if (command == Command::flt) compact.ROUTE_PER_PIN = compact.ROUTE_PER_PIN_PHASE2;
            scatter_routes<<<header.nets, 128>>>(compact, state->device, original_ids, command == Command::flt);
        }
#else
        if (!rank) {
            const auto original = state->device;
            // Only root mutates the authoritative demand/BIT state. All wire
            // differences and endpoint timestamps are visible before via commit.
            auto commit = original;
            commit.pre_demand = wire; commit.timestamp = timestamps; commit.net_ids = all_ids;
            commit.pin_acc_num = pin_offsets; commit.pin_acc_num_phase2 = pin_offsets;
            commit.routes = routes; commit.routes_phase2 = routes; commit.ripup_flag = flags;
            state->device = commit;
            graph::batch_wire_update(*state, header.stamp);
            graph::commit_via_demand<<<BLOCK_NUM(header.nets), THREAD_NUM>>>(commit, header.nets, 0, header.stamp);
            // Scatter only used route entries, not each net's reserved capacity.
            auto compact = commit;
            if (command == Command::flt) compact.ROUTE_PER_PIN = compact.ROUTE_PER_PIN_PHASE2;
            scatter_routes<<<header.nets, 128>>>(compact, original, original_ids, command == Command::flt);
            copy(original.timestamp, timestamps, cells * sizeof(int));
            state->device = original;
        }
#endif
        sync();
#ifdef GROTOR_MULTI_GRID
        const double elapsed_ms = measured[0];
        const double finalization_ms = ms(finalize_start);
#else
        const double elapsed_ms = ms(start);
#endif
        printf("MULTIGPU_PARTITION stage=%s rank=%d nets=%d nodes=%zu\n",
            command == Command::lshape ? "lshape" : "flt", rank, count,
            command == Command::lshape ? payload.nodes.size() : plans.total_nodes);
        if (!rank) {
#ifdef GROTOR_MULTI_GRID
            printf("MULTIGPU_TIMING_POLICY stage=%s topology=%s maintenance=%s\n",
                command == Command::lshape ? "lshape" : "flt", command == Command::lshape ? "preparation" : "batch",
                command == Command::lshape ? "after-commit" : "before-solve");
            printf("MULTIGPU_BATCH stage=%s gpus=%d batch_index=%d batches=%d nets=%d elapsed_ms=%.6f preparation_ms=%.6f snapshot_ms=%.6f solve_ms=%.6f commit_ms=%.6f finalization_ms=%.6f end_to_end_ms=%.6f critical_rank=%d\n",
                command == Command::lshape ? "lshape" : "flt", size, header.batch_index, header.batches,
                header.nets, elapsed_ms, preparation_ms, measured[1], measured[2], measured[3], finalization_ms, ms(preparation_start), critical);
#else
            printf("MULTIGPU_BATCH stage=%s gpus=%d batch_index=%d batches=%d nets=%d elapsed_ms=%.6f preparation_ms=%.6f snapshot_ms=%.6f solve_ms=%.6f commit_ms=%.6f\n",
                command == Command::lshape ? "lshape" : "flt", size, header.batch_index, header.batches,
                header.nets, elapsed_ms, preparation_ms, snapshot_ms, solve_ms, elapsed_ms - snapshot_ms - solve_ms);
#endif
        }
    }

    void lshape(RouterState& state, const std::vector<int>& input_ids, const int* offsets,
                const int* nodes, const int* parents, int batch_index, int batches) {
        const auto preparation_start = Clock::now();
        Command command = Command::lshape; broadcast(command);
        auto ids = input_ids;
        const auto cuts = spatial_order(state, ids);
        Header header{command, state.device, int(ids.size()), state.global_timestamp + 1, batch_index, batches};
        std::vector<int> original_slot(state.nets.size());
        uint64_t pins = 0;
        for (int i = 0; i < input_ids.size(); ++i) { original_slot[input_ids[i]] = i; pins += state.nets[input_ids[i]].pins.size(); }
        header.route_ints = pins * state.device.ROUTE_PER_PIN + 1;
        std::vector<Payload> payloads(size);
        for (int p = 0; p < size; ++p) {
            auto& data = payloads[p]; data.begin = cuts[p];
            data.offsets.push_back(0);
            for (int i = cuts[p]; i < cuts[p + 1]; ++i) {
                const int slot = original_slot[ids[i]], begin = offsets[slot], end = offsets[slot + 1];
                data.nodes.insert(data.nodes.end(), nodes + begin, nodes + end);
                data.parents.insert(data.parents.end(), parents + begin, parents + end);
                data.offsets.push_back(data.nodes.size());
            }
        }
        run(&state, ids, &header, &payloads, command, preparation_start);
    }

    void flt_batch(RouterState& state, const flt::Topology& topology,
                   const std::vector<int>& input_ids, int batch_index, int batches) {
        const auto preparation_start = Clock::now();
        Command command = Command::flt; broadcast(command);
        auto ids = input_ids;
        const auto cuts = spatial_order(state, ids);
        Header header{command, state.device, int(ids.size()), state.global_timestamp + 2, batch_index, batches};
        uint64_t pins = 0;
        for (int id : ids) {
            pins += state.nets[id].pins.size();
        }
        header.route_ints = pins * state.device.ROUTE_PER_PIN_PHASE2 + 1;
        std::vector<Payload> payloads(size);
        for (int p = 0; p < size; ++p) {
            auto& data = payloads[p]; data.begin = cuts[p];
            int depths = 0;
            for (int i = cuts[p]; i < cuts[p + 1]; ++i) {
                const auto& span = topology.spans.at(topology.net_to_slot.at(ids[i]));
                const auto& arena = topology.arenas[span.lane];
                const int node = data.flt_nodes.size(), edge = data.flt_edges.size();
                data.flt_nodes.resize(node + span.nodes); data.flt_edges.resize(edge + span.edges);
                arena.nodes.copy(span.node_begin, span.nodes, data.flt_nodes.data() + node);
                arena.edges.copy(span.edge_begin, span.edges, data.flt_edges.data() + edge);
                data.descriptors.push_back({i, unsigned(node), unsigned(edge), node, edge, span.nodes, span.edges, depths, span.depths});
                depths += span.depths + 1;
            }
        }
        run(&state, ids, &header, &payloads, command, preparation_start);
    }
};
} // namespace grotor::multi

#pragma once
// GGR Z-shape recurrence used only as a dense, one-warp correctness oracle.
// BSD 3-Clause attribution: baseline/GGR/LICENSE and NOTICE.md.
namespace zshape_reference {
constexpr int infinity = 1000000000;
__device__ __forceinline__ int boundedWireCost(
    const int64_t *wireCostSum, int lhs, int rhs) {
    int64_t value = lhs < rhs ? wireCostSum[lhs] - wireCostSum[rhs]
                              : wireCostSum[rhs] - wireCostSum[lhs];
    if (value <= 0) return 0;
    if (value >= infinity / 2) return infinity / 2;
    return static_cast<int>(value);
}
__global__ void initMap(int *map, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) map[index] = infinity;
}
template <typename T>
__device__ void inline cudaSwap(T &a, T &b) {
    T c(a);
    a = b;
    b = c;
}
__device__ void viaSweep(int *map, int *prev, int *viaCost, int x, int y, int LAYER, int N, int DIRECTION) {
    map += 4 * LAYER * N * N;
    prev += 4 * LAYER * N * N;
    for (int i = 1; i < LAYER; i++) {
        int cur = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        prev[cur + LAYER * N * N] = cur;
    }
    for (int i = 2; i < LAYER; i++) {
        int cur = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int last = (i - 1) * N * N + (((i & 1) ^ DIRECTION) ? y * N + x : x * N + y);
        if (map[cur] > map[last] + viaCost[last])
            map[cur] = map[last] + viaCost[last], prev[cur] = prev[last], prev[cur + LAYER * N * N] = last;
    }
    for (int i = LAYER - 2; i >= 1; i--) {
        int cur = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
        int last = (i + 1) * N * N + (((i & 1) ^ DIRECTION) ? y * N + x : x * N + y);
        if (map[cur] > map[last] + viaCost[cur])
            map[cur] = map[last] + viaCost[cur], prev[cur] = prev[last], prev[cur + LAYER * N * N] = last;
    }
}
__global__ void cudaPatternRouteParallel(int *map,
                                         int *points,
                                         int64_t *wireCostSum,
                                         int *viaCost,
                                         int *prev,
                                         int *prev2,
                                         int *wires,
                                         int *vias,
                                         int *routes,
                                         int *gbpoints,
                                         int *gbpinRoutes,
                                         int LAYER,
                                         int N,
                                         int DIRECTION) {
    gbpoints += gbpoints[blockIdx.x];
    points += points[blockIdx.x];
    routes += points[0];
    if (threadIdx.x == 0) routes[0] = 1;
    __syncthreads();
    int node_cnt = points[1];
    points += 2;
    for (int t = 6 * (node_cnt - 1); t >= 0; t -= 6) {
        int tox = points[t] / N, toy = points[t] % N, minlayer = points[t + 1], maxlayer = points[t + 2];
        for (int j = 1; j <= 3; j++) {
            if (points[t + 2 + j] != -1) {
                int childPos = points[points[t + 2 + j]];
                int fromx = childPos / N, fromy = childPos % N;
                if (fromx != tox && fromy != toy) {
                    for (int a = min(fromx, tox) + threadIdx.x; a <= max(fromx, tox); a += blockDim.x) {
                        for (int i = 1; i < LAYER; i++) {
                            if ((i & 1) ^ DIRECTION) {
                                map[4 * LAYER * N * N + i * N * N + a * N + toy] = infinity;
                                map[4 * LAYER * N * N + i * N * N + a * N + fromy] = infinity;
                            } else {
                                map[4 * LAYER * N * N + i * N * N + toy * N + a] = infinity;
                                map[4 * LAYER * N * N + i * N * N + fromy * N + a] = infinity;
                            }
                        }
                    }
                    __syncthreads();
                    for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x) {
                        for (int i = 1; i < LAYER; i++) {
                            if ((i & 1) ^ DIRECTION) {
                                map[4 * LAYER * N * N + i * N * N + tox * N + b] = infinity;
                                map[4 * LAYER * N * N + i * N * N + fromx * N + b] = infinity;
                            } else {
                                map[4 * LAYER * N * N + i * N * N + b * N + tox] = infinity;
                                map[4 * LAYER * N * N + i * N * N + b * N + fromx] = infinity;
                            }
                        }
                    }
                    __syncthreads();
                    for (int i = 1; i < LAYER; i++) {
                        if ((i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromx * N + fromy;
                            for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x) {
                                if (b != fromy) {
                                    int cur = i * N * N + fromx * N + b;
                                    int cost = boundedWireCost(wireCostSum, last, cur);
                                    map[4 * LAYER * N * N + cur] = map[last] + cost;
                                    prev[4 * LAYER * N * N + cur] = last;
                                }
                            }
                        } else {
                            int last = i * N * N + fromy * N + fromx;
                            for (int b = min(fromx, tox) + threadIdx.x; b <= max(fromx, tox); b += blockDim.x) {
                                if (b != fromx) {
                                    int cur = i * N * N + fromy * N + b;
                                    int cost = boundedWireCost(wireCostSum, last, cur);
                                    map[4 * LAYER * N * N + cur] = map[last] + cost;
                                    prev[4 * LAYER * N * N + cur] = last;
                                }
                            }
                        }
                        __syncthreads();
                    }
                    __syncthreads();
                    for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x)
                        if (b != fromy) viaSweep(map, prev, viaCost, fromx, b, LAYER, N, DIRECTION);
                    __syncthreads();
                    for (int b = min(fromx, tox) + threadIdx.x; b <= max(fromx, tox); b += blockDim.x)
                        if (b != fromx) viaSweep(map, prev, viaCost, b, fromy, LAYER, N, DIRECTION);
                    __syncthreads();
                    for (int i = 1; i < LAYER; i++) {
                        if ((i & 1) ^ DIRECTION) {
                            for (int b = min(fromx, tox) + threadIdx.x; b <= max(fromx, tox); b += blockDim.x) {
                                int last = i * N * N + b * N + fromy, cur = i * N * N + b * N + toy;
                                int cost = boundedWireCost(wireCostSum, last, cur);
                                if (map[4 * LAYER * N * N + last] + cost < map[4 * LAYER * N * N + cur]) {
                                    map[4 * LAYER * N * N + cur] = map[4 * LAYER * N * N + last] + cost;
                                    prev[4 * LAYER * N * N + cur] = prev[4 * LAYER * N * N + last];
                                }
                            }
                        } else {
                            for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x) {
                                int last = i * N * N + b * N + fromx, cur = i * N * N + b * N + tox;
                                int cost = boundedWireCost(wireCostSum, last, cur);
                                if (map[4 * LAYER * N * N + last] + cost < map[4 * LAYER * N * N + cur]) {
                                    map[4 * LAYER * N * N + cur] = map[4 * LAYER * N * N + last] + cost;
                                    prev[4 * LAYER * N * N + cur] = prev[4 * LAYER * N * N + last];
                                }
                            }
                        }
                        __syncthreads();
                    }
                    __syncthreads();
                    for (int b = min(fromy, toy) + threadIdx.x; b <= max(fromy, toy); b += blockDim.x)
                        if (b != toy) viaSweep(map, prev, viaCost, tox, b, LAYER, N, DIRECTION);
                    __syncthreads();
                    for (int b = min(fromx, tox) + threadIdx.x; b <= max(fromx, tox); b += blockDim.x)
                        if (b != tox) viaSweep(map, prev, viaCost, b, toy, LAYER, N, DIRECTION);
                    __syncthreads();
                    for (int i = 1 + threadIdx.x; i < LAYER; i += blockDim.x) {
                        if ((i & 1) ^ DIRECTION) {
                            int cur = i * N * N + tox * N + toy;
                            for (int b = min(fromy, toy); b <= max(fromy, toy); b++) {
                                if (b != toy) {
                                    int last = i * N * N + tox * N + b;
                                    int cost = boundedWireCost(wireCostSum, last, cur);
                                    if (map[4 * LAYER * N * N + last] + cost < map[cur + j * LAYER * N * N]) {
                                        map[cur + j * LAYER * N * N] = map[4 * LAYER * N * N + last] + cost;
                                        prev[cur + j * LAYER * N * N] = prev[4 * LAYER * N * N + last];
                                        prev2[cur + j * LAYER * N * N] = prev[5 * LAYER * N * N + last];
                                    }
                                }
                            }
                        } else {
                            int cur = i * N * N + toy * N + tox;
                            for (int b = min(fromx, tox); b <= max(fromx, tox); b++) {
                                if (b != tox) {
                                    int last = i * N * N + toy * N + b;
                                    int cost = boundedWireCost(wireCostSum, last, cur);
                                    if (map[4 * LAYER * N * N + last] + cost < map[cur + j * LAYER * N * N]) {
                                        map[cur + j * LAYER * N * N] = map[4 * LAYER * N * N + last] + cost;
                                        prev[cur + j * LAYER * N * N] = prev[4 * LAYER * N * N + last];
                                        prev2[cur + j * LAYER * N * N] = prev[5 * LAYER * N * N + last];
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();
                } else if (fromx == tox) {
                    for (int i = 1 + threadIdx.x; i < LAYER; i += blockDim.x) {
                        if ((i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromx * N + fromy, cur = i * N * N + tox * N + toy;
                            int cost = boundedWireCost(wireCostSum, last, cur);
                            if (map[cur + j * LAYER * N * N] > map[last] + cost)
                                map[cur + j * LAYER * N * N] = map[last] + cost, prev[cur + j * LAYER * N * N] = last;
                        } else {
                            map[i * N * N + toy * N + tox + j * LAYER * N * N] = infinity;
                        }
                    }
                    __syncthreads();
                } else if (fromy == toy) {
                    for (int i = 1 + threadIdx.x; i < LAYER; i += blockDim.x) {
                        if (!(i & 1) ^ DIRECTION) {
                            int last = i * N * N + fromy * N + fromx, cur = i * N * N + toy * N + tox;
                            int cost = boundedWireCost(wireCostSum, last, cur);
                            if (map[cur + j * LAYER * N * N] > map[last] + cost) {
                                map[cur + j * LAYER * N * N] = map[last] + cost, prev[cur + j * LAYER * N * N] = last;
                            }
                        } else {
                            map[i * N * N + tox * N + toy + j * LAYER * N * N] = infinity;
                        }
                    }
                    __syncthreads();
                } else {
                    printf("ERROR: points on the same location!\n");
                }
                __syncthreads();
            }
            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int lower = 0; lower < LAYER; lower++) {
                int viaSum = 0, minCost = infinity, bestpos[3] = {0, 0, 0},
                    costs[3] = {
                        points[t + 3] == -1 ? 0 : infinity, points[t + 4] == -1 ? 0 : infinity, points[t + 5] == -1 ? 0 : infinity};
                for (int upper = lower; upper < LAYER; upper++) {
                    int idx = upper * N * N + (((upper & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                    for (int j = 0; j < 3; j++)
                        if (points[t + 3 + j] != -1) {
                            if (map[idx + (j + 1) * LAYER * N * N] < costs[j])
                                costs[j] = map[idx + (j + 1) * LAYER * N * N], bestpos[j] = upper;
                        }
                    if (1LL * costs[0] + costs[1] + costs[2] < minCost) minCost = costs[0] + costs[1] + costs[2];
                    if (minlayer == -1 || (lower <= minlayer && maxlayer <= upper)) {
                        for (int i = lower; i <= upper; i++) {
                            int index = i * N * N + (((i & 1) ^ DIRECTION) ? tox * N + toy : toy * N + tox);
                            if (map[index] > minCost + viaSum)
                                map[index] = minCost + viaSum,
                                prev[index] =
                                    (((lower * LAYER + upper) * LAYER + bestpos[2]) * LAYER + bestpos[1]) * LAYER +
                                    bestpos[0];
                        }
                    }
                    viaSum += viaCost[idx];
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int minTotalDist = infinity;
        for (int i = 0; i < LAYER; i++) {
            int x = points[0] / N, y = points[0] % N;
            int idx = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
            if (map[idx] < minTotalDist) {
                minTotalDist = map[idx];
                map[5 * LAYER * N * N + points[0]] = i;
            }
        }
        if (minTotalDist == infinity || minTotalDist < 0) {
            routes[0] = -1;
        } else {
            for (int t = 0; t < 6 * node_cnt; t += 6) {
                int t_gbPinId = gbpoints[t / 6];
                int x = points[t] / N, y = points[t] % N, layer = map[5 * LAYER * N * N + points[t]];
                int idx = layer * N * N + (((layer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
                int minlayer = prev[idx] / LAYER / LAYER / LAYER / LAYER,
                    maxlayer = prev[idx] / LAYER / LAYER / LAYER % LAYER;
                for (int i = minlayer; i < maxlayer; i++) {
                    int id = i * N * N + (((i & 1) ^ DIRECTION) ? x * N + y : y * N + x);
                    routes[routes[0]++] = id;
                    routes[routes[0]++] = -1;
                    vias[id]++;
                }
                if (t_gbPinId != -1) {
                    gbpinRoutes[t_gbPinId * 6 + 5] = maxlayer - minlayer;
                }
                int bestpos = prev[idx];
                for (int j = 0; j < 3; j++) {
                    if (points[t + 3 + j] == -1) break;
                    int entry_routeId = -1, last_routeId = -1;
                    int child = points[points[t + 3 + j]];
                    int entrylayer = bestpos % LAYER,
                        entry = entrylayer * N * N + (((entrylayer & 1) ^ DIRECTION) ? x * N + y : y * N + x);
                    bestpos /= LAYER;
                    int last = prev[(j + 1) * LAYER * N * N + entry];
                    int lastlayer = last / N / N, lastx = last / N % N, lasty = last % N;
                    if (!(lastlayer & 1) ^ DIRECTION) cudaSwap(lastx, lasty);
                    if (lastx != x && lasty != y) {
                        int mid = prev2[(j + 1) * LAYER * N * N + entry];
                        int midlayer = mid / N / N, midx = mid / N % N, midy = mid % N;
                        if (!(midlayer & 1) ^ DIRECTION) cudaSwap(midx, midy);
                        int p1 = ((entrylayer & 1) ^ DIRECTION) ? entry - y + midy : entry - x + midx;
                        int p3 = ((midlayer & 1) ^ DIRECTION) ? mid - midy + lasty : mid - midx + lastx;
                        int p4 = ((lastlayer & 1) ^ DIRECTION) ? last - lasty + midy : last - lastx + midx;
                        if (entry != p1) {
                            int encodeId = entry <= p1 ? routes[0] : -routes[0];
                            if (entry_routeId == -1) entry_routeId = encodeId;
                            last_routeId = encodeId;
                            routes[routes[0]++] = min(entry, p1);
                            routes[routes[0]++] = max(entry, p1) - min(entry, p1);
                            for (int i = min(entry, p1); i < max(entry, p1); i++) wires[i]++;
                        }
                        if (p1 != mid) {
                            for (int i = min(entrylayer, midlayer); i < max(entrylayer, midlayer); i++) {
                                int id = i * N * N + (((i & 1) ^ DIRECTION) ? midx * N + midy : midy * N + midx);
                                routes[routes[0]++] = id;
                                routes[routes[0]++] = -1;
                                vias[id]++;
                            }
                        }
                        if (mid != p3) {
                            int encodeId = mid <= p3 ? routes[0] : -routes[0];
                            if (entry_routeId == -1) entry_routeId = encodeId;
                            last_routeId = encodeId;
                            routes[routes[0]++] = min(mid, p3);
                            routes[routes[0]++] = max(mid, p3) - min(mid, p3);
                            for (int i = min(mid, p3); i < max(mid, p3); i++) wires[i]++;
                        }
                        if (p3 != p4) {
                            int tempx = p3 / N % N, tempy = p3 % N;
                            if (!(p3 / N / N & 1) ^ DIRECTION) cudaSwap(tempx, tempy);
                            for (int i = min(midlayer, lastlayer); i < max(midlayer, lastlayer); i++) {
                                int id = i * N * N + (((i & 1) ^ DIRECTION) ? tempx * N + tempy : tempy * N + tempx);
                                routes[routes[0]++] = id;
                                routes[routes[0]++] = -1;
                                vias[id]++;
                            }
                        }
                        if (p4 != last) {
                            int encodeId = p4 <= last ? routes[0] : -routes[0];
                            if (entry_routeId == -1) entry_routeId = encodeId;
                            last_routeId = encodeId;
                            routes[routes[0]++] = min(p4, last);
                            routes[routes[0]++] = max(p4, last) - min(p4, last);
                            for (int i = min(p4, last); i < max(p4, last); i++) wires[i]++;
                        }
                    } else {
                        int encodeId = entry <= last ? routes[0] : -routes[0];
                        if (entry_routeId == -1) entry_routeId = encodeId;
                        last_routeId = encodeId;
                        routes[routes[0]++] = min(last, entry);
                        routes[routes[0]++] = max(last, entry) - min(last, entry);
                        for (int i = min(last, entry); i < max(last, entry); i++) wires[i]++;
                    }
                    map[5 * LAYER * N * N + child] = lastlayer;
                    if (t_gbPinId != -1) {
                        gbpinRoutes[t_gbPinId * 6 + 1 + gbpinRoutes[t_gbPinId * 6]] = entry_routeId;
                        gbpinRoutes[t_gbPinId * 6]++;
                        if (gbpinRoutes[t_gbPinId * 6] > 4) {
                            printf("Error: numRoutes for t_gbpin d%d\n", gbpinRoutes[t_gbPinId * 6]);
                        }
                    }
                    int c_gbPinId = gbpoints[points[t + 3 + j] / 6];
                    if (c_gbPinId != -1) {
                        gbpinRoutes[c_gbPinId * 6 + 1 + gbpinRoutes[c_gbPinId * 6]] = -1 * last_routeId;
                        gbpinRoutes[c_gbPinId * 6]++;
                        if (gbpinRoutes[c_gbPinId * 6] > 4) {
                            printf("Error: numRoutes for c_gbpin %d\n", gbpinRoutes[c_gbPinId * 6]);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}
}

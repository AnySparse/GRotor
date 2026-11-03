#include "routing/zshape.cuh"
#include "routing/zshape_topology.hpp"
#include "zshape_reference.cuh"
#include <random>

int main() {
    using grotor::cuda_check;
    {
        std::vector<cudb::net> nets(5);
        const int boxes[5][4]={{0,2,0,2},{2,4,2,4},{5,6,0,1},{8,9,0,1},{0,1,8,9}};
        for (int i=0; i<5; ++i) {
            nets[i].minx=boxes[i][0]; nets[i].maxx=boxes[i][1];
            nets[i].miny=boxes[i][2]; nets[i].maxy=boxes[i][3];
        }
        const auto batches=grotor::zshape::generate_bbox_batches({0,1,2,3,4},nets,10,10,2);
        const auto reversed=grotor::zshape::generate_bbox_batches({4,3,2,1,0},nets,10,10,2);
        if (batches!=reversed) throw std::runtime_error("Bounding-box batches depend on input order");
        std::vector<int> seen(5);
        for (const auto& batch : batches) {
            if (batch.empty() || batch.size()>2) throw std::runtime_error("Invalid bounding-box batch size");
            for (int id : batch) ++seen[id];
            for (int i=0; i<batch.size(); ++i) for (int j=0; j<i; ++j) {
                const auto& a=nets[batch[i]]; const auto& b=nets[batch[j]];
                if (a.minx<=b.maxx && b.minx<=a.maxx && a.miny<=b.maxy && b.miny<=a.maxy)
                    throw std::runtime_error("Inclusive bounding boxes overlap in one batch");
            }
        }
        for (int count : seen) if (count!=1) throw std::runtime_error("Missing or duplicate net task");
    }
    std::mt19937 random(271828);
    for (int direction : {0, 1}) {
        cudb::RouterState state;
        auto& v = state.device;
        v.L=5; v.X=48; v.Y=40; v.XY=48; v.DIR=direction;
        const int n=48, layers=v.L+1, grid=layers*n*n, cells=v.L*v.X*v.Y;
        state.memory.allocate(&v.presum, cells*sizeof(double));
        state.memory.allocate(&v.vcost, cells*sizeof(float));
        grotor::DeviceMemory mem;
        int *map=nullptr, *prev=nullptr, *map_ref=nullptr, *prev_ref=nullptr,
            *routes=nullptr, *routes_ref=nullptr, *points=nullptr, *gb=nullptr,
            *wires=nullptr, *vias=nullptr, *via_cost=nullptr;
        int64_t* prefix=nullptr;
        mem.allocate(&map, 6ULL*grid*sizeof(int)); mem.allocate(&prev, 16ULL*grid*sizeof(int));
        mem.allocate(&map_ref, 6ULL*grid*sizeof(int)); mem.allocate(&prev_ref, 16ULL*grid*sizeof(int));
        mem.allocate(&routes, 5*2048*sizeof(int)); mem.allocate(&routes_ref, 5*2048*sizeof(int));
        mem.allocate(&points, 4096*sizeof(int)); mem.allocate(&gb, 4096*sizeof(int));
        mem.allocate(&wires, grid*sizeof(int)); mem.allocate(&vias, grid*sizeof(int));
        mem.allocate(&prefix, grid*sizeof(int64_t)); mem.allocate(&via_cost, grid*sizeof(int));
        cuda_check(cudaMemset(map, 0xa5, 6ULL*grid*sizeof(int)), "poison reusable map");
        for (int trial=0; trial<80; ++trial) {
            std::vector<double> sums(cells);
            std::vector<float> vc(cells);
            std::vector<int64_t> dense(grid, 0);
            std::vector<int> via_dense(grid, 1000000000);
            for (int l=0;l<v.L;++l) {
                const bool along_y=(l&1)^direction;
                const int tracks=along_y?v.X:v.Y, len=along_y?v.Y:v.X;
                for (int track=0;track<tracks;++track) {
                    int64_t sum=0;
                    for (int p=0;p<len;++p) {
                        const int x=along_y?track:p, y=along_y?p:track;
                        const int idx=l*v.X*v.Y+x*v.Y+y;
                        sums[idx]=sum;
                        dense[(l+1)*n*n+track*n+p]=-sum;
                        vc[idx]=1+random()%100;
                        via_dense[(l+1)*n*n+track*n+p]=int(vc[idx]);
                        sum += 1 + (trial%9==0 ? random()%10000000 : random()%1000);
                    }
                }
            }
            std::vector<cudb::net> nets(5);
            for (int k=0;k<5;++k) {
                const int x=2+k*8, y=3+(trial+k)%3;
                auto pos=[&](int a,int b,int l) { return l*v.X*v.Y+a*v.Y+b; };
                auto& net=nets[k];
                net.pins={pos(x,y,trial%v.L), pos(x+4,y+5,(trial+k+2)%v.L)};
                net.rsmt={{1},{0},{net.pins[0],net.pins[1]}};
                if (k==1) {
                    net.pins.push_back(pos(x+6,y+1,(trial+3)%v.L));
                    net.rsmt={{1},{0,2,3},{1},{1},{net.pins[0],pos(x+2,y+2,v.L),net.pins[1],net.pins[2]}};
                }
                if (k==2) net.rsmt.back()[1]=pos(x,y+5,(trial+2)%v.L);
                if (k==3) net.rsmt.back()[1]=pos(x+4,y,(trial+2)%v.L);
            }
            const std::vector<int> offsets={0,2048,4096,6144,8192};
            const int count=1+trial%5;
            std::vector<int> ids(count); std::iota(ids.begin(),ids.end(),0);
            auto encoded=grotor::zshape::encode_batch(ids,nets,offsets,v.X,v.Y,v.L);
            auto upload=[](auto* dst, const auto& src) {
                cuda_check(cudaMemcpy(dst,src.data(),src.size()*sizeof(src[0]),cudaMemcpyHostToDevice),"fixture upload");
            };
            upload(v.presum,sums); upload(v.vcost,vc); upload(prefix,dense); upload(via_cost,via_dense);
            upload(points,encoded.points); upload(gb,encoded.gbpoints);
            cuda_check(cudaMemset(wires,0,grid*sizeof(int)),"clear reference demand");
            cuda_check(cudaMemset(vias,0,grid*sizeof(int)),"clear reference vias");
            zshape_reference::initMap<<<(6*grid+255)/256,256>>>(map_ref,6*grid);
            zshape_reference::cudaPatternRouteParallel<<<count,32>>>(map_ref,points,prefix,via_cost,
                prev_ref,prev_ref+10*grid,wires,vias,routes_ref,gb,nullptr,layers,n,direction^1);
            grotor::zshape::reset_nodes<<<count,128>>>(map,points,layers,n,direction^1,count);
            grotor::zshape::solve<4><<<(count+3)/4,128>>>(map,points,{v,n},{v,n},prev,prev+10*grid,
                routes,gb,nullptr,layers,n,direction^1,count);
            cuda_check(cudaDeviceSynchronize(),"compare Z-shape kernels");
            std::vector<int> a(5*2048), b(a.size());
            cuda_check(cudaMemcpy(a.data(),routes,a.size()*sizeof(int),cudaMemcpyDeviceToHost),"download optimized");
            cuda_check(cudaMemcpy(b.data(),routes_ref,b.size()*sizeof(int),cudaMemcpyDeviceToHost),"download original");
            for (int k=0;k<count;++k) {
                const int base=offsets[k];
                if (a[base]!=b[base] || a[base]<=0 || a[base]>2048 ||
                    !std::equal(a.begin()+base,a.begin()+base+a[base],b.begin()+base)) {
                    fprintf(stderr,"Z-shape mismatch direction=%d trial=%d net=%d words=%d/%d\n",
                        direction,trial,k,a[base],b[base]); return 1;
                }
            }
        }
    }
    puts("Z-shape dense/original vs cooperative/reused-scratch: 160 fixtures passed");
}

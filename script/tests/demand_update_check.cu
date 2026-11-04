#include "routing/demand.cuh"
#include <numeric>

namespace {
__global__ void stage_wire_counts(cudb::RoutingDeviceView view, int count, int offset, int stamp, int phase) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    int id = view.net_ids[slot + offset];
    bool second = phase != 0 && view.ripup_flag[id];
    const int* routes = second ? view.routes_phase2 + view.pin_acc_num_phase2[id] * view.ROUTE_PER_PIN_PHASE2
                               : view.routes + view.pin_acc_num[id] * view.ROUTE_PER_PIN;
    const int X=view.X,Y=view.Y;
    for (int i=1;i<routes[0];i+=2) {
        int a=routes[i],b=routes[i+1];
        if(a/(X*Y)!=b/(X*Y))continue;
        graph::atomic_add_unit_demand_wire_segment(view,a/(X*Y),min(a/Y%X,b/Y%X),max(a/Y%X,b/Y%X),min(a%Y,b%Y),max(a%Y,b%Y),stamp);
    }
}
struct Fixture {
    cudb::RouterState state;
    static constexpr int count=37,words=128;
    template<class T> void alloc(T*& p,size_t n){state.memory.allocate(&p,n*sizeof(T));}
    template<class T> void upload(T* p,const std::vector<T>& v){grotor::cuda_check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"upload demand fixture");}
    template<class T> std::vector<T> read(const T* p,size_t n){std::vector<T> v(n);grotor::cuda_check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"read demand fixture");return v;}
    explicit Fixture(int dir,bool zero) {
        auto& v=state.device;v.L=3;v.X=37;v.Y=65;v.XY=65;v.DIR=dir;
        v.ROUTE_PER_PIN=v.ROUTE_PER_PIN_PHASE2=words;
        const int cells=v.L*v.X*v.Y;
        alloc(v.routes,count*words);alloc(v.routes_phase2,count*words);
        alloc(v.pin_acc_num,count+1);alloc(v.pin_acc_num_phase2,count+1);alloc(v.net_ids,count);alloc(v.ripup_flag,count);
        alloc(v.demand,cells);alloc(v.pre_demand,cells);alloc(v.timestamp,cells);alloc(v.statistics,1);
        alloc(v.x_edge_len,v.X);alloc(v.y_edge_len,v.Y);
        alloc(v.dirty_wire,(cells+31)/32);alloc(v.dirty_via,(cells+31)/32);
        std::vector<int> pins(count+1),ids(count),xlen(v.X),ylen(v.Y),tracks;
        std::vector<int> ripup(count);
        for(int i=0;i<count;++i){pins[i]=i;ids[i]=(11*i+7)%count;ripup[i]=i&1;}pins.back()=count;
        upload(v.pin_acc_num,pins);upload(v.pin_acc_num_phase2,pins);upload(v.net_ids,ids);
        upload(v.ripup_flag,ripup);
        std::iota(xlen.begin(),xlen.end(),1);std::iota(ylen.begin(),ylen.end(),3);upload(v.x_edge_len,xlen);upload(v.y_edge_len,ylen);
        for(int l=0;l<v.L;++l)for(int t=0;t<(((l&1)^dir)?v.X:v.Y);++t)tracks.push_back(l*v.XY+t);
        v.all_track_cnt=tracks.size();alloc(v.idx2track,tracks.size());upload(v.idx2track,tracks);
        for(int phase=0;phase<2;++phase){
            std::vector<int> routes(count*words);
            for(int id=0;id<count;++id){
                int* r=routes.data()+id*words;int end=1;
                auto point=[&](int l,int x,int y){return(l*v.X+x)*v.Y+y;};
                auto segment=[&](int a,int b){r[end++]=a;r[end++]=b;};
                for(int l=0;l<v.L;++l){
                    bool along_y=(l&1)^dir;
                    int x=(id*3+phase)%v.X,y=(id*7+phase)%v.Y;
                    if(along_y){segment(point(l,x,v.Y-1),point(l,x,0));segment(point(l,x,4),point(l,x,11));}
                    else{segment(point(l,v.X-1,y),point(l,0,y));segment(point(l,3,y),point(l,8,y));}
                    if(zero)segment(point(l,x,y),point(l,x,y));
                }
                segment(point(0,0,0),point(2,0,0));
                segment(point(2,v.X-1,v.Y-1),point(0,v.X-1,v.Y-1));
                segment(point(0,5,7),point(2,5,7));
                r[0]=end;
            }
            upload(phase?v.routes_phase2:v.routes,routes);
        }
    }
    void reset(){auto& v=state.device;size_t cells=size_t(v.L)*v.X*v.Y;upload(v.demand,std::vector<float>(cells,2048.5f));
        grotor::cuda_check(cudaMemset(v.pre_demand,0,cells*sizeof(int)),"reset pre-demand");
        grotor::cuda_check(cudaMemset(v.timestamp,0,cells*sizeof(int)),"reset timestamps");
        grotor::cuda_check(cudaMemset(v.dirty_wire,0,((cells+31)/32)*sizeof(unsigned)),"reset wire dirtiness");
        grotor::cuda_check(cudaMemset(v.dirty_via,0,((cells+31)/32)*sizeof(unsigned)),"reset via dirtiness");
        grotor::cuda_check(cudaMemset(v.statistics,0,sizeof(*v.statistics)),"reset demand statistics");}
};
void check(int dir,bool zero){
    Fixture f(dir,zero);auto& v=f.state.device;int cells=v.L*v.X*v.Y;
    for(auto span:std::vector<std::pair<int,int>>{{0,37},{3,31},{36,1},{7,0}})for(int phase:{-1,0})for(int K:{1,-1}){
        std::vector<float> demand;std::vector<int> stamps;std::vector<unsigned> wire,via;cudb::RoutingStatistics stats{};
        for(int method=0;method<4;++method){
            if(method==3&&(zero||K<0))continue;
            f.reset();
            if(method==0){graph::commit_wire_demand<<<1,64>>>(v,span.second,span.first,7,K,phase);graph::commit_via_demand<<<1,64>>>(v,span.second,span.first,7,K,phase);}
            else if(method<3)graph::commit_batch_demand(v,span.second,span.first,7,K,phase);
            else{stage_wire_counts<<<1,64>>>(v,span.second,span.first,7,phase);graph::batch_wire_update(f.state,7);graph::commit_via_demand<<<1,64>>>(v,span.second,span.first,7,K,phase);}
            auto d=f.read(v.demand,cells);auto t=f.read(v.timestamp,cells);auto w=f.read(v.dirty_wire,(cells+31)/32);auto z=f.read(v.dirty_via,(cells+31)/32);auto s=f.read(v.statistics,1)[0];
            if(!method){demand=d;stamps=t;wire=w;via=z;stats=s;}
            else if(d!=demand||t!=stamps||w!=wire||z!=via||s.total_wirelength!=stats.total_wirelength||s.total_via_count!=stats.total_via_count)
                throw std::runtime_error("Cooperative demand commit differs from route/grid oracle: method="+std::to_string(method));
        }
    }
    printf("PASS DEMAND_COMMIT direction=%d zero_segments=%d exact_demand_timestamps_dirty_bits=1\n",dir,int(zero));
}
}
int main(){try{for(int dir:{0,1})for(bool zero:{false,true})check(dir,zero);puts("DEMAND_COMMIT_CHECK_PASSED");return 0;}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}

#include "batching/generate_batches_cpu.hpp"
#include "batching/generate_batches_rtcore_synchronous.cuh"
#include <random>

namespace {
void check(const char* name,cudb::RouterState& state,std::vector<int> order,int expected_rounds=-1,bool chain=false) {
    state.rtcore_geometry.begin(order,state.nets.size(),3);
    for (int i=0;i<order.size();++i)state.rtcore_geometry.emit(i,i%3,state.nets[order[i]],state.device.Y);
    const auto expected=cudb::generate_batches_rsmt(state,order,1000000,1);
    std::vector<unsigned long long> prior_without,prior_with;
    for (int repeat=0;repeat<2;++repeat) {
        auto result=grotor::rtcore::synchronous::generate_batches_rtcore_synchronous(state,order,1000000,GROTOR_RTCORE_SYNC_PTX_PATH);
        if (!result.exact_assignment_rounds || !result.exact_batch_ids || result.batches!=expected)
            throw std::runtime_error("Fig7 does not match CPU colors/history");
        if (expected_rounds>=0 && result.with_blocker.size()!=expected_rounds)
            throw std::runtime_error("Fig7 round count must follow dependency depth");
        std::vector<unsigned long long> without,with;
        for (int i=0;i<result.with_blocker.size();++i) {
            const auto a=result.without_blocker[i],b=result.with_blocker[i];
            if (a.before!=b.before || a.remaining!=b.remaining || b.rays>a.rays)
                throw std::runtime_error("Fig7 round histories are inconsistent");
            without.push_back(a.rays);with.push_back(b.rays);
            if (chain && (a.rays!=order.size()-i || b.rays!=(i ? 1:order.size()))) {
                fprintf(stderr,"chain round=%d without=%llu with=%llu expected_without=%zu before=%d after=%d\n",i,a.rays,b.rays,order.size()-i,a.before,a.remaining);
                throw std::runtime_error("Fig7 chain trace counts do not match independent prediction");
            }
        }
        if (repeat && (without!=prior_without || with!=prior_with))
            throw std::runtime_error("Fig7 ray counts must be repeatable");
        prior_without=without;prior_with=with;
    }
    printf("PASS RTCORE_BLOCKER case=%s rounds=%zu repeated_counts_exact=1 cpu_lists_equal=1\n",name,prior_without.size());
}
std::vector<int> order(int n){std::vector<int> v(n);std::iota(v.begin(),v.end(),0);return v;}
}
int main(){
    try {
        {
            cudb::RouterState s;s.device.X=s.device.Y=64;s.nets.resize(12);
            for(int i=0;i<12;++i)s.nets[i].points={20*64+20};
            check("clique",s,order(12),12,true);
        }
        {
            cudb::RouterState s;s.device.X=s.device.Y=64;s.nets.resize(17);
            for(int i=0;i<17;++i)s.nets[i].points={(i+2)*64+20};
            check("chain-two-colors-many-rounds",s,order(17),17,true);
        }
        {
            cudb::RouterState s;s.device.X=s.device.Y=128;s.nets.resize(257);
            for(int i=0;i<257;++i)s.nets[i].points={(i/32*4+1)*128+i%32*4+1};
            check("counter-bin-tail",s,order(257),1);
        }
        for(unsigned seed=1;seed<=2;++seed){
            cudb::RouterState s;s.device.X=s.device.Y=64;s.nets.resize(97);
            std::mt19937 random(seed);
            for(auto& n:s.nets){
                for(int p=0;p<7;++p)n.points.push_back(random()%4096);
                n.points.push_back(n.points.front());
                for(int k=0;k<3;++k){int x=random()%64,y=random()%64,z=random()%64;
                    n.rsmt_h_segments.emplace_back(std::min(x,z)*64+y,std::max(x,z)*64+y);
                    n.rsmt_v_segments.emplace_back(x*64+std::min(y,z),x*64+std::max(y,z));
                }
            }
            auto ids=order(97);std::shuffle(ids.begin(),ids.end(),random);check("random-and-reordered",s,ids);
        }
        puts("RTCORE_RAY_CHECK_PASSED");return 0;
    }catch(const std::exception& e){fprintf(stderr,"Fig7 check failed: %s\n",e.what());return 1;}
}

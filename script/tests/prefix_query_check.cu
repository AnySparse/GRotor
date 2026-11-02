#include "experiments/prefix_update.cuh"
#include <vector>

namespace {
__global__ void check_queries(const double* tree,int length,int stride,int* errors) {
    const int first=blockIdx.x*blockDim.x+threadIdx.x;
    if(first>=length)return;
    for(int last=first;last<length;++last){
        unsigned old_reads=0,new_reads=0;
        const double a=grotor::experiments::bit_range_unpruned(tree,stride,first,last,old_reads);
        const double b=grotor::experiments::bit_range_pruned(tree,stride,first,last,new_reads);
        // Exactly representable integer costs make the arithmetic oracle exact.
        double expected=0;for(int i=first;i<last;++i)expected+=(i*37+5)%101;
        unsigned expected_reads=__popc(unsigned(first+1))+__popc(unsigned(last+1));
        const int stop=grotor::experiments::common_prefix_node(first,last);
        unsigned common_reads=__popc(unsigned(stop+1));
        if(a!=expected || b!=expected || old_reads!=expected_reads || new_reads!=old_reads-2*common_reads)
            atomicAdd(errors,1);
    }
}
}
int main(){
    try{
        grotor::DeviceMemory memory;
        int* errors=nullptr;memory.allocate(&errors,sizeof(int));
        for(int length:{1,2,3,17,64,129,1025})for(int stride:{1,7}){
            std::vector<double> tree(length*stride);
            for(int i=0;i<length;++i){
                const int begin=i&(i+1);
                for(int j=begin;j<=i;++j)tree[i*stride]+=j?((j-1)*37+5)%101:0;
            }
            double* device=nullptr;memory.allocate(&device,tree.size()*sizeof(double));
            grotor::cuda_check(cudaMemcpy(device,tree.data(),tree.size()*sizeof(double),cudaMemcpyHostToDevice),"query fixture upload");
            grotor::cuda_check(cudaMemset(errors,0,sizeof(int)),"query fixture errors reset");
            check_queries<<<(length+127)/128,128>>>(device,length,stride,errors);
            int count=0;grotor::cuda_check(cudaMemcpy(&count,errors,sizeof(int),cudaMemcpyDeviceToHost),"query fixture readback");
            if(count)throw std::runtime_error("BIT range value or exact access count mismatch");
            memory.release(device);
        }
        printf("PREFIX_QUERY_CHECK_PASSED values=exact counts=exact empty_intervals=1 strided_tracks=1\n");return 0;
    }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}
}

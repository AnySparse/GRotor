# InstantGR

InstantGR is a GPU-accelerated global routing tool. 

Check out the following paper for more details.

* Liang Xiao, Shiju Lin, Jinwei Liu, Qinkai Duan, Tsung-Yi Ho, and Evangeline Young, ["InstantGR: Scalable GPU Parallelization for 3-D Global Routing"](https://ieeexplore.ieee.org/document/11015529), IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2025

* Shiju Lin, Liang Xiao, Jinwei Liu and Evangeline Young, ["InstantGR: Scalable GPU Parallelization for Global Routing"](https://shijulin.github.io/files/1239_Final_Manuscript.pdf), ACM/IEEE International Conference on Computer-Aided Design (ICCAD), New Jersey, USA, Oct 27–31, 2024.

## Compile
```bash
cd src
nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=sm_80
```
You may want to change `-arch=sm_80` according to your GPU. For example, if you are running InstantGR on NVIDIA RTX 3090, you need to change it to `-arch=sm_86`.

## Run
```bash
cd run
./InstantGR -cap <cap_file_path> -net <net_file_path> -out <output_path>
```
Example:
```bash
./InstantGR -cap ../benchmarks/mempool_tile_rank.cap -net ../benchmarks/mempool_tile_rank.net -out mempool_tile_rank.out
```

## Evaluate
```bash
cd run
g++ -o evaluator evaluator.cpp -O3 -std=c++17 #compile the evaluator
./evaluator <cap_file_path> <net_file_path> <output_path> # run the evaluator
```
Example:
```bash
./evaluator ../benchmarks/mempool_tile_rank.cap ../benchmarks/mempool_tile_rank.net mempool_tile_rank.out
```

## Benchmarks

### Host Memory Saving

With `INSTANTGR2_HOST_MEMORY_SAVING=ON` (explicitly enabled), this vendored baseline
releases pin-access candidates after selecting pins,
subnet staging after moving records into the routing database, and host capacity
and transfer buffers after their device uploads. Output records are released
after they have been written. For designs with more than 16 million routing
tasks, phase-one output completes and finished subnet topology is released
before detour construction; freed heap pages are returned to the OS on glibc.
Topology-only congestion buffers are released after their workers join.

These changes manage host storage lifetimes. They retain the original routing
kernels, full cost/prefix updates, batch construction and GPU timing scopes.
`--batching dual --sparsity 1` still times GPU batching and uses the CPU batch
plan for routing. End-to-end wall time includes all cleanup and preparation
waits. Logs identify the policy with `HOST_MEMORY_POLICY` and the large-design
boundary with `HOST_PREPARATION_BARRIER`.

On A100, a complete `tera_cluster` run with `--batching dual --sparsity 1`
reduced maximum process RSS from 218.74 GiB to 166.43 GiB. The modified run
completed with a cgroup limit of 180,000,000,000 bytes and swap disabled,
with no OOM events and memory saving enabled. These are local full-run measurements, including output;
RSS and cgroup memory accounting are separate metrics.

The default is `INSTANTGR2_HOST_MEMORY_SAVING=OFF`. This retains host
preprocessing/output records, skips heap trimming and the
large-design preparation barrier, and overlaps phase-one output with detour.
Normal final cleanup and allocation-size/leak fixes remain enabled. It can use
more than 180 GB on `tera_cluster`; this option itself sets no OS memory limit.
GRotor has a separate `GROTOR_HOST_MEMORY_SAVING` option.

For memory-constrained runs, configure with
`-DINSTANTGR2_HOST_MEMORY_SAVING=ON` and rebuild `instantgr2_reference`.
Existing CMake build directories preserve cached values; explicitly set `OFF`
to disable a previously enabled policy, or use a fresh build directory.

For the standalone nvcc build, append `-DINSTANTGR2_HOST_MEMORY_SAVING=1` to the
compile command above to enable memory saving. Omitting it defaults to disabled.
The running binary prints `HOST_MEMORY_POLICY saving=0|1 bounded_preparation=0|1`.

```bash
./InstantGR -cap /path/to/tera_cluster.cap -net /path/to/tera_cluster.net \
  -out tera_cluster.out --batching dual --sparsity 1
```

The ISPD2024 benchmarks can be downloaded [here](https://drive.google.com/drive/folders/1bon65UEAx8cjSvVhYJ-lgC8QMDX0fvUm).
We provide a small case `mempool_tile_rank` in the folder `benchmarks` for simple testing.

## Contact
[Shiju Lin](https://shijulin.github.io/) (email: sjlin@cse.cuhk.edu.hk)

## License
BSD 3-Clause License

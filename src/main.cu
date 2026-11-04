#include "pipeline.hpp"
#ifdef GROTOR_CUGR2_ROUTING
#include "pipeline_cugr2.hpp"
#endif
#ifdef GROTOR_GGR_ROUTING
#include "pipeline_ggr.hpp"
#endif

#include <filesystem>
#include <future>

#ifndef G_ROTOR_RESOURCE_DIR
#error "G_ROTOR_RESOURCE_DIR must point to the FLUTE lookup tables"
#endif

int main(int argc, char** argv) {
    try {
#ifdef GROTOR_ENABLE_NVSHMEM
        grotor::multi::Session multi_session(argc, argv);
        if (multi_session.rank) {
            multi_session.serve();
            return EXIT_SUCCESS;
        }
#endif
        std::string cap, net, output, lut_dir = G_ROTOR_RESOURCE_DIR;
        int sparsity = 1;
        bool profile_stages = false;
        bool stop_after_lshape = false;
        bool rtcore_batching = false, validate_rtcore = false;
        if (argc < 7 || argc % 2 != 1)
            throw std::invalid_argument("Usage: GRotor -cap FILE -net FILE -out FILE [--batching cpu|rtcore] [--validate-batches 0|1] [--sparsity N] [--lut-dir DIR] [--profile-stages 0|1] [--stop-after lshape|routing]");
        for (int i = 1; i < argc; i += 2) {
            const std::string option = argv[i], value = argv[i + 1];
            if (option == "-cap") cap = value;
            else if (option == "-net") net = value;
            else if (option == "-out") output = value;
            else if (option == "--lut-dir") lut_dir = value;
            else if (option == "--batching") {
                if (value != "cpu" && value != "rtcore")
                    throw std::invalid_argument("--batching expects cpu or rtcore");
                rtcore_batching = value == "rtcore";
            }
            else if (option == "--validate-batches") {
                if (value != "0" && value != "1")
                    throw std::invalid_argument("--validate-batches expects 0 or 1");
                validate_rtcore = value == "1";
            }
            else if (option == "--profile-stages") {
                if (value != "0" && value != "1")
                    throw std::invalid_argument("--profile-stages expects 0 or 1");
                profile_stages = value == "1";
            }
            else if (option == "--stop-after") {
                if (value != "lshape" && value != "routing")
                    throw std::invalid_argument("--stop-after expects lshape or routing");
                stop_after_lshape = value == "lshape";
            }
            else if (option == "--sparsity") {
                size_t consumed = 0;
                sparsity = std::stoi(value, &consumed);
                if (consumed != value.size() || sparsity < 1)
                    throw std::invalid_argument("Sparsity must be a positive integer");
            } else throw std::invalid_argument("Unknown option: " + option);
        }
        if (cap.empty() || net.empty() || output.empty())
            throw std::invalid_argument("-cap, -net and -out are required");
        if (validate_rtcore && !rtcore_batching)
            throw std::invalid_argument("--validate-batches requires --batching rtcore");
        if (rtcore_batching && sparsity != 1)
            throw std::invalid_argument("RT-Core batching requires --sparsity 1");
#if defined(GROTOR_CUGR2_ROUTING) || defined(GROTOR_GGR_ROUTING)
        if (stop_after_lshape)
            throw std::invalid_argument("--stop-after lshape requires the GRotor routing pipeline");
#endif
#ifndef GROTOR_ENABLE_RTCORE
        if (rtcore_batching)
            throw std::invalid_argument("Rebuild with -DGROTOR_ENABLE_RTCORE=ON to use RT-Core batching");
#endif
        program_start = std::chrono::high_resolution_clock::now();
        auto gpu_ready = std::async(std::launch::async, [] {
            grotor::cuda_check(cudaFree(nullptr), "initialize CUDA");
            cudaFuncAttributes attributes{};
            grotor::cuda_check(cudaFuncGetAttributes(&attributes, graph::update_wcost_cuda_ispd24),
                               "load CUDA module");
        });
        db::read(cap.data(), net.data());
        // The LUT loader uses relative filenames. Restore the caller's
        // directory immediately, so routing and output paths remain independent.
        const auto caller_directory = std::filesystem::current_path();
        std::filesystem::current_path(lut_dir);
        flute::readLUT();
        std::filesystem::current_path(caller_directory);
        gpu_ready.get();

        out_file = fopen(output.c_str(), "w");
        if (!out_file) throw std::runtime_error("Cannot open output: " + output);
        // RouteWriter supplies a large buffer; avoid a second stdio buffer.
        if (setvbuf(out_file, nullptr, _IONBF, 0))
            throw std::runtime_error("Cannot configure routing output");
#ifdef GROTOR_FULL_COST_REFERENCE
        printf("GROTOR_CONFIG batching=%s cost=full prefix=full memory=explicit\n", rtcore_batching ? "rtcore" : "cpu");
#elif defined(GROTOR_DENSE_PRESUM)
        printf("GROTOR_CONFIG batching=%s cost=incremental prefix=dense bit=off memory=explicit\n", rtcore_batching ? "rtcore" : "cpu");
#else
        printf("GROTOR_CONFIG batching=%s cost=incremental prefix=bit memory=explicit\n", rtcore_batching ? "rtcore" : "cpu");
#endif
#ifdef GROTOR_MULTI_LEGACY_LSHAPE
        printf("DAG_SCHEDULING lshape=warp flt=block\n");
#elif defined(GROTOR_SERIAL_DAG)
        printf("DAG_SCHEDULING lshape=thread flt=global_depth\n");
#else
        printf("DAG_SCHEDULING lshape=warp flt=block\n");
#endif
#ifdef GROTOR_REPORT_ABLATION_CONFIG
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
#error "Routing ablation timing must not enable prefix access counters"
#endif
        printf("ABLATION_CONFIG variant=%s counters=off batching=%s\n", GROTOR_ABLATION_VARIANT, rtcore_batching ? "rtcore" : "cpu");
#endif
        printf("CONGESTION_REDUCTION=per-cell\n");
        {
            cudb::RouterState state;
            state.batch_sparsity_override = sparsity;
            state.profile_stages = profile_stages;
            state.stop_after_lshape = stop_after_lshape;
            state.rtcore_batching = rtcore_batching;
            state.validate_rtcore_batches = validate_rtcore;
#if defined(GROTOR_GGR_ROUTING)
            grotor::route_ggr(state);
#elif defined(GROTOR_CUGR2_ROUTING)
            grotor::route_cugr2(state);
#else
            grotor::route(state);
#endif
        } // All host/device workspaces are released before exit.
        const int close_status = fclose(out_file);
        out_file = nullptr;
        if (close_status) throw std::runtime_error("Failed to close routing output");
        printf("GROTOR_COMPLETE wall_seconds=%.6f\n", elapsed_time());
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        if (out_file) fclose(out_file);
        fprintf(stderr, "GRotor error: %s\n", error.what());
#ifdef GROTOR_ENABLE_NVSHMEM
        int initialized = 0, finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Finalized(&finalized);
        if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
        return EXIT_FAILURE;
    }
}

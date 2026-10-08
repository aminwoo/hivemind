#include "interface/uci.h"
#include "environment/constants.h"
#include "common/globals.h"
#include "nn/engine.h"
#include "nn/onnx_utils.h"
#include "tools/benchmark.h"
#include "tools/selfplay.h"
#include "tools/nnue_data.h"
#include "tools/search_diag.h"
#include "nnue/network.h"
#include "search/alphabeta.h"
#include "environment/planes.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "tools/tournament.h"
#include "search/search_params.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/misc.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/types.h"
#include <iostream>
#include "nn/backend_compat.h"
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace std; 

namespace {

/**
 * @brief Reads the optional numeric argument of `bench` / `perft`.
 *
 * Both subcommands may also be given the --model flag the other subcommands
 * accept, so scan for the count instead of assuming argv[2] is a number -
 * stoi() on a flag terminates the process.
 */
int optional_count_argument(int argc, char** argv, int defaultValue) {
    for (int index = 2; index < argc; ++index) {
        const string argument = argv[index];
        if (!argument.empty()
            && argument.find_first_not_of("0123456789") == string::npos) {
            return stoi(argument);
        }
    }
    return defaultValue;
}

/**
 * @brief Reads the --batch-size flag of a subcommand.
 * @return The requested size, the compiled default when absent, or -1 on error.
 */
int batch_size_argument(int argc, char** argv) {
    for (int index = 2; index + 1 < argc; ++index) {
        if (string(argv[index]) != "--batch-size") {
            continue;
        }
        try {
            const int requested = stoi(argv[index + 1]);
            if (requested < 1 || requested > 1024) {
                cerr << "--batch-size must be between 1 and 1024" << endl;
                return -1;
            }
            return requested;
        } catch (const exception&) {
            cerr << "--batch-size expects a number" << endl;
            return -1;
        }
    }
    return SearchParams::BATCH_SIZE;
}

/** Returns the --model / --network path of a subcommand, or "" when absent. */
string model_path_argument(int argc, char** argv) {
    for (int index = 2; index + 1 < argc; ++index) {
        const string argument = argv[index];
        if (argument == "--model" || argument == "--network") {
            return argv[index + 1];
        }
    }
    return {};
}

bool parse_bool_argument(const string& value) {
    if (value == "true" || value == "1" || value == "on") return true;
    if (value == "false" || value == "0" || value == "off") return false;
    throw invalid_argument("Expected true/false, got: " + value);
}

}  // namespace

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --log <level>      Set log level: none, info, debug (default: none)" << endl;
    cout << "  --model <onnx>     Load this model in UCI mode (or --network, default: scans ./models)" << endl;
    cout << "  --nnue <file>      Search with alpha-beta on this NNUE (no ONNX model or GPU needed)" << endl;
    cout << "  --batch-size <n>   Inference batch size in UCI mode (default: "
         << SearchParams::BATCH_SIZE << "; also settable with the BatchSize UCI option)" << endl;
    cout << "  bench [iters]      Run inference benchmark (accepts --model, --batch-size)" << endl;
    cout << "  perft [depth]      Run move generation benchmark" << endl;
    cout << "  selfplay [options] Generate HVM5 training chunks and bughouse PGN" << endl;
    cout << "    --model <onnx> --games <n> --nodes <n> --output <dir> --seed <n>" << endl;
    cout << "    --max-macro-plies <n> --raw-policy-mean-macro-plies <x>" << endl;
    cout << "    --raw-policy-max-macro-plies <n> --raw-policy-high-temp-probability <x>" << endl;
    cout << "    --mcts-temperature <x> --mcts-temperature-decay <x>" << endl;
    cout << "    --node-random-factor <x>" << endl;
    cout << "    --fairy-stockfish-mate-nodes <n> (0 disables; default "
         << SearchParams::MATE_PROBE_ROOT_NODE_BUDGET << ")" << endl;
    cout << "    --chunk-samples <n> --dirichlet-alpha <x> --dirichlet-epsilon <x>" << endl;
    cout << "    --distill-output <dir> (also write HDST chunks of the search targets)" << endl;
    cout << "    --distill-chunk-positions <n> --parallel-games <n> --root-scans <bool>" << endl;
    cout << "    --training-chunks <bool> (write HVM5 chunks; default true)" << endl;
    cout << "  searchdiag         Root candidates of searches at several node budgets, per FEN" << endl;
    cout << "    --model <onnx> --fens <file> --output <jsonl> --budgets 50,200,800 --positions <n> --every <n>" << endl;
    cout << "  gennnue [options]  Generate NNUE distillation data from teacher-policy games" << endl;
    cout << "    --model <onnx> --positions <n> --threads <n> --batch-size <n> --output <dir>" << endl;
    cout << "    --seed <n> --max-macro-plies <n> --chunk-positions <n> --random-move-prob <x> --fens <bool>" << endl;
    cout << "    --format nnue|distill (distill: packed planes + value/WDL/moves-left + legal-move policies)" << endl;
    cout << "  tournament [options] Run a paired model-vs-model tournament" << endl;
    cout << "    --contender <onnx> --baseline <onnx> --games <even-n>" << endl;
    cout << "    --nodes <n> or --movetime <ms>" << endl;
    cout << "    --contender-batch-size <n> --baseline-batch-size <n>" << endl;
    cout << "    --output <dir> --seed <n> --max-macro-plies <n>" << endl;
    cout << "    --dirichlet-alpha <x> --dirichlet-epsilon <x>" << endl;
    cout << "    --contender-pw-coefficient <x> --baseline-pw-coefficient <x>" << endl;
    cout << "    --contender-pw-exponent <x> --baseline-pw-exponent <x>" << endl;
    cout << "    --contender-pw-mass <m0> --baseline-pw-mass <m0> (prior-mass widening; 0 = off)" << endl;
    cout << "    --{contender,baseline}-pw-mass-exponent <x> --{contender,baseline}-pw-mass-cap <x>" << endl;
    cout << "    --{contender,baseline}-root-pw-mass <m0> (-1 inherits internal target)" << endl;
    cout << "    --{contender,baseline}-pw-mass-normalize <bool> --{contender,baseline}-cpuct-init <x>" << endl;
    cout << "    --contender-threads <n> --baseline-threads <n> --positions <tsv>" << endl;
    cout << "    --contender-{mcgs,transpositions,root-mate-search,wdl-eval} <bool>" << endl;
    cout << "    --baseline-{mcgs,transpositions,root-mate-search,wdl-eval} <bool>" << endl;
    cout << "    --{contender,baseline}-{root-pw-coefficient,wdl-weight,moves-left-discount,q-value-weight,q-veto-delta} <x>" << endl;
    cout << "    --sprt-elo0 <x> --sprt-elo1 <x> [--sprt-alpha <x> --sprt-beta <x>]" << endl;
    cout << "    --contender-nnue <file> (alpha-beta contender instead of --contender)" << endl;
    cout << "    --contender-ab-movetime <ms> --contender-ab-depth <n> --contender-hash <mb>" << endl;
    cout << "    --baseline-nnue <file> --baseline-ab-movetime <ms> --baseline-ab-depth <n>" << endl;
    cout << "    --{contender,baseline}-ab-{check-extension,qsearch-checks} <bool>" << endl;
    cout << "    --{contender,baseline}-ab-set name=value[,...] (lmp_base, lmp_scale, pv_lmp_scale," << endl;
    cout << "      root_lmp_scale, lmr_divisor, qsearch_plies, rfp_margin, futility_margin)" << endl;
}

int main(int argc, char* argv[]) {
    // Model discovery must be relative to the executable for release bundles;
    // chess GUIs commonly launch engines with an unrelated working directory.
    Stockfish::CommandLine::init(argc, argv);

    // Parse --log argument first (can appear anywhere)
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--help") == 0) || (strcmp(argv[i], "-h") == 0)) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            g_logLevel = parseLogLevel(argv[i + 1]);
            // Remove these args from consideration
            for (int j = i; j + 2 < argc; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            i--;  // Recheck this position
        }
    }

    // How many Engine instances to build: one per CUDA device, or a single
    // host engine on the portable backend. Handle --help first so packaged
    // GPU builds can be smoke-tested on CI hosts without an NVIDIA device.
    int deviceCount = 1;
#if defined(HIVEMIND_BACKEND_TENSORRT)
    deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(error_id) << std::endl;
        return EXIT_FAILURE;
    }
#endif

    init_fairy_stockfish();

    init_policy_index();

    // Check for benchmark flag
    if (argc > 1 && string(argv[1]) == "bench") {
        cout << "Running inference benchmark..." << endl;
        const int benchBatchSize = batch_size_argument(argc, argv);
        if (benchBatchSize <= 0) {
            return EXIT_FAILURE;
        }
        Engine engine(0, benchBatchSize);

        const std::string onnxFile = resolveModelPath(
            model_path_argument(argc, argv));
        if (onnxFile.empty()) {
            cerr << "No ONNX model found in ./models or ./engine/models" << endl;
            return EXIT_FAILURE;
        }
        const std::string engineFile = getEnginePath(onnxFile, "fp16", benchBatchSize, 0, "v3");
        
        if (!engine.loadNetwork(onnxFile, engineFile)) {
            cerr << "Failed to load engine" << endl;
            return EXIT_FAILURE;
        }
        
        int iterations = optional_count_argument(argc, argv, 1000);
        benchmark_inference(engine, iterations);
        return EXIT_SUCCESS;
    }

    if (argc > 1 && string(argv[1]) == "searchdiag") {
        SearchDiagConfig config;
        string modelPath;
        try {
            for (int i = 2; i + 1 < argc; i += 2) {
                const string option = argv[i];
                const string value = argv[i + 1];
                if (option == "--model" || option == "--network") modelPath = value;
                else if (option == "--fens") config.fenFile = value;
                else if (option == "--output") config.outputFile = value;
                else if (option == "--positions") config.positions = stoull(value);
                else if (option == "--every") config.every = stoull(value);
                else if (option == "--budgets") {
                    config.budgets.clear();
                    stringstream budgets(value);
                    for (string budget; getline(budgets, budget, ',');) config.budgets.push_back(stoull(budget));
                }
                else throw invalid_argument("Unknown searchdiag option: " + option);
            }
            if (config.fenFile.empty() || config.outputFile.empty() || config.budgets.empty()) {
                throw invalid_argument("--fens, --output and --budgets are required");
            }
        } catch (const exception& error) {
            cerr << "Invalid searchdiag arguments: " << error.what() << endl;
            return EXIT_FAILURE;
        }
        const string onnxFile = resolveModelPath(modelPath);
        Engine engine(0, SearchParams::BATCH_SIZE);
        if (onnxFile.empty()
            || !engine.loadNetwork(onnxFile, getEnginePath(onnxFile, "fp16", SearchParams::BATCH_SIZE, 0, "v3"))) {
            cerr << "Failed to load a network" << endl;
            return EXIT_FAILURE;
        }
        try {
            return run_search_diag(engine, config);
        } catch (const exception& error) {
            cerr << "searchdiag failed: " << error.what() << endl;
            return EXIT_FAILURE;
        }
    }

    // Check for perft benchmark flag
    if (argc > 1 && string(argv[1]) == "perft") {
        int depth = optional_count_argument(argc, argv, 5);
        benchmark_movegen(depth);
        return EXIT_SUCCESS;
    }

    if (argc > 1 && string(argv[1]) == "selfplay") {
        SelfPlayConfig config;
        filesystem::path modelPath;
        try {
            for (int i = 2; i < argc; ++i) {
                const string option = argv[i];
                if (i + 1 >= argc) {
                    throw invalid_argument("Missing value for " + option);
                }
                const string value = argv[++i];
                if (option == "--games") config.games = stoull(value);
                else if (option == "--nodes") config.nodes = stoull(value);
                else if (option == "--model" || option == "--network") modelPath = value;
                else if (option == "--output") config.outputDirectory = value;
                else if (option == "--seed") config.seed = stoull(value);
                else if (option == "--max-macro-plies") config.maxMacroPlies = stoull(value);
                else if (option == "--raw-policy-mean-macro-plies") config.rawPolicyMeanMacroPlies = stod(value);
                else if (option == "--raw-policy-max-macro-plies") config.rawPolicyMaxMacroPlies = stoull(value);
                else if (option == "--raw-policy-high-temp-probability") config.rawPolicyHighTemperatureProbability = stod(value);
                else if (option == "--mcts-temperature") config.mctsTemperature = stod(value);
                else if (option == "--mcts-temperature-decay") config.mctsTemperatureDecay = stod(value);
                else if (option == "--mcts-temperature-plies") config.mctsTemperaturePlies = stoull(value);
                else if (option == "--resign-threshold") config.resignThreshold = stof(value);
                else if (option == "--resign-consecutive-plies") config.resignConsecutivePlies = stoull(value);
                else if (option == "--resign-disable-fraction") config.resignDisableFraction = stod(value);
                else if (option == "--q-value-ratio") config.qValueRatio = stod(value);
                else if (option == "--node-random-factor") config.nodeRandomFactor = stod(value);
                else if (option == "--fairy-stockfish-mate-nodes") config.fairyStockfishMateNodes = stoull(value);
                else if (option == "--chunk-samples") config.chunkSamples = stoull(value);
                else if (option == "--distill-output") config.distillOutputDirectory = value;
                else if (option == "--root-scans") config.rootScans = parse_bool_argument(value);
                else if (option == "--parallel-games") config.parallelGames = stoull(value);
                else if (option == "--training-chunks") config.writeTrainingChunks = parse_bool_argument(value);
                else if (option == "--distill-chunk-positions") config.distillChunkPositions = stoull(value);
                else if (option == "--dirichlet-alpha") config.dirichletAlpha = stof(value);
                else if (option == "--dirichlet-epsilon") config.dirichletEpsilon = stof(value);
                else if (option == "--batch-size") config.batchSize = stoi(value);
                else throw invalid_argument("Unknown selfplay option: " + option);
            }
        } catch (const exception& error) {
            cerr << "Invalid selfplay arguments: " << error.what() << endl;
            return EXIT_FAILURE;
        }

        const string onnxFile = resolveModelPath(modelPath.string());
        if (onnxFile.empty()) {
            cerr << "No ONNX model found; pass --model <onnx>" << endl;
            return EXIT_FAILURE;
        }
        vector<unique_ptr<Engine>> ownedEngines;
        vector<Engine*> engines;
        if (config.batchSize < 1 || config.batchSize > 1024) {
            cerr << "Self-play batch size must be between 1 and 1024" << endl;
            return EXIT_FAILURE;
        }
        // With parallel games, every game gets its own engine (and so its own
        // execution contexts) on each device.
        const size_t enginesPerDevice = std::max<size_t>(1, config.parallelGames);
        for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
            for (size_t copy = 0; copy < enginesPerDevice; ++copy) {
                auto engine = make_unique<Engine>(deviceId, config.batchSize);
                const string engineFile = getEnginePath(
                    onnxFile, "fp16", config.batchSize, deviceId, "v3");
                if (!engine->loadNetwork(onnxFile, engineFile)) {
                    cerr << "Failed to load engine on device " << deviceId << endl;
                    return EXIT_FAILURE;
                }
                engines.push_back(engine.get());
                ownedEngines.push_back(std::move(engine));
            }
        }
        if (engines.empty()) {
            cerr << "Failed to load an engine on any CUDA device" << endl;
            return EXIT_FAILURE;
        }
        cout << "Self-play using " << engines.size() << " inference engine(s)" << endl;
        cout << "Fairy-Stockfish mate search "
             << (config.fairyStockfishMateNodes > 0
                 ? "enabled (" + to_string(config.fairyStockfishMateNodes)
                   + " nodes per searched position)"
                 : "disabled")
             << endl;
        try {
            return run_selfplay(engines, config);
        } catch (const exception& error) {
            cerr << "Self-play failed: " << error.what() << endl;
            return EXIT_FAILURE;
        }
    }

    if (argc > 1 && string(argv[1]) == "gennnue") {
        NnueDataConfig config;
        filesystem::path modelPath;
        int batchSize = 256;
        try {
            for (int i = 2; i < argc; ++i) {
                const string option = argv[i];
                if (i + 1 >= argc) {
                    throw invalid_argument("Missing value for " + option);
                }
                const string value = argv[++i];
                if (option == "--positions") config.positions = stoull(value);
                else if (option == "--threads") config.threads = stoull(value);
                else if (option == "--model" || option == "--network") modelPath = value;
                else if (option == "--output") config.outputDirectory = value;
                else if (option == "--seed") config.seed = stoull(value);
                else if (option == "--max-macro-plies") config.maxMacroPlies = stoull(value);
                else if (option == "--chunk-positions") config.chunkPositions = stoull(value);
                else if (option == "--random-move-prob") config.randomMoveProbability = stod(value);
                else if (option == "--fens") config.writeFens = parse_bool_argument(value);
                else if (option == "--format") {
                    if (value != "nnue" && value != "distill") {
                        throw invalid_argument("--format must be nnue or distill");
                    }
                    config.distill = value == "distill";
                }
                else if (option == "--batch-size") batchSize = stoi(value);
                else throw invalid_argument("Unknown gennnue option: " + option);
            }
            if (batchSize < 1 || batchSize > 1024) {
                throw invalid_argument("--batch-size must be between 1 and 1024");
            }
        } catch (const exception& error) {
            cerr << "Invalid gennnue arguments: " << error.what() << endl;
            return EXIT_FAILURE;
        }
        const string onnxFile = resolveModelPath(modelPath.string());
        if (onnxFile.empty()) {
            cerr << "No ONNX model found; pass --model <onnx>" << endl;
            return EXIT_FAILURE;
        }
        Engine engine(0, batchSize);
        if (!engine.loadNetwork(onnxFile, getEnginePath(onnxFile, "fp16", batchSize, 0, "v3"))) {
            cerr << "Failed to load engine" << endl;
            return EXIT_FAILURE;
        }
        try {
            return run_nnue_data(engine, config);
        } catch (const exception& error) {
            cerr << "gennnue failed: " << error.what() << endl;
            return EXIT_FAILURE;
        }
    }

    // dumpplanes writes the network input planes of FEN lines (float32,
    // N x 74 x 8 x 8) with the engine's own encoder: calibration data for
    // scripts/quantize_int8.py.
    if (argc > 1 && string(argv[1]) == "dumpplanes") {
        string fenFile, outputFile;
        size_t every = 1;
        size_t limit = 0;
        for (int i = 2; i + 1 < argc; i += 2) {
            const string option = argv[i];
            if (option == "--fens") fenFile = argv[i + 1];
            else if (option == "--output") outputFile = argv[i + 1];
            else if (option == "--every") every = max<size_t>(1, stoull(argv[i + 1]));
            else if (option == "--limit") limit = stoull(argv[i + 1]);
            else {
                cerr << "Unknown option " << option << endl;
                return EXIT_FAILURE;
            }
        }
        ifstream fens(fenFile);
        ofstream out(outputFile, ios::binary);
        if (!fens || !out) {
            cerr << "Usage: dumpplanes --fens <file> --output <file> [--every n] [--limit n]" << endl;
            return EXIT_FAILURE;
        }
        vector<float> planes(NB_INPUT_VALUES());
        size_t index = 0;
        size_t written = 0;
        for (string line; getline(fens, line) && (limit == 0 || written < limit); ++index) {
            if (index % every != 0) {
                continue;
            }
            vector<string> fields;
            stringstream stream(line);
            for (string field; getline(stream, field, ';');) {
                fields.push_back(field);
            }
            if (fields.size() != 4) {
                continue;
            }
            Board board;
            board.set(fields[0] + "|" + fields[1]);
            board_to_planes(board, planes.data(),
                            fields[2] == "w" ? Stockfish::WHITE : Stockfish::BLACK, fields[3] == "1");
            out.write(reinterpret_cast<const char*>(planes.data()),
                      static_cast<streamsize>(planes.size() * sizeof(float)));
            ++written;
        }
        cout << "wrote " << written << " positions to " << outputFile << endl;
        return EXIT_SUCCESS;
    }

    // nnueeval / absearchbench read "fenA;fenB;team(w|b);advantage(0|1)" lines,
    // the format gennnue writes with --fens true.
    if (argc > 1 && (string(argv[1]) == "nnueeval" || string(argv[1]) == "absearchbench")) {
        const bool bench = string(argv[1]) == "absearchbench";
        string nnueFile, fenFile;
        size_t limit = bench ? 50 : 1000;
        ab::Options benchOptions;
        int depth = 6;
        int moveTime = 0;
        for (int i = 2; i + 1 < argc; i += 2) {
            const string option = argv[i];
            if (option == "--nnue") nnueFile = argv[i + 1];
            else if (option == "--fens") fenFile = argv[i + 1];
            else if (option == "--limit") limit = stoull(argv[i + 1]);
            else if (option == "--depth") depth = stoi(argv[i + 1]);
            else if (option == "--movetime") moveTime = stoi(argv[i + 1]);
            else if (option == "--check-extension") benchOptions.checkExtension = parse_bool_argument(argv[i + 1]);
            else if (option == "--qsearch-checks") benchOptions.quietChecksInQsearch = parse_bool_argument(argv[i + 1]);
            else if (option == "--set") {
                stringstream assignments(argv[i + 1]);
                for (string assignment; getline(assignments, assignment, ',');) {
                    if (!benchOptions.set(assignment)) {
                        cerr << "Unknown alpha-beta option: " << assignment << endl;
                        return EXIT_FAILURE;
                    }
                }
            }
            else {
                cerr << "Unknown option " << option << endl;
                return EXIT_FAILURE;
            }
        }
        nnue::Network network;
        string error;
        if (!network.load(nnueFile, &error)) {
            cerr << error << endl;
            return EXIT_FAILURE;
        }
        ifstream fens(fenFile);
        if (!fens) {
            cerr << "Cannot open " << fenFile << endl;
            return EXIT_FAILURE;
        }
        ab::Searcher searcher(network, 64);
        searcher.options = benchOptions;
        uint64_t totalNodes = 0;
        double totalSeconds = 0.0;
        int depthSum = 0;
        string line;
        size_t count = 0;
        while (count < limit && getline(fens, line)) {
            vector<string> fields;
            stringstream stream(line);
            for (string field; getline(stream, field, ';');) {
                fields.push_back(field);
            }
            if (fields.size() != 4) {
                continue;
            }
            Board board;
            board.set(fields[0] + "|" + fields[1]);
            const Stockfish::Color team = fields[2] == "w" ? Stockfish::WHITE : Stockfish::BLACK;
            const bool advantage = fields[3] == "1";
            ++count;
            if (!bench) {
                cout << network.evaluate(board, team, advantage) << '\n';
                continue;
            }
            searcher.clear();
            ab::Limits limits;
            limits.depth = depth;
            limits.moveTimeMs = moveTime;
            const auto start = chrono::steady_clock::now();
            const ab::Result result = searcher.search(board, team, advantage, limits);
            totalSeconds += chrono::duration<double>(chrono::steady_clock::now() - start).count();
            totalNodes += result.nodes;
            depthSum += result.depth;
        }
        if (bench) {
            cout << "positions " << count << " nodes " << totalNodes
                 << " seconds " << totalSeconds
                 << " nps " << static_cast<uint64_t>(totalNodes / max(1e-9, totalSeconds))
                 << " mean depth " << static_cast<double>(depthSum) / max<size_t>(1, count) << endl;
            const auto& st = searcher.stats;
            cout << "main " << st.mainNodes << " ttcuts " << st.ttCuts << " rfp " << st.rfpCuts
                 << " legalPairs " << st.legalPairs << " mainPairs " << st.mainPairs
                 << " q " << st.qNodes << " qCaptures " << st.qCaptures
                 << " evasionNodes " << st.evasionNodes << " evasionPairs " << st.evasionPairs << endl;
        }
        return EXIT_SUCCESS;
    }

    if (argc > 1 && string(argv[1]) == "tournament") {
        TournamentConfig config;
        filesystem::path contenderPath;
        string contenderNnuePath;
        string baselineNnuePath;
        size_t contenderHashMb = 64;
        ab::Options contenderOptions;
        ab::Options baselineOptions;
        filesystem::path baselinePath;
        try {
            for (int i = 2; i < argc; ++i) {
                const string option = argv[i];
                if (i + 1 >= argc) {
                    throw invalid_argument("Missing value for " + option);
                }
                const string value = argv[++i];
                if (option == "--games") config.games = stoull(value);
                else if (option == "--nodes") {
                    config.nodes = stoull(value);
                    config.moveTimeMs = 0;
                }
                else if (option == "--movetime") {
                    config.moveTimeMs = stoi(value);
                    config.nodes = 0;
                }
                else if (option == "--contender-batch-size") config.contenderBatchSize = stoi(value);
                else if (option == "--baseline-batch-size") config.baselineBatchSize = stoi(value);
                else if (option == "--contender-threads") config.contenderThreads = stoi(value);
                else if (option == "--baseline-threads") config.baselineThreads = stoi(value);
                else if (option == "--contender") contenderPath = value;
                else if (option == "--baseline") baselinePath = value;
                else if (option == "--output") config.outputDirectory = value;
                else if (option == "--positions") config.positionsFile = value;
                else if (option == "--seed") config.seed = stoull(value);
                else if (option == "--max-macro-plies") config.maxMacroPlies = stoull(value);
                else if (option == "--dirichlet-alpha") config.dirichletAlpha = stof(value);
                else if (option == "--dirichlet-epsilon") config.dirichletEpsilon = stof(value);
                else if (option == "--contender-pw-coefficient") {
                    config.contenderPwCoefficient = stof(value);
                    config.contenderRootPwCoefficient = config.contenderPwCoefficient;
                }
                else if (option == "--baseline-pw-coefficient") {
                    config.baselinePwCoefficient = stof(value);
                    config.baselineRootPwCoefficient = config.baselinePwCoefficient;
                }
                else if (option == "--contender-root-pw-coefficient") config.contenderRootPwCoefficient = stof(value);
                else if (option == "--baseline-root-pw-coefficient") config.baselineRootPwCoefficient = stof(value);
                else if (option == "--contender-pw-exponent") config.contenderPwExponent = stof(value);
                else if (option == "--baseline-pw-exponent") config.baselinePwExponent = stof(value);
                else if (option == "--contender-pw-mass") config.contenderPwMassStart = stof(value);
                else if (option == "--baseline-pw-mass") config.baselinePwMassStart = stof(value);
                else if (option == "--contender-pw-mass-exponent") config.contenderPwMassExponent = stof(value);
                else if (option == "--baseline-pw-mass-exponent") config.baselinePwMassExponent = stof(value);
                else if (option == "--contender-pw-mass-cap") config.contenderPwMassCap = stof(value);
                else if (option == "--baseline-pw-mass-cap") config.baselinePwMassCap = stof(value);
                else if (option == "--contender-root-pw-mass") config.contenderRootPwMassStart = stof(value);
                else if (option == "--baseline-root-pw-mass") config.baselineRootPwMassStart = stof(value);
                else if (option == "--contender-pw-mass-normalize") config.contenderPwMassNormalize = parse_bool_argument(value);
                else if (option == "--baseline-pw-mass-normalize") config.baselinePwMassNormalize = parse_bool_argument(value);
                else if (option == "--contender-cpuct-init") config.contenderCpuctInit = stof(value);
                else if (option == "--baseline-cpuct-init") config.baselineCpuctInit = stof(value);
                else if (option == "--contender-mcgs") config.contenderMcgs = parse_bool_argument(value);
                else if (option == "--baseline-mcgs") config.baselineMcgs = parse_bool_argument(value);
                else if (option == "--contender-transpositions") config.contenderTranspositions = parse_bool_argument(value);
                else if (option == "--baseline-transpositions") config.baselineTranspositions = parse_bool_argument(value);
                else if (option == "--contender-root-mate-search") config.contenderRootMateSearch = parse_bool_argument(value);
                else if (option == "--baseline-root-mate-search") config.baselineRootMateSearch = parse_bool_argument(value);
                else if (option == "--contender-wdl-eval") config.contenderWdlEval = parse_bool_argument(value);
                else if (option == "--baseline-wdl-eval") config.baselineWdlEval = parse_bool_argument(value);
                else if (option == "--contender-wdl-weight") config.contenderWdlWeight = stof(value);
                else if (option == "--baseline-wdl-weight") config.baselineWdlWeight = stof(value);
                else if (option == "--contender-moves-left-discount") config.contenderMovesLeftDiscount = stof(value);
                else if (option == "--baseline-moves-left-discount") config.baselineMovesLeftDiscount = stof(value);
                else if (option == "--contender-q-value-weight") config.contenderQValueWeight = stof(value);
                else if (option == "--baseline-q-value-weight") config.baselineQValueWeight = stof(value);
                else if (option == "--contender-q-veto-delta") config.contenderQVetoDelta = stof(value);
                else if (option == "--baseline-q-veto-delta") config.baselineQVetoDelta = stof(value);
                else if (option == "--sprt-elo0") config.sprtElo0 = stod(value);
                else if (option == "--sprt-elo1") config.sprtElo1 = stod(value);
                else if (option == "--sprt-alpha") config.sprtAlpha = stod(value);
                else if (option == "--sprt-beta") config.sprtBeta = stod(value);
                else if (option == "--contender-nnue") contenderNnuePath = value;
                else if (option == "--contender-ab-movetime") config.contenderAbMoveTimeMs = stoi(value);
                else if (option == "--contender-ab-depth") config.contenderAbDepth = stoi(value);
                else if (option == "--contender-hash") contenderHashMb = stoull(value);
                else if (option == "--baseline-nnue") baselineNnuePath = value;
                else if (option == "--baseline-ab-movetime") config.baselineAbMoveTimeMs = stoi(value);
                else if (option == "--baseline-ab-depth") config.baselineAbDepth = stoi(value);
                else if (option == "--contender-ab-check-extension") contenderOptions.checkExtension = parse_bool_argument(value);
                else if (option == "--baseline-ab-check-extension") baselineOptions.checkExtension = parse_bool_argument(value);
                else if (option == "--contender-ab-qsearch-checks") contenderOptions.quietChecksInQsearch = parse_bool_argument(value);
                else if (option == "--baseline-ab-qsearch-checks") baselineOptions.quietChecksInQsearch = parse_bool_argument(value);
                else if (option == "--contender-ab-set" || option == "--baseline-ab-set") {
                    ab::Options& target = option == "--contender-ab-set" ? contenderOptions : baselineOptions;
                    stringstream assignments(value);
                    for (string assignment; getline(assignments, assignment, ',');) {
                        if (!target.set(assignment)) {
                            throw invalid_argument("Unknown alpha-beta option: " + assignment);
                        }
                    }
                }
                else throw invalid_argument("Unknown tournament option: " + option);
            }
        } catch (const exception& error) {
            cerr << "Invalid tournament arguments: " << error.what() << endl;
            return EXIT_FAILURE;
        }
        const bool alphaBetaContender = !contenderNnuePath.empty();
        const bool alphaBetaBaseline = !baselineNnuePath.empty();
        if ((contenderPath.empty() && !alphaBetaContender)
            || (baselinePath.empty() && !alphaBetaBaseline)) {
            cerr << "Tournament requires --contender <onnx> (or --contender-nnue <file>)"
                    " and --baseline <onnx>" << endl;
            return EXIT_FAILURE;
        }

        unique_ptr<Engine> contender;
        nnue::Network contenderNetwork;
        unique_ptr<ab::Searcher> contenderSearcher;
        if (alphaBetaContender) {
            string error;
            if (!contenderNetwork.load(contenderNnuePath, &error)) {
                cerr << "Failed to load contender NNUE: " << error << endl;
                return EXIT_FAILURE;
            }
            contenderSearcher = make_unique<ab::Searcher>(contenderNetwork, contenderHashMb);
            contenderSearcher->options = contenderOptions;
            contenderPath = contenderNnuePath;
        } else {
            contender = make_unique<Engine>(0, config.contenderBatchSize);
            const string contenderEngine = getEnginePath(
                contenderPath.string(), "fp16", config.contenderBatchSize, 0, "v3");
            if (!contender->loadNetwork(contenderPath.string(), contenderEngine)) {
                cerr << "Failed to load contender model" << endl;
                return EXIT_FAILURE;
            }
        }
        unique_ptr<Engine> baseline;
        nnue::Network baselineNetwork;
        unique_ptr<ab::Searcher> baselineSearcher;
        if (alphaBetaBaseline) {
            string error;
            if (!baselineNetwork.load(baselineNnuePath, &error)) {
                cerr << "Failed to load baseline NNUE: " << error << endl;
                return EXIT_FAILURE;
            }
            baselineSearcher = make_unique<ab::Searcher>(baselineNetwork, contenderHashMb);
            baselineSearcher->options = baselineOptions;
            baselinePath = baselineNnuePath;
        } else {
            baseline = make_unique<Engine>(0, config.baselineBatchSize);
            const string baselineEngine = getEnginePath(
                baselinePath.string(), "fp16", config.baselineBatchSize, 0, "v3");
            if (!baseline->loadNetwork(baselinePath.string(), baselineEngine)) {
                cerr << "Failed to load baseline model" << endl;
                return EXIT_FAILURE;
            }
        }
        config.contenderModelSignature = computeFileSignature(
            contenderPath.string(), "hivemind-tournament-model");
        config.baselineModelSignature = computeFileSignature(
            baselinePath.string(), "hivemind-tournament-model");
        try {
            const string contenderName = alphaBetaContender
                ? contenderPath.stem().string() + "-alphabeta"
                : contenderPath.stem().string() + "-b" + std::to_string(config.contenderBatchSize);
            const string baselineName = alphaBetaBaseline
                ? baselinePath.stem().string() + "-alphabeta"
                : baselinePath.stem().string() + "-b" + std::to_string(config.baselineBatchSize);
            return run_tournament(
                contender.get(), baseline.get(), contenderName, baselineName,
                config, contenderSearcher.get(), baselineSearcher.get());
        } catch (const exception& error) {
            cerr << "Tournament failed: " << error.what() << endl;
            return EXIT_FAILURE;
        }
    }

    filesystem::path modelPath;
    string nnuePath;
    int uciBatchSize = SearchParams::BATCH_SIZE;
    try {
        for (int i = 1; i < argc; ++i) {
            const string option = argv[i];
            if (option != "--model" && option != "--network"
                && option != "--batch-size" && option != "--nnue") {
                throw invalid_argument("Unknown UCI option: " + option);
            }
            if (i + 1 >= argc) {
                throw invalid_argument("Missing value for " + option);
            }
            const string value = argv[++i];
            if (option == "--batch-size") {
                uciBatchSize = stoi(value);
                if (uciBatchSize < 1 || uciBatchSize > 1024) {
                    throw invalid_argument("--batch-size must be between 1 and 1024");
                }
            } else if (option == "--nnue") {
                nnuePath = value;
            } else {
                modelPath = value;
            }
        }
    } catch (const exception& error) {
        cerr << "Invalid UCI arguments: " << error.what() << endl;
        return EXIT_FAILURE;
    }

    UCI uci;
    std::vector<int> deviceIds(deviceCount);
    iota(deviceIds.begin(), deviceIds.end(), 0);

    std::cout << "Hivemind " << HIVEMIND_VERSION << std::endl;

    // With --nnue alone the engine runs alpha-beta on the CPU and never
    // touches an ONNX model or GPU.
    if ((nnuePath.empty() || !modelPath.empty())
        && !uci.initializeEngines(deviceIds, modelPath.string(), uciBatchSize)) {
        return EXIT_FAILURE;
    }
    if (!nnuePath.empty() && !uci.load_nnue(nnuePath)) {
        return EXIT_FAILURE;
    }
    uci.loop();
}

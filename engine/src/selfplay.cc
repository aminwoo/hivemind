#include "selfplay.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "agent.h"
#include "board.h"
#include "constants.h"
#include "engine.h"
#include "globals.h"
#include "planes.h"
#include "utils.h"

namespace {

constexpr std::array<char, 4> CHUNK_MAGIC = {'H', 'V', 'M', '3'};
constexpr uint32_t CHUNK_VERSION = 3;

struct SparsePolicyEntry {
    uint16_t index;
    float probability;
};

struct TrainingSample {
    uint64_t gameId = 0;
    uint32_t nodes = 0;
    uint16_t macroPly = 0;
    uint16_t movesLeft = 0;
    uint8_t team = 0;
    uint8_t hasTimeAdvantage = 0;
    int8_t outcome = 0;
    uint8_t wdl = 1;
    std::array<uint8_t, NB_INPUT_VALUES()> planes{};
    std::vector<SparsePolicyEntry> policyA;
    std::vector<SparsePolicyEntry> policyB;
};

struct PgnMove {
    std::string token;
    std::string san;
    float remainingSeconds = 0.0f;
};

template <typename T>
void write_scalar(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("Failed to write self-play chunk");
    }
}

class ChunkWriter {
public:
    ChunkWriter(std::filesystem::path directory, size_t samplesPerChunk, uint64_t runId)
        : directory_(std::move(directory)),
          samplesPerChunk_(std::max<size_t>(1, samplesPerChunk)),
          runId_(runId) {
        std::filesystem::create_directories(directory_);
    }

    void append(std::vector<TrainingSample> samples) {
        for (TrainingSample& sample : samples) {
            pending_.push_back(std::move(sample));
            if (pending_.size() >= samplesPerChunk_) {
                flush(samplesPerChunk_);
            }
        }
    }

    void finish() {
        if (!pending_.empty()) {
            flush(pending_.size());
        }
    }

private:
    void write_policy(std::ostream& stream, const std::vector<SparsePolicyEntry>& policy) {
        if (policy.size() > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("Sparse policy is too large for HVM3");
        }
        write_scalar(stream, static_cast<uint16_t>(policy.size()));
        for (const SparsePolicyEntry& entry : policy) {
            write_scalar(stream, entry.index);
            write_scalar(stream, entry.probability);
        }
    }

    void flush(size_t count) {
        std::ostringstream filename;
        filename << "chunk_" << runId_ << '_' << std::setw(6) << std::setfill('0')
                 << chunkIndex_++ << ".hvm";
        const std::filesystem::path finalPath = directory_ / filename.str();
        const std::filesystem::path temporaryPath = finalPath.string() + ".tmp";

        std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("Unable to create " + temporaryPath.string());
        }
        stream.write(CHUNK_MAGIC.data(), CHUNK_MAGIC.size());
        write_scalar(stream, CHUNK_VERSION);
        write_scalar(stream, static_cast<uint16_t>(NB_INPUT_CHANNELS));
        write_scalar(stream, static_cast<uint16_t>(NB_POLICY_VALUES()));
        write_scalar(stream, static_cast<uint64_t>(count));

        for (size_t index = 0; index < count; ++index) {
            const TrainingSample& sample = pending_[index];
            write_scalar(stream, sample.gameId);
            write_scalar(stream, sample.nodes);
            write_scalar(stream, sample.macroPly);
            write_scalar(stream, sample.movesLeft);
            write_scalar(stream, sample.team);
            write_scalar(stream, sample.hasTimeAdvantage);
            write_scalar(stream, sample.outcome);
            write_scalar(stream, sample.wdl);
            stream.write(
                reinterpret_cast<const char*>(sample.planes.data()),
                static_cast<std::streamsize>(sample.planes.size()));
            write_policy(stream, sample.policyA);
            write_policy(stream, sample.policyB);
        }
        stream.close();
        if (!stream) {
            throw std::runtime_error("Failed to finalize " + temporaryPath.string());
        }

        std::error_code error;
        std::filesystem::rename(temporaryPath, finalPath, error);
        if (error) {
            std::filesystem::remove(temporaryPath);
            throw std::runtime_error("Unable to publish " + finalPath.string() + ": " + error.message());
        }
        pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(count));
    }

    std::filesystem::path directory_;
    size_t samplesPerChunk_;
    uint64_t runId_;
    size_t chunkIndex_ = 0;
    std::vector<TrainingSample> pending_;
};

uint64_t mix_seed(uint64_t seed, uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return seed ^ (value ^ (value >> 31));
}

void apply_temperature(std::vector<float>& probabilities, double temperature) {
    if (probabilities.empty() || temperature <= 0.0) {
        throw std::invalid_argument("Policy temperature must be positive");
    }
    const double exponent = 1.0 / temperature;
    double total = 0.0;
    for (float& probability : probabilities) {
        probability = static_cast<float>(std::pow(std::max(0.0f, probability), exponent));
        total += probability;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        const float uniform = 1.0f / static_cast<float>(probabilities.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
        return;
    }
    for (float& probability : probabilities) {
        probability = static_cast<float>(probability / total);
    }
}

size_t sample_initialization_length(const SelfPlayConfig& config,
                                    std::mt19937_64& randomEngine) {
    if (config.rawPolicyMeanMacroPlies <= 0.0 || config.rawPolicyMaxMacroPlies == 0) {
        return 0;
    }
    std::exponential_distribution<double> distribution(
        1.0 / config.rawPolicyMeanMacroPlies);
    size_t length = static_cast<size_t>(std::llround(distribution(randomEngine)));
    if (length > config.rawPolicyMaxMacroPlies) {
        std::uniform_int_distribution<size_t> clipped(0, config.rawPolicyMaxMacroPlies);
        length = clipped(randomEngine);
    }
    return length;
}

double sample_raw_policy_temperature(const SelfPlayConfig& config,
                                     std::mt19937_64& randomEngine) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (unit(randomEngine) >= config.rawPolicyHighTemperatureProbability) {
        return 1.0;
    }
    const double choice = unit(randomEngine);
    if (choice < 0.75) {
        return 2.0;
    }
    if (choice < 0.95) {
        return 5.0;
    }
    return 10.0;
}

size_t randomized_node_budget(const SelfPlayConfig& config,
                              std::mt19937_64& randomEngine) {
    std::uniform_real_distribution<double> jitter(
        -config.nodeRandomFactor, config.nodeRandomFactor);
    return std::max<size_t>(1, static_cast<size_t>(std::llround(
        static_cast<double>(config.nodes) * (1.0 + jitter(randomEngine)))));
}

double mcts_temperature(const SelfPlayConfig& config, size_t macroPly) {
    return config.mctsTemperature
        * std::pow(config.mctsTemperatureDecay, static_cast<double>(macroPly / 2));
}

class RawPolicyEvaluator {
public:
    explicit RawPolicyEvaluator(size_t batchSize)
        : batchSize_(batchSize),
          observations_(batchSize * NB_INPUT_VALUES()),
          values_(batchSize),
          policyA_(batchSize * NB_POLICY_VALUES()),
          policyB_(batchSize * NB_POLICY_VALUES()),
          wdl_(batchSize * 3),
          movesLeft_(batchSize) {}

    void evaluate(Engine& engine,
                  Board& board,
                  Stockfish::Color team,
                  bool hasTimeAdvantage) {
        std::array<float, NB_INPUT_VALUES()> planes{};
        board_to_planes(board, planes.data(), team, hasTimeAdvantage);
        for (size_t batch = 0; batch < batchSize_; ++batch) {
            std::copy(planes.begin(), planes.end(),
                      observations_.begin() + static_cast<std::ptrdiff_t>(batch * planes.size()));
        }
        if (!engine.runInference(
                observations_.data(), values_.data(), policyA_.data(), policyB_.data(),
                wdl_.data(), movesLeft_.data())) {
            throw std::runtime_error("Raw-policy inference failed");
        }
    }

    float* policy(int boardNumber) {
        return boardNumber == BOARD_A ? policyA_.data() : policyB_.data();
    }

private:
    size_t batchSize_;
    std::vector<float> observations_;
    std::vector<float> values_;
    std::vector<float> policyA_;
    std::vector<float> policyB_;
    std::vector<float> wdl_;
    std::vector<float> movesLeft_;
};

void prepare_raw_policy(
    Board& board,
    int boardNumber,
    bool boardOnTurn,
    float* policyOutput,
    double temperature,
    std::vector<Stockfish::Move>& actions,
    std::vector<float>& probabilities) {
    if (boardOnTurn) {
        actions = board.legal_moves(boardNumber);
        std::erase_if(actions, [&board, boardNumber](Stockfish::Move move) {
            return !is_policy_move_representable(board, boardNumber, move);
        });
    }
    if (actions.empty()) {
        actions.push_back(Stockfish::MOVE_NONE);
        probabilities.push_back(1.0f);
        return;
    }
    actions.push_back(Stockfish::MOVE_NONE);
    probabilities = get_normalized_probability(
        policyOutput, actions, boardNumber, board);
    apply_temperature(probabilities, temperature);
}

JointActionCandidate sample_raw_policy_action(
    Engine& engine,
    RawPolicyEvaluator& evaluator,
    Board& board,
    Stockfish::Color team,
    bool hasTimeAdvantage,
    double temperature,
    std::mt19937_64& randomEngine) {
    evaluator.evaluate(engine, board, team, hasTimeAdvantage);
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == team;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~team;
    std::vector<Stockfish::Move> actionsA;
    std::vector<Stockfish::Move> actionsB;
    std::vector<float> probabilitiesA;
    std::vector<float> probabilitiesB;
    prepare_raw_policy(
        board, BOARD_A, boardAOnTurn, evaluator.policy(BOARD_A), temperature,
        actionsA, probabilitiesA);
    prepare_raw_policy(
        board, BOARD_B, boardBOnTurn, evaluator.policy(BOARD_B), temperature,
        actionsB, probabilitiesB);

    std::discrete_distribution<size_t> sampleA(probabilitiesA.begin(), probabilitiesA.end());
    std::discrete_distribution<size_t> sampleB(probabilitiesB.begin(), probabilitiesB.end());
    size_t indexA = sampleA(randomEngine);
    size_t indexB = sampleB(randomEngine);
    if (!is_double_sit_legal(hasTimeAdvantage, boardAOnTurn, boardBOnTurn)
        && actionsA[indexA] == Stockfish::MOVE_NONE
        && actionsB[indexB] == Stockfish::MOVE_NONE) {
        std::vector<float> validProbabilitiesA = probabilitiesA;
        validProbabilitiesA.back() *= 1.0f - probabilitiesB.back();
        if (std::accumulate(
                validProbabilitiesA.begin(), validProbabilitiesA.end(), 0.0) <= 0.0) {
            throw std::runtime_error("Raw policy produced no legal joint action");
        }
        std::discrete_distribution<size_t> validSampleA(
            validProbabilitiesA.begin(), validProbabilitiesA.end());
        indexA = validSampleA(randomEngine);
        if (actionsA[indexA] == Stockfish::MOVE_NONE) {
            std::vector<float> validProbabilitiesB = probabilitiesB;
            validProbabilitiesB.back() = 0.0f;
            std::discrete_distribution<size_t> validSampleB(
                validProbabilitiesB.begin(), validProbabilitiesB.end());
            indexB = validSampleB(randomEngine);
        } else {
            indexB = sampleB(randomEngine);
        }
    }
    return JointActionCandidate(
        actionsA[indexA], probabilitiesA[indexA], indexA,
        actionsB[indexB], probabilitiesB[indexB], indexB,
        boardAOnTurn, boardBOnTurn, hasTimeAdvantage);
}

bool action_leads_to_terminal(
    Board& board,
    const JointActionCandidate& action,
    Stockfish::Color team,
    bool hasTimeAdvantage) {
    Board future(board);
    future.make_moves(action.moveA, action.moveB);
    const Stockfish::Color nextTeam = ~team;
    const bool nextTeamHasTimeAdvantage = !hasTimeAdvantage;
    return future.is_checkmate(nextTeam, nextTeamHasTimeAdvantage)
        || future.is_checkmate(team, hasTimeAdvantage)
        || future.is_draw();
}

int policy_index(Board& board, int boardNumber, Stockfish::Move move) {
    std::string label = board.uci_move(boardNumber, move);
    if (label.size() == 5 && label.back() == 'q') {
        label.pop_back();
    }
    if (board.side_to_move(boardNumber) == Stockfish::BLACK && move != Stockfish::MOVE_NONE) {
        label = mirror_move(label);
    }
    const auto found = POLICY_INDEX.find(label);
    if (found == POLICY_INDEX.end()) {
        throw std::runtime_error("Move is absent from policy map: " + label);
    }
    return found->second;
}

std::vector<SparsePolicyEntry> marginal_policy(
    Board& board,
    int boardNumber,
    const std::vector<RootEdgeStats>& edges) {
    std::map<uint16_t, uint64_t> visitsByMove;
    uint64_t totalVisits = 0;
    for (const RootEdgeStats& edge : edges) {
        if (edge.visits <= 0) {
            continue;
        }
        const Stockfish::Move move = boardNumber == BOARD_A
            ? edge.action.moveA
            : edge.action.moveB;
        const int index = policy_index(board, boardNumber, move);
        visitsByMove[static_cast<uint16_t>(index)] += static_cast<uint64_t>(edge.visits);
        totalVisits += static_cast<uint64_t>(edge.visits);
    }
    if (totalVisits == 0) {
        throw std::runtime_error("Search returned no visited root edges");
    }

    std::vector<SparsePolicyEntry> policy;
    policy.reserve(visitsByMove.size());
    for (const auto& [index, visits] : visitsByMove) {
        policy.push_back({index, static_cast<float>(visits) / static_cast<float>(totalVisits)});
    }
    return policy;
}

JointActionCandidate select_action(
    const std::vector<RootEdgeStats>& edges,
    double temperature,
    std::mt19937_64& randomEngine) {
    if (edges.empty()) {
        throw std::runtime_error("Cannot select from an empty root");
    }
    if (temperature <= 1e-6) {
        return std::max_element(
            edges.begin(), edges.end(),
            [](const RootEdgeStats& left, const RootEdgeStats& right) {
                return left.visits < right.visits;
            })->action;
    }

    std::vector<double> weights;
    weights.reserve(edges.size());
    const int maxVisits = std::max_element(
        edges.begin(), edges.end(),
        [](const RootEdgeStats& left, const RootEdgeStats& right) {
            return left.visits < right.visits;
        })->visits;
    for (const RootEdgeStats& edge : edges) {
        weights.push_back(edge.visits > 0 && maxVisits > 0
            ? std::exp((std::log(static_cast<double>(edge.visits))
                        - std::log(static_cast<double>(maxVisits))) / temperature)
            : 0.0);
    }
    if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
        return edges.front().action;
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return edges[distribution(randomEngine)].action;
}

std::array<uint8_t, NB_INPUT_VALUES()> encode_planes(
    Board& board,
    Stockfish::Color team,
    bool hasTimeAdvantage) {
    std::array<float, NB_INPUT_VALUES()> raw{};
    std::array<uint8_t, NB_INPUT_VALUES()> encoded{};
    board_to_planes(board, raw.data(), team, hasTimeAdvantage);
    for (size_t index = 0; index < raw.size(); ++index) {
        encoded[index] = static_cast<uint8_t>(std::lround(
            std::clamp(raw[index], 0.0f, 1.0f) * 255.0f));
    }
    return encoded;
}

std::string current_date() {
    const std::time_t now = std::time(nullptr);
    std::tm localTime{};
    localtime_r(&now, &localTime);
    std::ostringstream date;
    date << std::put_time(&localTime, "%Y.%m.%d");
    return date.str();
}

void append_pgn(
    const std::filesystem::path& path,
    size_t round,
    const std::vector<PgnMove>& moves,
    int winner,
    int startingTeam,
    size_t rawPolicyMacroPlies,
    size_t rawPolicyEvents,
    float initialClockSeconds,
    const std::string& termination) {
    std::ofstream stream(path, std::ios::app);
    if (!stream) {
        throw std::runtime_error("Unable to append " + path.string());
    }
    const std::string result = winner == 0 ? "1-0" : winner == 1 ? "0-1" : "1/2-1/2";
    const std::string winnerName = winner == 0 ? "Hivemind-A" : winner == 1 ? "Hivemind-B" : "Draw";
    stream << "[Event \"Hivemind Self-Play\"]\n"
           << "[Site \"Hivemind Engine\"]\n"
           << "[Date \"" << current_date() << "\"]\n"
           << "[Round \"" << round << "\"]\n"
           << "[Variant \"bughouse\"]\n"
           << "[TimeControl \"" << static_cast<int>(initialClockSeconds) << "\"]\n"
           << "[WhiteTeam \"Hivemind-A\"]\n"
           << "[BlackTeam \"Hivemind-B\"]\n"
           << "[WhiteA \"Hivemind-A1\"]\n"
           << "[BlackA \"Hivemind-B2\"]\n"
           << "[WhiteB \"Hivemind-B1\"]\n"
           << "[BlackB \"Hivemind-A2\"]\n"
           << "[TimeAdvantage \"" << (startingTeam == 0 ? "Hivemind-B" : "Hivemind-A") << "\"]\n"
           << "[RawPolicyMacroPlies \"" << rawPolicyMacroPlies << "\"]\n"
           << "[RawPolicyEvents \"" << rawPolicyEvents << "\"]\n"
           << "[PlyCount \"" << moves.size() << "\"]\n"
           << "[Result \"" << result << "\"]\n"
           << "[Termination \"" << termination << "\"]\n\n";

    stream << std::fixed << std::setprecision(1);
    for (const PgnMove& move : moves) {
        stream << move.token << ". " << move.san << " {" << move.remainingSeconds << "} ";
    }
    stream << "{C:" << termination << ' ' << result << "}\n"
           << '{' << winnerName << (winner < 0 ? " game" : " won") << " by " << termination << "} *\n\n";
}

void append_pgn_move(
    Board& board,
    int boardNumber,
    Stockfish::Move move,
    std::array<int, 2>& moveNumbers,
    size_t eventIndex,
    float initialClockSeconds,
    std::vector<PgnMove>& pgnMoves) {
    if (move == Stockfish::MOVE_NONE) {
        return;
    }
    const bool whiteToMove = board.side_to_move(boardNumber) == Stockfish::WHITE;
    const char boardLetter = boardNumber == BOARD_A
        ? (whiteToMove ? 'A' : 'a')
        : (whiteToMove ? 'B' : 'b');
    PgnMove pgnMove;
    pgnMove.token = std::to_string(moveNumbers[boardNumber]) + boardLetter;
    pgnMove.san = board.san_move(boardNumber, move);
    pgnMove.remainingSeconds = std::max(
        0.0f, initialClockSeconds - 0.1f * static_cast<float>(eventIndex + 1));
    pgnMoves.push_back(std::move(pgnMove));
    if (!whiteToMove) {
        moveNumbers[boardNumber]++;
    }
}

} // namespace

int run_selfplay(Engine& engine, const SelfPlayConfig& config) {
    if (config.games == 0 || config.nodes == 0 || config.maxMacroPlies == 0) {
        throw std::invalid_argument("games, nodes, and max-macro-plies must be positive");
    }
    if (config.rawPolicyMeanMacroPlies < 0.0
        || config.rawPolicyHighTemperatureProbability < 0.0
        || config.rawPolicyHighTemperatureProbability > 1.0
        || config.mctsTemperature <= 0.0
        || config.mctsTemperatureDecay <= 0.0
        || config.mctsTemperatureDecay > 1.0
        || config.nodeRandomFactor < 0.0
        || config.nodeRandomFactor >= 1.0
        || config.waitPassPriorFloor < 0.0f
        || config.waitPassPriorFloor > 1.0f
        || config.coordinationPassPriorFloor < 0.0f
        || config.coordinationPassPriorFloor > 1.0f) {
        throw std::invalid_argument("Invalid self-play exploration configuration");
    }
    const uint64_t runId = config.seed != 0
        ? config.seed
        : static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937_64 randomEngine(runId);
    const std::filesystem::path trainingDirectory = config.outputDirectory / "training_data";
    std::filesystem::create_directories(config.outputDirectory);
    ChunkWriter chunkWriter(trainingDirectory, config.chunkSamples, runId);
    const std::filesystem::path pgnPath = config.outputDirectory / "games.pgn";
    std::vector<Engine*> engines = {&engine};
    RawPolicyEvaluator rawPolicyEvaluator(static_cast<size_t>(engine.getBatchSize()));

    for (size_t gameIndex = 0; gameIndex < config.games; ++gameIndex) {
        Board board;
        Agent agent;
        const int startingTeam = static_cast<int>(randomEngine() & 1ULL);
        Stockfish::Color team = startingTeam == 0 ? Stockfish::WHITE : Stockfish::BLACK;
        bool hasTimeAdvantage = false;
        int winner = -1;
        std::string termination = "macro-ply limit";
        std::vector<TrainingSample> samples;
        std::vector<PgnMove> pgnMoves;
        std::array<int, 2> moveNumbers = {1, 1};
        const size_t initializationLength = sample_initialization_length(config, randomEngine);
        bool rawInitializationActive = initializationLength > 0;
        size_t rawPolicyMacroPlies = 0;
        size_t rawPolicyEvents = 0;

        for (size_t macroPly = 0; macroPly < config.maxMacroPlies; ++macroPly) {
            if (board.is_checkmate(team, hasTimeAdvantage)) {
                winner = team == Stockfish::WHITE ? 1 : 0;
                termination = "checkmate";
                break;
            }
            if (board.is_draw()) {
                termination = "draw";
                break;
            }

            if (rawInitializationActive && macroPly < initializationLength) {
                const JointActionCandidate rawAction = sample_raw_policy_action(
                    engine, rawPolicyEvaluator, board, team, hasTimeAdvantage,
                    sample_raw_policy_temperature(config, randomEngine), randomEngine);
                if (!action_leads_to_terminal(
                        board, rawAction, team, hasTimeAdvantage)) {
                    if (rawAction.moveA != Stockfish::MOVE_NONE) {
                        append_pgn_move(
                            board, BOARD_A, rawAction.moveA, moveNumbers, pgnMoves.size(),
                            config.initialClockSeconds, pgnMoves);
                        board.push_move(BOARD_A, rawAction.moveA);
                    }
                    if (rawAction.moveB != Stockfish::MOVE_NONE) {
                        append_pgn_move(
                            board, BOARD_B, rawAction.moveB, moveNumbers, pgnMoves.size(),
                            config.initialClockSeconds, pgnMoves);
                        board.push_move(BOARD_B, rawAction.moveB);
                    }
                    rawPolicyMacroPlies++;
                    rawPolicyEvents = pgnMoves.size();
                    team = ~team;
                    hasTimeAdvantage = !hasTimeAdvantage;
                    continue;
                }
                rawInitializationActive = false;
            }

            TrainingSample sample;
            sample.gameId = gameIndex;
            sample.macroPly = static_cast<uint16_t>(std::min<size_t>(
                macroPly, std::numeric_limits<uint16_t>::max()));
            sample.team = team == Stockfish::WHITE ? 0 : 1;
            sample.hasTimeAdvantage = hasTimeAdvantage ? 1 : 0;
            sample.planes = encode_planes(board, team, hasTimeAdvantage);

            agent.reset_search_state();
            SearchOptions searchOptions;
            searchOptions.targetNodes = randomized_node_budget(config, randomEngine);
            searchOptions.search.rootDirichletAlpha = config.dirichletAlpha;
            searchOptions.search.rootDirichletEpsilon = config.dirichletEpsilon;
            searchOptions.search.waitPassPriorFloor = config.waitPassPriorFloor;
            searchOptions.search.coordinationPassPriorFloor = config.coordinationPassPriorFloor;
            searchOptions.search.rootNoiseSeed = mix_seed(runId, gameIndex * config.maxMacroPlies + macroPly);
            agent.run_search(board, engines, team, hasTimeAdvantage, searchOptions);
            const std::vector<RootEdgeStats> edges = agent.root_edge_stats();
            if (edges.empty()) {
                winner = team == Stockfish::WHITE ? 1 : 0;
                termination = "no legal action";
                break;
            }
            const uint64_t actualVisits = std::accumulate(
                edges.begin(), edges.end(), uint64_t{0},
                [](uint64_t total, const RootEdgeStats& edge) {
                    return total + static_cast<uint64_t>(std::max(0, edge.visits));
                });
            sample.nodes = static_cast<uint32_t>(std::min<uint64_t>(
                actualVisits, std::numeric_limits<uint32_t>::max()));

            sample.policyA = marginal_policy(board, BOARD_A, edges);
            sample.policyB = marginal_policy(board, BOARD_B, edges);
            samples.push_back(std::move(sample));

            const JointActionCandidate action = select_action(
                edges, mcts_temperature(config, macroPly), randomEngine);
            if (action.moveA == Stockfish::MOVE_NONE && action.moveB == Stockfish::MOVE_NONE) {
                team = ~team;
                hasTimeAdvantage = !hasTimeAdvantage;
                continue;
            }

            if (action.moveA != Stockfish::MOVE_NONE) {
                append_pgn_move(
                    board, BOARD_A, action.moveA, moveNumbers, pgnMoves.size(),
                    config.initialClockSeconds, pgnMoves);
                board.push_move(BOARD_A, action.moveA);
            }
            if (action.moveB != Stockfish::MOVE_NONE) {
                append_pgn_move(
                    board, BOARD_B, action.moveB, moveNumbers, pgnMoves.size(),
                    config.initialClockSeconds, pgnMoves);
                board.push_move(BOARD_B, action.moveB);
            }

            team = ~team;
            hasTimeAdvantage = !hasTimeAdvantage;
        }

        if (winner < 0 && board.is_checkmate(team, hasTimeAdvantage)) {
            winner = team == Stockfish::WHITE ? 1 : 0;
            termination = "checkmate";
        } else if (winner < 0 && board.is_draw()) {
            termination = "draw";
        }

        for (size_t index = 0; index < samples.size(); ++index) {
            TrainingSample& sample = samples[index];
            sample.outcome = winner < 0 ? 0 : (sample.team == winner ? 1 : -1);
            sample.wdl = static_cast<uint8_t>(sample.outcome + 1);
            sample.movesLeft = static_cast<uint16_t>(std::min<size_t>(
                samples.size() - index, std::numeric_limits<uint16_t>::max()));
        }
        const size_t sampleCount = samples.size();
        chunkWriter.append(std::move(samples));
        append_pgn(
            pgnPath, gameIndex + 1, pgnMoves, winner, startingTeam,
            rawPolicyMacroPlies, rawPolicyEvents,
            config.initialClockSeconds, termination);
        std::cout << "selfplay game " << (gameIndex + 1) << '/' << config.games
                  << " raw " << rawPolicyMacroPlies
                  << " samples " << sampleCount
                  << " events " << pgnMoves.size()
                  << " termination " << termination << '\n';
    }

    chunkWriter.finish();
    return 0;
}
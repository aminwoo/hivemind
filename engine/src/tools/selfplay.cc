#include "tools/selfplay.h"

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
#include <atomic>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "search/agent.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "nn/engine.h"
#include "common/globals.h"
#include "environment/planes.h"
#include "common/utils.h"
#include "tools/distill_data.h"
#include "tools/selfplay_policy.h"

namespace {

constexpr std::array<char, 4> CHUNK_MAGIC = {'H', 'V', 'M', '5'};
constexpr uint32_t CHUNK_VERSION = 5;
constexpr uint16_t PASS_POLICY_INDEX = static_cast<uint16_t>(NB_POLICY_VALUES());

struct SparsePolicyEntry {
    uint16_t index;
    float probability;
};

struct SparseJointPolicyEntry {
    uint16_t indexA;
    uint16_t indexB;
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
    float rootQ = 0.0f;
    std::array<uint8_t, NB_INPUT_VALUES()> planes{};
    std::vector<SparsePolicyEntry> policyA;
    std::vector<SparsePolicyEntry> policyB;
    std::vector<SparseJointPolicyEntry> jointPolicy;
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
            throw std::runtime_error("Sparse policy is too large for HVM5");
        }
        write_scalar(stream, static_cast<uint16_t>(policy.size()));
        for (const SparsePolicyEntry& entry : policy) {
            write_scalar(stream, entry.index);
            write_scalar(stream, entry.probability);
        }
    }

    void write_joint_policy(
        std::ostream& stream,
        const std::vector<SparseJointPolicyEntry>& policy) {
        if (policy.size() > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("Sparse joint policy is too large for HVM5");
        }
        write_scalar(stream, static_cast<uint16_t>(policy.size()));
        for (const SparseJointPolicyEntry& entry : policy) {
            write_scalar(stream, entry.indexA);
            write_scalar(stream, entry.indexB);
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
            write_scalar(stream, sample.rootQ);
            stream.write(
                reinterpret_cast<const char*>(sample.planes.data()),
                static_cast<std::streamsize>(sample.planes.size()));
            write_policy(stream, sample.policyA);
            write_policy(stream, sample.policyB);
            write_joint_policy(stream, sample.jointPolicy);
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
    if (config.mctsTemperaturePlies > 0 && macroPly >= config.mctsTemperaturePlies) {
        return 0.0;
    }
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

    auto capture_flags = [&board](const std::vector<Stockfish::Move>& actions, int boardNumber) {
        std::vector<uint8_t> captures;
        captures.reserve(actions.size());
        for (Stockfish::Move move : actions) {
            captures.push_back(move != Stockfish::MOVE_NONE
                && board.is_capture(boardNumber, move) ? 1 : 0);
        }
        return captures;
    };
    const std::vector<uint8_t> capturesA = capture_flags(actionsA, BOARD_A);
    const std::vector<uint8_t> capturesB = capture_flags(actionsB, BOARD_B);

    JointActionRules rules;
    rules.boardAOnTurn = boardAOnTurn;
    rules.boardBOnTurn = boardBOnTurn;
    rules.teamHasTimeAdvantage = hasTimeAdvantage;
    rules.boardACanMove = boardAOnTurn && actionsA.size() > 1;
    rules.boardBCanMove = boardBOnTurn && actionsB.size() > 1;

    auto make_candidate = [&](size_t iA, size_t iB) {
        return JointActionCandidate(
            actionsA[iA], probabilitiesA[iA], iA,
            actionsB[iB], probabilitiesB[iB], iB,
            rules, capturesA[iA] != 0, capturesB[iB] != 0);
    };

    std::discrete_distribution<size_t> sampleA(probabilitiesA.begin(), probabilitiesA.end());
    std::discrete_distribution<size_t> sampleB(probabilitiesB.begin(), probabilitiesB.end());
    JointActionCandidate candidate = make_candidate(sampleA(randomEngine), sampleB(randomEngine));
    if (candidate.jointPrior >= 0.0f) {
        return candidate;
    }

    // The independent sample produced an illegal joint action, so resample from
    // the legal pairs only.
    std::vector<std::pair<size_t, size_t>> legalPairs;
    std::vector<double> weights;
    for (size_t iA = 0; iA < actionsA.size(); ++iA) {
        for (size_t iB = 0; iB < actionsB.size(); ++iB) {
            if (make_candidate(iA, iB).jointPrior < 0.0f) {
                continue;
            }
            legalPairs.emplace_back(iA, iB);
            weights.push_back(
                static_cast<double>(probabilitiesA[iA]) * static_cast<double>(probabilitiesB[iB]));
        }
    }
    if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
        throw std::runtime_error("Raw policy produced no legal joint action");
    }
    std::discrete_distribution<size_t> legalSample(weights.begin(), weights.end());
    const auto [indexA, indexB] = legalPairs[legalSample(randomEngine)];
    return make_candidate(indexA, indexB);
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
    const int idx = get_fast_policy_index(move, board.side_to_move(boardNumber));
    if (idx < 0) {
        throw std::runtime_error("Move is absent from policy map: " + board.uci_move(boardNumber, move));
    }
    return idx;
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

/// A board's search policy over all of its legal moves and the pass (zero for
/// the ones the search did not visit), for distillation. Empty when the board
/// is not on turn.
std::vector<distill::PolicyEntry> distill_search_policy(
    Board& board,
    int boardNumber,
    bool onTurn,
    const std::vector<RootEdgeStats>& edges) {
    std::vector<distill::PolicyEntry> entries;
    if (!onTurn) {
        return entries;
    }
    std::map<Stockfish::Move, uint64_t> visitsByMove;
    uint64_t totalVisits = 0;
    for (const RootEdgeStats& edge : edges) {
        if (edge.visits <= 0) {
            continue;
        }
        const Stockfish::Move move = boardNumber == BOARD_A ? edge.action.moveA : edge.action.moveB;
        visitsByMove[move] += static_cast<uint64_t>(edge.visits);
        totalVisits += static_cast<uint64_t>(edge.visits);
    }
    if (totalVisits == 0) {
        throw std::runtime_error("Search returned no visited root edges");
    }
    const std::vector<Stockfish::Move> actions = distill::policy_actions(board, boardNumber);
    entries.reserve(actions.size());
    uint64_t covered = 0;
    for (const Stockfish::Move move : actions) {
        const auto found = visitsByMove.find(move);
        const uint64_t visits = found == visitsByMove.end() ? 0 : found->second;
        covered += visits;
        entries.push_back({static_cast<uint16_t>(policy_index(board, boardNumber, move)),
                           distill::half_bits(__float2half_rn(
                               static_cast<float>(visits) / static_cast<float>(totalVisits)))});
    }
    if (covered != totalVisits) {
        throw std::runtime_error("Search visited a move outside the legal policy moves");
    }
    return entries;
}

uint16_t joint_policy_index(Board& board, int boardNumber,
                            Stockfish::Move move) {
    if (move == Stockfish::MOVE_NONE) {
        return PASS_POLICY_INDEX;
    }
    return static_cast<uint16_t>(policy_index(board, boardNumber, move));
}

std::vector<SparseJointPolicyEntry> joint_policy(
    Board& board, const std::vector<RootEdgeStats>& edges) {
    std::map<std::pair<uint16_t, uint16_t>, uint64_t> visitsByAction;
    uint64_t totalVisits = 0;
    for (const RootEdgeStats& edge : edges) {
        if (edge.visits <= 0) {
            continue;
        }
        const auto key = std::make_pair(
            joint_policy_index(board, BOARD_A, edge.action.moveA),
            joint_policy_index(board, BOARD_B, edge.action.moveB));
        visitsByAction[key] += static_cast<uint64_t>(edge.visits);
        totalVisits += static_cast<uint64_t>(edge.visits);
    }
    if (totalVisits == 0) {
        throw std::runtime_error("Search returned no visited root edges");
    }

    std::vector<SparseJointPolicyEntry> policy;
    policy.reserve(visitsByAction.size());
    for (const auto& [indices, visits] : visitsByAction) {
        policy.push_back({
            indices.first,
            indices.second,
            static_cast<float>(visits) / static_cast<float>(totalVisits),
        });
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
#if defined(_WIN32)
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif
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

int run_selfplay(const std::vector<Engine*>& engines, const SelfPlayConfig& config) {
    if (config.games == 0 || config.nodes == 0 || config.maxMacroPlies == 0) {
        throw std::invalid_argument("games, nodes, and max-macro-plies must be positive");
    }
    if (engines.empty()) {
        throw std::invalid_argument("Self-play requires at least one inference engine");
    }
    if (config.rawPolicyMeanMacroPlies < 0.0
        || config.rawPolicyHighTemperatureProbability < 0.0
        || config.rawPolicyHighTemperatureProbability > 1.0
        || config.mctsTemperature <= 0.0
        || config.mctsTemperatureDecay <= 0.0
        || config.mctsTemperatureDecay > 1.0
        || config.nodeRandomFactor < 0.0
        || config.nodeRandomFactor >= 1.0) {
        throw std::invalid_argument("Invalid self-play exploration configuration");
    }
    const size_t gameThreads = std::max<size_t>(1, config.parallelGames);
    std::set<Engine*> uniqueEngines(engines.begin(), engines.end());
    if (uniqueEngines.size() != engines.size() || uniqueEngines.count(nullptr)
        || engines.size() < gameThreads) {
        throw std::invalid_argument("Self-play requires a distinct engine per game thread");
    }
    if (std::filesystem::exists(config.outputDirectory)
        && !std::filesystem::is_empty(config.outputDirectory)) {
        throw std::invalid_argument("Self-play output directory must be fresh");
    }
    if (!config.distillOutputDirectory.empty()
        && std::filesystem::exists(config.distillOutputDirectory)
        && !std::filesystem::is_empty(config.distillOutputDirectory)) {
        throw std::invalid_argument("Self-play distillation directory must be fresh");
    }
    const uint64_t runId = config.seed != 0
        ? config.seed
        : static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    const std::filesystem::path trainingDirectory = config.outputDirectory / "training_data";
    std::filesystem::create_directories(config.outputDirectory);
    ChunkWriter chunkWriter(trainingDirectory, config.chunkSamples, runId);
    const std::filesystem::path pgnPath = config.outputDirectory / "games.pgn";
    std::unique_ptr<distill::ChunkWriter> distillWriter;
    if (!config.distillOutputDirectory.empty()) {
        distillWriter = std::make_unique<distill::ChunkWriter>(
            config.distillOutputDirectory, "search_" + std::to_string(runId),
            config.distillChunkPositions);
    }
    // Games run in `parallelGames` threads, each with its own engine, so the
    // GPU overlaps many small searches; outputs are serialised.
    std::atomic<size_t> nextGame{0};
    std::mutex outputMutex;
    std::exception_ptr failure;
    // Finished games waiting for every earlier game to be written (guarded by
    // outputMutex), so the corpus order depends only on the seed.
    struct FinishedGame {
        std::vector<TrainingSample> samples;
        std::vector<distill::Record> distillRecords;
        std::vector<PgnMove> pgnMoves;
        int winner;
        int startingTeam;
        size_t rawPolicyMacroPlies;
        size_t rawPolicyEvents;
        std::string termination;
    };
    std::map<size_t, FinishedGame> finishedGames;
    size_t nextOutput = 0;
    const auto writeGame = [&](size_t gameNumber, FinishedGame& game) {
        if (distillWriter) {
            // Finished games give WDL (loss, draw, win) and moves-left targets
            // (remaining team decisions, capped at 100, over 100, the scale
            // the search assumes). A game cut off at the macro-ply limit has
            // no real result: its WDL stays zero, which the trainer masks.
            const bool finished = game.termination != "macro-ply limit";
            for (size_t index = 0; index < game.distillRecords.size(); ++index) {
                distill::Record& record = game.distillRecords[index];
                record.outcome = game.samples[index].outcome;
                if (finished) {
                    record.wdl = {0.0f, 0.0f, 0.0f};
                    record.wdl[static_cast<size_t>(game.samples[index].outcome + 1)] = 1.0f;
                    const size_t remaining = std::min<size_t>(game.samples.size() - index, 100);
                    record.movesLeft = distill::half_bits(
                        __float2half_rn(static_cast<float>(remaining) / 100.0f));
                }
                distillWriter->append(std::move(record));
            }
        }
        const size_t sampleCount = game.samples.size();
        if (config.writeTrainingChunks) {
            chunkWriter.append(std::move(game.samples));
        }
        append_pgn(
            pgnPath, gameNumber + 1, game.pgnMoves, game.winner, game.startingTeam,
            game.rawPolicyMacroPlies, game.rawPolicyEvents,
            config.initialClockSeconds, game.termination);
        std::cout << "selfplay game " << (gameNumber + 1) << '/' << config.games
                  << " raw " << game.rawPolicyMacroPlies
                  << " samples " << sampleCount
                  << " events " << game.pgnMoves.size()
                  << " termination " << game.termination << '\n'
                  // Piped to `hivemind contribute`, which counts these lines.
                  << std::flush;
    };
    auto playGames = [&](const std::vector<Engine*>& threadEngines) {
        std::vector<__half> distillPlanes(NB_INPUT_VALUES());
        Engine& rawPolicyEngine = *threadEngines.front();
        RawPolicyEvaluator rawPolicyEvaluator(
            static_cast<size_t>(rawPolicyEngine.getBatchSize()));

        for (;;) {
            const size_t gameIndex = nextGame.fetch_add(1);
            if (gameIndex >= config.games) {
                break;
            }
            std::mt19937_64 randomEngine(selfplay_seed(runId, gameIndex));
            Board board;
            Agent agent;
            const int startingTeam = static_cast<int>(randomEngine() & 1ULL);
            Stockfish::Color team = startingTeam == 0 ? Stockfish::WHITE : Stockfish::BLACK;
            bool hasTimeAdvantage = false;
            int winner = -1;
            std::string termination = "macro-ply limit";
            std::vector<TrainingSample> samples;
            std::vector<distill::Record> distillRecords;
            std::vector<PgnMove> pgnMoves;
            std::array<int, 2> moveNumbers = {1, 1};
            const size_t initializationLength = sample_initialization_length(config, randomEngine);
            bool rawInitializationActive = initializationLength > 0;
            size_t rawPolicyMacroPlies = 0;
            size_t rawPolicyEvents = 0;
            const bool canResign = config.resignThreshold < 0.0f
                && (config.resignDisableFraction <= 0.0
                    || std::uniform_real_distribution<double>(0.0, 1.0)(randomEngine) >= config.resignDisableFraction);
            std::array<size_t, 2> consecutiveResignPlies = {0, 0};

            for (size_t macroPly = 0; macroPly < config.maxMacroPlies; ++macroPly) {
                const GameStatus status = adjudicate_game(board, team, hasTimeAdvantage);
                if (status != GameStatus::ONGOING) {
                    winner = adjudicated_winner(status, team);
                    termination = adjudicated_termination(status);
                    break;
                }

                if (rawInitializationActive && macroPly < initializationLength) {
                    const JointActionCandidate rawAction = sample_raw_policy_action(
                        rawPolicyEngine, rawPolicyEvaluator, board, team, hasTimeAdvantage,
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
                searchOptions.search.enableMateProbe = config.fairyStockfishMateNodes > 0;
                searchOptions.mateProbeNodes = config.fairyStockfishMateNodes;
                searchOptions.completeMateProbe = config.fairyStockfishMateNodes > 0;
                // Self-play policy targets and temperature sampling are visit-based.
                // Keep that data-generation contract until it has a dedicated
                // Gumbel-improved policy target rather than biased halving visits.
                searchOptions.search.enableGumbelRootSearch = false;
                if (!config.rootScans) {
                    searchOptions.search.enableRootMateSearch = false;
                    searchOptions.search.certifySelectedMove = false;
                }
                searchOptions.search.rootDirichletAlpha = config.dirichletAlpha;
                searchOptions.search.rootDirichletEpsilon = config.dirichletEpsilon;
                searchOptions.search.rootNoiseSeed = selfplay_seed(runId, gameIndex * config.maxMacroPlies + macroPly);
                const JointActionCandidate choice = agent.run_search(
                    board, threadEngines, team, hasTimeAdvantage, searchOptions);
                std::vector<RootEdgeStats> edges = agent.root_edge_stats();
                if (agent.search_status() != GameStatus::ONGOING) {
                    winner = adjudicated_winner(agent.search_status(), team);
                    termination = adjudicated_termination(agent.search_status());
                    break;
                }
                if (edges.empty()) {
                    throw std::runtime_error("Ongoing search returned no root edges");
                }
                const uint64_t actualVisits = std::accumulate(
                    edges.begin(), edges.end(), uint64_t{0},
                    [](uint64_t total, const RootEdgeStats& edge) {
                        return total + static_cast<uint64_t>(std::max(0, edge.visits));
                    });
                sample.nodes = static_cast<uint32_t>(std::min<uint64_t>(
                    actualVisits, std::numeric_limits<uint32_t>::max()));

                edges = selfplay_policy_edges(edges, agent.root_type(), choice, agent.certified_choice());
                const float rootQ = agent.root_q();
                sample.rootQ = rootQ;
                sample.policyA = marginal_policy(board, BOARD_A, edges);
                sample.policyB = marginal_policy(board, BOARD_B, edges);
                sample.jointPolicy = joint_policy(board, edges);
                if (distillWriter) {
                    // WDL stays zero: search has a value but no WDL distribution,
                    // so these records carry no WDL or moves-left target.
                    distill::Record record;
                    board_to_planes(board, distillPlanes.data(), team, hasTimeAdvantage);
                    distill::pack_planes(distillPlanes.data(), record);
                    record.value = rootQ;
                    record.ply = sample.macroPly;
                    record.policy[0] = distill_search_policy(
                        board, BOARD_A, board.side_to_move(BOARD_A) == team, edges);
                    record.policy[1] = distill_search_policy(
                        board, BOARD_B, board.side_to_move(BOARD_B) == ~team, edges);
                    distillRecords.push_back(std::move(record));
                }
                samples.push_back(std::move(sample));

                const size_t teamIdx = team == Stockfish::WHITE ? 0 : 1;
                if (canResign) {
                    if (rootQ <= config.resignThreshold) {
                        consecutiveResignPlies[teamIdx]++;
                        if (consecutiveResignPlies[teamIdx] >= config.resignConsecutivePlies) {
                            winner = team == Stockfish::WHITE ? 1 : 0;
                            termination = "resignation";
                            break;
                        }
                    } else {
                        consecutiveResignPlies[teamIdx] = 0;
                    }
                }

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

            const GameStatus finalStatus = adjudicate_game(board, team, hasTimeAdvantage);
            if (winner < 0 && finalStatus != GameStatus::ONGOING) {
                winner = adjudicated_winner(finalStatus, team);
                termination = adjudicated_termination(finalStatus);
            }

            for (size_t index = 0; index < samples.size(); ++index) {
                TrainingSample& sample = samples[index];
                sample.outcome = winner < 0 ? 0 : (sample.team == winner ? 1 : -1);
                sample.wdl = static_cast<uint8_t>(sample.outcome + 1);
                sample.movesLeft = static_cast<uint16_t>(std::min<size_t>(
                    samples.size() - index, std::numeric_limits<uint16_t>::max()));
            }
            // Hand the game to the writer; games are written in index order, but a
            // finished thread starts its next game instead of waiting for earlier
            // (possibly much longer) games to finish.
            const std::lock_guard<std::mutex> lock(outputMutex);
            if (failure) {
                return;
            }
            finishedGames.emplace(gameIndex, FinishedGame{
                std::move(samples), std::move(distillRecords), std::move(pgnMoves), winner,
                startingTeam, rawPolicyMacroPlies, rawPolicyEvents, std::move(termination)});
            for (auto next = finishedGames.find(nextOutput); next != finishedGames.end();
                 next = finishedGames.find(nextOutput)) {
                writeGame(next->first, next->second);
                finishedGames.erase(next);
                ++nextOutput;
            }
        }
    };

    if (gameThreads == 1) {
        playGames(engines);
    } else {
        std::vector<std::thread> threads;
        for (size_t threadIndex = 0; threadIndex < gameThreads; ++threadIndex) {
            threads.emplace_back([&, threadIndex] {
                try {
                    playGames({engines[threadIndex]});
                } catch (...) {
                    const std::lock_guard<std::mutex> lock(outputMutex);
                    if (!failure) {
                        failure = std::current_exception();
                    }
                    nextGame = config.games;  // stop the other threads
                }
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

    chunkWriter.finish();
    if (distillWriter) {
        distillWriter->flush();
    }
    return 0;
}

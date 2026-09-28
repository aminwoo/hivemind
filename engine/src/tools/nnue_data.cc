#include "tools/nnue_data.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/utils.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "environment/joint_action.h"
#include "environment/planes.h"
#include "nn/backend_compat.h"
#include "nn/engine.h"
#include "nnue/features.h"
#include "search/search_params.h"

namespace nnue_datagen {

constexpr std::array<char, 4> NNUE_DATA_MAGIC = {'H', 'N', 'U', 'E'};
constexpr uint32_t NNUE_DATA_VERSION = 1;
constexpr int8_t OUTCOME_UNKNOWN = 2;

struct PositionRecord {
    std::array<uint16_t, nnue::MAX_ACTIVE_FEATURES> features{};
    uint8_t featureCount = 0;
    uint8_t team = 0;
    uint16_t ply = 0;
    float value = 0.0f;
    std::array<float, 3> wdl{};
    std::string fen;
};

/**
 * @brief Column-major chunk writer.
 *
 * Layout: magic, version, feature count, positions N, total active features F,
 * then value f32[N], wdl f32[N*3] (loss, draw, win), outcome i8[N],
 * ply u16[N], count u8[N], features u16[F]. Outcome is from the side to move:
 * -1 loss, 0 draw, 1 win, 2 unfinished.
 */
class NnueChunkWriter {
public:
    NnueChunkWriter(std::filesystem::path directory, std::string prefix,
                    size_t chunkPositions, bool writeFens)
        : directory_(std::move(directory)), prefix_(std::move(prefix)),
          chunkPositions_(std::max<size_t>(1, chunkPositions)),
          writeFens_(writeFens) {}

    void append(PositionRecord&& record, int8_t outcome) {
        records_.push_back(std::move(record));
        outcomes_.push_back(outcome);
        if (records_.size() >= chunkPositions_) {
            flush();
        }
    }

    void flush() {
        if (records_.empty()) {
            return;
        }
        std::ostringstream name;
        name << prefix_ << '_' << std::setw(5) << std::setfill('0') << chunkIndex_++;
        const std::filesystem::path finalPath = directory_ / (name.str() + ".bin");
        const std::filesystem::path temporaryPath = finalPath.string() + ".tmp";
        std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("Unable to create " + temporaryPath.string());
        }
        const uint64_t count = records_.size();
        uint64_t totalFeatures = 0;
        for (const PositionRecord& record : records_) {
            totalFeatures += record.featureCount;
        }
        auto write = [&stream](const void* data, size_t bytes) {
            stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        };
        const uint32_t featureSpace = nnue::NUM_FEATURES;
        write(NNUE_DATA_MAGIC.data(), NNUE_DATA_MAGIC.size());
        write(&NNUE_DATA_VERSION, sizeof(NNUE_DATA_VERSION));
        write(&featureSpace, sizeof(featureSpace));
        write(&count, sizeof(count));
        write(&totalFeatures, sizeof(totalFeatures));
        for (const PositionRecord& record : records_) write(&record.value, sizeof(float));
        for (const PositionRecord& record : records_) write(record.wdl.data(), 3 * sizeof(float));
        write(outcomes_.data(), outcomes_.size());
        for (const PositionRecord& record : records_) write(&record.ply, sizeof(uint16_t));
        for (const PositionRecord& record : records_) write(&record.featureCount, 1);
        for (const PositionRecord& record : records_) {
            write(record.features.data(), record.featureCount * sizeof(uint16_t));
        }
        stream.close();
        if (!stream) {
            throw std::runtime_error("Failed to write " + temporaryPath.string());
        }
        std::filesystem::rename(temporaryPath, finalPath);

        if (writeFens_) {
            std::ofstream fens(directory_ / (name.str() + ".fen"), std::ios::trunc);
            for (const PositionRecord& record : records_) {
                fens << record.fen << '\n';
            }
        }
        records_.clear();
        outcomes_.clear();
    }

private:
    std::filesystem::path directory_;
    std::string prefix_;
    size_t chunkPositions_;
    bool writeFens_;
    size_t chunkIndex_ = 0;
    std::vector<PositionRecord> records_;
    std::vector<int8_t> outcomes_;
};

struct GameSlot {
    std::unique_ptr<Board> board;
    Stockfish::Color team = Stockfish::WHITE;
    bool hasTimeAdvantage = false;
    size_t macroPly = 0;
    double temperature = 1.0;
    double randomMoveProbability = 0.0;
    std::vector<PositionRecord> records;
};

void apply_temperature(std::vector<float>& probabilities, double temperature) {
    if (temperature == 1.0) {
        return;
    }
    const double exponent = 1.0 / temperature;
    double total = 0.0;
    for (float& probability : probabilities) {
        probability = static_cast<float>(std::pow(std::max(0.0f, probability), exponent));
        total += probability;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        std::fill(probabilities.begin(), probabilities.end(),
                  1.0f / static_cast<float>(probabilities.size()));
        return;
    }
    for (float& probability : probabilities) {
        probability = static_cast<float>(probability / total);
    }
}

/// Legal moves (plus the pass) on one board and their sampling weights.
void board_actions(Board& board, int boardNumber, bool onTurn,
                   const __half* policy, const GameSlot& slot,
                   std::mt19937_64& rng,
                   std::vector<Stockfish::Move>& actions,
                   std::vector<float>& probabilities,
                   std::vector<uint8_t>& captures) {
    actions.clear();
    probabilities.clear();
    captures.clear();
    if (onTurn) {
        actions = board.legal_moves(boardNumber);
        std::erase_if(actions, [&board, boardNumber](Stockfish::Move move) {
            return !is_policy_move_representable(board, boardNumber, move);
        });
    }
    const bool hasMoves = !actions.empty();
    actions.push_back(Stockfish::MOVE_NONE);
    if (!hasMoves) {
        probabilities.push_back(1.0f);
    } else if (std::uniform_real_distribution<double>(0.0, 1.0)(rng)
               < slot.randomMoveProbability) {
        probabilities.assign(actions.size(), 1.0f / static_cast<float>(actions.size()));
    } else {
        probabilities = get_normalized_probability(policy, actions, boardNumber, board);
        apply_temperature(probabilities, slot.temperature);
    }
    for (Stockfish::Move move : actions) {
        captures.push_back(move != Stockfish::MOVE_NONE
            && board.is_capture(boardNumber, move) ? 1 : 0);
    }
}

/// Samples a legal joint action, or returns false when the team has none.
bool sample_joint_action(Board& board, GameSlot& slot,
                         const __half* policyA, const __half* policyB,
                         std::mt19937_64& rng,
                         Stockfish::Move& moveA, Stockfish::Move& moveB) {
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == slot.team;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~slot.team;
    thread_local std::vector<Stockfish::Move> actionsA, actionsB;
    thread_local std::vector<float> probabilitiesA, probabilitiesB;
    thread_local std::vector<uint8_t> capturesA, capturesB;
    board_actions(board, BOARD_A, boardAOnTurn, policyA, slot, rng,
                  actionsA, probabilitiesA, capturesA);
    board_actions(board, BOARD_B, boardBOnTurn, policyB, slot, rng,
                  actionsB, probabilitiesB, capturesB);

    JointActionRules rules;
    rules.boardAOnTurn = boardAOnTurn;
    rules.boardBOnTurn = boardBOnTurn;
    rules.teamHasTimeAdvantage = slot.hasTimeAdvantage;
    rules.boardACanMove = boardAOnTurn && actionsA.size() > 1;
    rules.boardBCanMove = boardBOnTurn && actionsB.size() > 1;
    auto legal = [&](size_t iA, size_t iB) {
        return is_joint_action_legal(rules, actionsA[iA], actionsB[iB],
                                     capturesA[iA] != 0, capturesB[iB] != 0);
    };

    std::discrete_distribution<size_t> sampleA(probabilitiesA.begin(), probabilitiesA.end());
    std::discrete_distribution<size_t> sampleB(probabilitiesB.begin(), probabilitiesB.end());
    size_t iA = sampleA(rng);
    size_t iB = sampleB(rng);
    if (!legal(iA, iB)) {
        std::vector<std::pair<size_t, size_t>> pairs;
        std::vector<double> weights;
        for (size_t a = 0; a < actionsA.size(); ++a) {
            for (size_t b = 0; b < actionsB.size(); ++b) {
                if (legal(a, b)) {
                    pairs.emplace_back(a, b);
                    weights.push_back(std::max(1e-12,
                        static_cast<double>(probabilitiesA[a]) * probabilitiesB[b]));
                }
            }
        }
        if (pairs.empty()) {
            return false;
        }
        std::discrete_distribution<size_t> samplePair(weights.begin(), weights.end());
        std::tie(iA, iB) = pairs[samplePair(rng)];
    }
    moveA = actionsA[iA];
    moveB = actionsB[iB];
    return true;
}

void reset_slot(GameSlot& slot, std::mt19937_64& rng, double baseRandomMoveProbability) {
    slot.board = std::make_unique<Board>();
    slot.team = (rng() & 1ULL) ? Stockfish::WHITE : Stockfish::BLACK;
    slot.hasTimeAdvantage = (rng() & 1ULL) != 0;
    slot.macroPly = 0;
    slot.records.clear();
    const double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    slot.temperature = roll < 0.6 ? 1.0 : (roll < 0.85 ? 1.5 : 3.0);
    const double chaos = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    slot.randomMoveProbability = chaos < 0.5 ? 0.0
        : (chaos < 0.85 ? baseRandomMoveProbability : 3.0 * baseRandomMoveProbability);
}

/// Flushes a finished game. `winner` is the winning team, or -1 for a draw
/// and -2 for an unfinished game.
void finish_game(GameSlot& slot, int winner, NnueChunkWriter& writer,
                 std::atomic<uint64_t>& written) {
    for (PositionRecord& record : slot.records) {
        int8_t outcome = OUTCOME_UNKNOWN;
        if (winner == -1) {
            outcome = 0;
        } else if (winner >= 0) {
            outcome = record.team == winner ? 1 : -1;
        }
        writer.append(std::move(record), outcome);
    }
    written += slot.records.size();
    slot.records.clear();
}

int team_index(Stockfish::Color team) {
    return team == Stockfish::WHITE ? 0 : 1;
}

void generation_worker(Engine& engine, const NnueDataConfig& config,
                       size_t workerIndex, uint64_t runSeed,
                       std::atomic<uint64_t>& written,
                       std::atomic<uint64_t>& games,
                       std::mutex& logMutex) {
    const size_t batchSize = static_cast<size_t>(engine.getBatchSize());
    std::mt19937_64 rng(runSeed ^ (0x9e3779b97f4a7c15ULL * (workerIndex + 1)));
    std::ostringstream prefix;
    prefix << "nnue_" << runSeed << '_' << workerIndex;
    NnueChunkWriter writer(config.outputDirectory, prefix.str(),
                           config.chunkPositions, config.writeFens);

    __half* observations = nullptr;
    if (!hm::alloc_pinned(reinterpret_cast<void**>(&observations),
                          batchSize * NB_INPUT_VALUES() * sizeof(__half))) {
        throw std::runtime_error("Unable to allocate pinned observations");
    }
    std::unique_ptr<__half, void (*)(__half*)> observationGuard(
        observations, [](__half* ptr) { hm::free_pinned(ptr); });

    std::vector<GameSlot> slots(batchSize);
    for (GameSlot& slot : slots) {
        reset_slot(slot, rng, config.randomMoveProbability);
    }
    const uint64_t quota = config.positions / config.threads
        + (workerIndex < config.positions % config.threads ? 1 : 0);
    uint64_t workerWritten = 0;
    std::atomic<uint64_t> localWritten{0};

    while (localWritten.load() < quota) {
        // Resolve finished games so every slot holds a live position.
        for (GameSlot& slot : slots) {
            for (;;) {
                Board& board = *slot.board;
                int winner = -3;
                if (board.is_checkmate(~slot.team, !slot.hasTimeAdvantage)) {
                    winner = team_index(slot.team);
                } else if (board.is_checkmate(slot.team, slot.hasTimeAdvantage)) {
                    winner = team_index(~slot.team);
                } else if (board.is_draw()) {
                    winner = -1;
                } else if (slot.macroPly >= config.maxMacroPlies) {
                    winner = -2;
                }
                if (winner == -3) {
                    break;
                }
                finish_game(slot, winner, writer, localWritten);
                ++games;
                reset_slot(slot, rng, config.randomMoveProbability);
            }
        }

        for (size_t index = 0; index < batchSize; ++index) {
            GameSlot& slot = slots[index];
            board_to_planes(*slot.board, observations + index * NB_INPUT_VALUES(),
                            slot.team, slot.hasTimeAdvantage);
        }
        Engine::HalfInferenceOutputs outputs;
        if (!engine.runInferenceHalf(observations, outputs, workerIndex)) {
            throw std::runtime_error("NNUE data inference failed");
        }

        for (size_t index = 0; index < batchSize; ++index) {
            GameSlot& slot = slots[index];
            Board& board = *slot.board;

            PositionRecord record;
            record.featureCount = static_cast<uint8_t>(nnue::extract_features(
                board, slot.team, slot.hasTimeAdvantage, record.features.data()));
            record.team = static_cast<uint8_t>(team_index(slot.team));
            record.ply = static_cast<uint16_t>(std::min<size_t>(slot.macroPly, 65535));
            record.value = __half2float(outputs.value[index]);
            if (outputs.wdl) {
                std::array<float, 3> logits;
                float maxLogit = -1e30f;
                for (int k = 0; k < 3; ++k) {
                    logits[k] = __half2float(outputs.wdl[index * 3 + k]);
                    maxLogit = std::max(maxLogit, logits[k]);
                }
                float total = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    record.wdl[k] = std::exp(logits[k] - maxLogit);
                    total += record.wdl[k];
                }
                for (float& probability : record.wdl) {
                    probability /= total;
                }
            }
            if (config.writeFens) {
                record.fen = board.fen(BOARD_A) + ';' + board.fen(BOARD_B) + ';'
                    + (slot.team == Stockfish::WHITE ? 'w' : 'b') + ';'
                    + (slot.hasTimeAdvantage ? '1' : '0');
            }
            slot.records.push_back(std::move(record));

            Stockfish::Move moveA = Stockfish::MOVE_NONE;
            Stockfish::Move moveB = Stockfish::MOVE_NONE;
            if (!sample_joint_action(
                    board, slot,
                    outputs.policyA + index * NB_POLICY_VALUES(),
                    outputs.policyB + index * NB_POLICY_VALUES(),
                    rng, moveA, moveB)) {
                finish_game(slot, team_index(~slot.team), writer, localWritten);
                ++games;
                reset_slot(slot, rng, config.randomMoveProbability);
                continue;
            }
            if (moveA != Stockfish::MOVE_NONE) {
                board.push_move(BOARD_A, moveA);
            }
            if (moveB != Stockfish::MOVE_NONE) {
                board.push_move(BOARD_B, moveB);
            }
            slot.team = ~slot.team;
            slot.hasTimeAdvantage = !slot.hasTimeAdvantage;
            ++slot.macroPly;
        }

        const uint64_t now = localWritten.load();
        if (now / 100000 != workerWritten / 100000) {
            std::lock_guard<std::mutex> lock(logMutex);
            std::cout << "worker " << workerIndex << " positions " << now
                      << " / " << quota << std::endl;
        }
        workerWritten = now;
    }
    written += localWritten.load();
    writer.flush();
}

}  // namespace nnue_datagen

int run_nnue_data(Engine& engine, const NnueDataConfig& config) {
    using namespace nnue_datagen;
    if (config.threads == 0
        || config.threads > static_cast<size_t>(SearchParams::NUM_SEARCH_THREADS)) {
        throw std::invalid_argument(
            "--threads must be between 1 and "
            + std::to_string(SearchParams::NUM_SEARCH_THREADS));
    }
    std::filesystem::create_directories(config.outputDirectory);
    const uint64_t runSeed = config.seed != 0
        ? config.seed
        : static_cast<uint64_t>(
            std::chrono::system_clock::now().time_since_epoch().count());
    std::atomic<uint64_t> written{0};
    std::atomic<uint64_t> games{0};
    std::mutex logMutex;
    const auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(config.threads);
    for (size_t worker = 0; worker < config.threads; ++worker) {
        workers.emplace_back([&, worker] {
            try {
                generation_worker(engine, config, worker, runSeed, written, games, logMutex);
            } catch (...) {
                errors[worker] = std::current_exception();
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (const std::exception_ptr& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "wrote " << written.load() << " positions from " << games.load()
              << " games in " << seconds << " s ("
              << static_cast<double>(written.load()) / seconds << " pos/s)" << std::endl;
    return 0;
}

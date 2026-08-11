#include "tournament.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "agent.h"
#include "board.h"
#include "constants.h"
#include "engine.h"

namespace {

uint64_t tournament_seed(uint64_t seed, uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return seed ^ (value ^ (value >> 31));
}

JointActionCandidate most_visited_action(const std::vector<RootEdgeStats>& edges) {
    if (edges.empty()) {
        throw std::runtime_error("Tournament search returned no root edges");
    }
    return std::max_element(
        edges.begin(), edges.end(),
        [](const RootEdgeStats& left, const RootEdgeStats& right) {
            return left.visits < right.visits;
        })->action;
}

std::string action_uci(Board& board, const JointActionCandidate& action) {
    const std::string moveA = action.moveA == Stockfish::MOVE_NONE
        ? "pass"
        : board.uci_move(BOARD_A, action.moveA);
    const std::string moveB = action.moveB == Stockfish::MOVE_NONE
        ? "pass"
        : board.uci_move(BOARD_B, action.moveB);
    return "(" + moveA + "," + moveB + ")";
}

std::string pgn_result(int winner) {
    if (winner == static_cast<int>(Stockfish::WHITE)) {
        return "1-0";
    }
    if (winner == static_cast<int>(Stockfish::BLACK)) {
        return "0-1";
    }
    return "1/2-1/2";
}

void record_outcome(TournamentBreakdown& breakdown, int contenderOutcome) {
    if (contenderOutcome > 0) {
        breakdown.wins++;
    } else if (contenderOutcome < 0) {
        breakdown.losses++;
    } else {
        breakdown.draws++;
    }
}

void write_breakdown_json(
    std::ostream& stream,
    const std::string& name,
    const TournamentBreakdown& breakdown,
    bool trailingComma) {
    stream << "    \"" << name << "\": {\"wins\": " << breakdown.wins
           << ", \"losses\": " << breakdown.losses
           << ", \"draws\": " << breakdown.draws << "}"
           << (trailingComma ? "," : "") << '\n';
}

void print_breakdown(
    const std::string& label,
    const TournamentBreakdown& breakdown) {
    std::cout << "  " << std::left << std::setw(12) << label << std::right
              << breakdown.wins << '-' << breakdown.losses << '-'
              << breakdown.draws << " (" << breakdown.games() << " games)\n";
}

void append_game_pgn(
    const std::filesystem::path& path,
    size_t round,
    const std::string& whiteTeam,
    const std::string& blackTeam,
    int winner,
    const std::string& termination,
    const std::vector<std::string>& actions) {
    std::ofstream stream(path, std::ios::app);
    if (!stream) {
        throw std::runtime_error("Unable to append tournament PGN: " + path.string());
    }
    const std::string result = pgn_result(winner);
    stream << "[Event \"Hivemind Network Tournament\"]\n"
           << "[Site \"Hivemind Engine\"]\n"
           << "[Round \"" << round << "\"]\n"
           << "[Variant \"bughouse\"]\n"
           << "[WhiteTeam \"" << whiteTeam << "\"]\n"
           << "[BlackTeam \"" << blackTeam << "\"]\n"
           << "[Result \"" << result << "\"]\n"
           << "[Termination \"" << termination << "\"]\n\n";
    for (size_t index = 0; index < actions.size(); ++index) {
        stream << (index + 1) << ". " << actions[index] << ' ';
    }
    stream << result << "\n\n";
}

void write_summary(
    const std::filesystem::path& path,
    const std::string& contenderName,
    const std::string& baselineName,
    const TournamentConfig& config,
    const TournamentResult& result) {
    const std::filesystem::path temporaryPath = path.string() + ".tmp";
    std::ofstream stream(temporaryPath, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Unable to write tournament summary: " + path.string());
    }
    stream << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"contender\": \"" << contenderName << "\",\n"
           << "  \"baseline\": \"" << baselineName << "\",\n"
           << "  \"games\": " << result.games() << ",\n"
           << "  \"nodes_per_move\": " << config.nodes << ",\n"
           << "  \"move_time_ms\": " << config.moveTimeMs << ",\n"
           << "  \"contender_batch_size\": " << config.contenderBatchSize << ",\n"
           << "  \"baseline_batch_size\": " << config.baselineBatchSize << ",\n"
           << "  \"seed\": " << config.seed << ",\n"
           << "  \"contender_pw_coefficient\": "
           << config.contenderPwCoefficient << ",\n"
           << "  \"baseline_pw_coefficient\": "
           << config.baselinePwCoefficient << ",\n"
           << "  \"contender_wins\": " << result.contenderWins << ",\n"
           << "  \"baseline_wins\": " << result.baselineWins << ",\n"
           << "  \"draws\": " << result.draws << ",\n"
           << "  \"contender_score\": " << result.contenderScore() << ",\n"
           << "  \"contender_elo\": ";
    if (const auto elo = result.contenderElo()) {
        stream << *elo;
    } else {
        stream << "null";
    }
        stream << ",\n  \"confidence_method\": \"" << result.confidenceMethod() << "\",\n"
            << "  \"score_confidence_95\": ";
    if (const auto interval = result.scoreConfidence95()) {
        stream << '[' << interval->first << ", " << interval->second << ']';
    } else {
        stream << "null";
    }
    stream << ",\n  \"elo_confidence_95\": ";
    if (const auto interval = result.eloConfidence95()) {
        stream << '[' << interval->first << ", " << interval->second << ']';
    } else {
        stream << "null";
    }
    stream << ",\n  \"contender_breakdown\": {\n";
    write_breakdown_json(stream, "white", result.asWhite, true);
    write_breakdown_json(stream, "black", result.asBlack, true);
    write_breakdown_json(stream, "up_time", result.upTime, true);
    write_breakdown_json(stream, "down_time", result.downTime, false);
    stream << "  },\n"
           << "  \"terminations\": {\n"
           << "    \"checkmate\": " << result.checkmates << ",\n"
           << "    \"no_legal_action\": " << result.noLegalActions << ",\n"
           << "    \"draw\": " << result.drawnTerminations << ",\n"
           << "    \"macro_ply_limit\": " << result.macroPlyLimits << "\n"
           << "  }\n"
           << "}\n";
    stream.close();
    if (!stream) {
        throw std::runtime_error("Failed to finalize tournament summary: " + path.string());
    }
    std::filesystem::rename(temporaryPath, path);
}

void print_final_summary(
    const std::string& contenderName,
    const std::string& baselineName,
    const TournamentConfig& config,
    const TournamentResult& result) {
    std::cout << std::fixed << std::setprecision(2)
              << "\n============================================================\n"
              << "TOURNAMENT RESULTS\n"
              << "============================================================\n"
              << "Contender: " << contenderName << '\n'
              << "Baseline:  " << baselineName << '\n'
              << "Games: " << result.games() << " (" << result.games() / 2
              << " paired openings), ";
    if (config.moveTimeMs > 0) {
        std::cout << "movetime: " << config.moveTimeMs << " ms";
    } else {
        std::cout << "nodes/move: " << config.nodes;
    }
    std::cout << ", batches: " << config.contenderBatchSize
              << " vs " << config.baselineBatchSize << '\n'
              << "Result: " << result.contenderWins << '-' << result.baselineWins
              << '-' << result.draws << " (W-L-D)\n"
              << "Score: " << 100.0 * result.contenderScore() << "%\n";
    if (const auto elo = result.contenderElo()) {
        std::cout << "Elo difference: " << std::showpos << *elo << std::noshowpos;
        if (const auto interval = result.eloConfidence95()) {
            std::cout << " (95% CI " << std::showpos << interval->first
                      << " to " << interval->second << std::noshowpos << ')';
        }
        std::cout << '\n';
    } else {
        std::cout << "Elo difference: undefined (score is 0% or 100%)\n";
    }
    if (const auto interval = result.scoreConfidence95()) {
        std::cout << "Score 95% CI: " << 100.0 * interval->first << "% to "
                  << 100.0 * interval->second << "%\n";
    }
    std::cout << "\nContender breakdown (W-L-D):\n";
    print_breakdown("White", result.asWhite);
    print_breakdown("Black", result.asBlack);
    print_breakdown("Up time", result.upTime);
    print_breakdown("Down time", result.downTime);
    std::cout << "\nTerminations:\n"
              << "  Checkmate: " << result.checkmates << '\n'
              << "  No legal action: " << result.noLegalActions << '\n'
              << "  Draw: " << result.drawnTerminations << '\n'
              << "  Macro-ply limit: " << result.macroPlyLimits << '\n'
                  << "\nConfidence method: " << result.confidenceMethod() << ".\n"
              << "Reports: " << (config.outputDirectory / "summary.json")
              << " and " << (config.outputDirectory / "games.pgn") << '\n'
              << "============================================================\n";
}

} // namespace

size_t TournamentBreakdown::games() const {
    return wins + losses + draws;
}

size_t TournamentResult::games() const {
    return contenderWins + baselineWins + draws;
}

double TournamentResult::contenderScore() const {
    if (games() == 0) {
        return 0.0;
    }
    return (static_cast<double>(contenderWins) + 0.5 * static_cast<double>(draws))
        / static_cast<double>(games());
}

std::optional<double> TournamentResult::contenderElo() const {
    const double score = contenderScore();
    if (games() == 0 || score <= 0.0 || score >= 1.0) {
        return std::nullopt;
    }
    return 400.0 * std::log10(score / (1.0 - score));
}

std::optional<std::pair<double, double>> TournamentResult::scoreConfidence95() const {
    if (games() == 0) {
        return std::nullopt;
    }
    constexpr double z = 1.959963984540054;
    if (pairScores.size() >= 2) {
        const double count = static_cast<double>(pairScores.size());
        const double mean = std::accumulate(
            pairScores.begin(), pairScores.end(), 0.0) / count;
        const double squaredError = std::accumulate(
            pairScores.begin(), pairScores.end(), 0.0,
            [mean](double total, double score) {
                const double difference = score - mean;
                return total + difference * difference;
            });
        const double sampleVariance = squaredError / (count - 1.0);
        const double margin = z * std::sqrt(sampleVariance / count);
        return std::pair{
            std::max(0.0, mean - margin),
            std::min(1.0, mean + margin)};
    }
    const double count = static_cast<double>(games());
    const double score = contenderScore();
    const double denominator = 1.0 + z * z / count;
    const double center = (score + z * z / (2.0 * count)) / denominator;
    const double margin = z * std::sqrt(
        score * (1.0 - score) / count + z * z / (4.0 * count * count))
        / denominator;
    return std::pair{
        std::max(0.0, center - margin),
        std::min(1.0, center + margin)};
}

std::optional<std::pair<double, double>> TournamentResult::eloConfidence95() const {
    const auto scoreInterval = scoreConfidence95();
    if (!scoreInterval
        || scoreInterval->first <= 0.0
        || scoreInterval->second >= 1.0) {
        return std::nullopt;
    }
    auto score_to_elo = [](double score) {
        return 400.0 * std::log10(score / (1.0 - score));
    };
    return std::pair{
        score_to_elo(scoreInterval->first),
        score_to_elo(scoreInterval->second)};
}

std::string TournamentResult::confidenceMethod() const {
    return pairScores.size() >= 2
        ? "paired-opening normal approximation"
        : "game-level Wilson approximation";
}

int run_tournament(
    Engine& contender,
    Engine& baseline,
    const std::string& contenderName,
    const std::string& baselineName,
    const TournamentConfig& config) {
    if (config.games == 0 || config.games % 2 != 0) {
        throw std::invalid_argument("Tournament games must be a positive even number");
    }
    if ((config.nodes == 0) == (config.moveTimeMs <= 0) ||
        config.maxMacroPlies == 0) {
        throw std::invalid_argument(
            "Tournament requires exactly one positive nodes or movetime limit");
    }
    if (config.contenderBatchSize <= 0 || config.baselineBatchSize <= 0) {
        throw std::invalid_argument("Tournament batch sizes must be positive");
    }
    if (config.dirichletAlpha < 0.0f
        || config.dirichletEpsilon < 0.0f
        || config.dirichletEpsilon > 1.0f) {
        throw std::invalid_argument("Invalid tournament Dirichlet configuration");
    }
    if (!std::isfinite(config.contenderPwCoefficient)
        || !std::isfinite(config.baselinePwCoefficient)
        || config.contenderPwCoefficient <= 0.0f
        || config.baselinePwCoefficient <= 0.0f) {
        throw std::invalid_argument("Tournament PW coefficients must be positive and finite");
    }

    std::filesystem::create_directories(config.outputDirectory);
    const std::filesystem::path pgnPath = config.outputDirectory / "games.pgn";
    const std::filesystem::path summaryPath = config.outputDirectory / "summary.json";
    std::filesystem::remove(pgnPath);
    TournamentResult result;
    double currentPairPoints = 0.0;

    for (size_t gameIndex = 0; gameIndex < config.games; ++gameIndex) {
        Board board;
        Agent agent;
        const size_t pairIndex = gameIndex / 2;
        const Stockfish::Color contenderTeam = gameIndex % 2 == 0
            ? Stockfish::WHITE
            : Stockfish::BLACK;
        Stockfish::Color team = pairIndex % 2 == 0
            ? Stockfish::WHITE
            : Stockfish::BLACK;
        const bool contenderHasTimeAdvantage = contenderTeam != team;
        bool hasTimeAdvantage = false;
        int winner = -1;
        std::string termination = "macro-ply limit";
        std::vector<std::string> actions;

        for (size_t macroPly = 0; macroPly < config.maxMacroPlies; ++macroPly) {
            if (board.is_checkmate(team, hasTimeAdvantage)) {
                winner = static_cast<int>(~team);
                termination = "checkmate";
                break;
            }
            if (board.is_draw()) {
                termination = "draw";
                break;
            }

            const bool contenderActing = team == contenderTeam;
            Engine& actingEngine = contenderActing ? contender : baseline;
            std::vector<Engine*> engines = {&actingEngine};
            agent.reset_search_state();
            SearchOptions options;
            options.targetNodes = config.nodes;
            options.moveTimeMs = config.moveTimeMs;
            options.search = config.searchConfigFor(contenderActing);
            options.search.rootDirichletAlpha = config.dirichletAlpha;
            options.search.rootDirichletEpsilon = config.dirichletEpsilon;
            options.search.rootNoiseSeed = tournament_seed(
                config.seed,
                pairIndex * config.maxMacroPlies + macroPly);
            agent.run_search(board, engines, team, hasTimeAdvantage, options);
            const std::vector<RootEdgeStats> edges = agent.root_edge_stats();
            if (edges.empty()) {
                winner = static_cast<int>(~team);
                termination = "no legal action";
                break;
            }
            const JointActionCandidate action = most_visited_action(edges);
            actions.push_back(action_uci(board, action));
            board.make_moves(action.moveA, action.moveB);
            team = ~team;
            hasTimeAdvantage = !hasTimeAdvantage;
        }

        if (winner < 0 && board.is_checkmate(team, hasTimeAdvantage)) {
            winner = static_cast<int>(~team);
            termination = "checkmate";
        } else if (winner < 0 && board.is_draw()) {
            termination = "draw";
        }

        int contenderOutcome = 0;
        if (winner < 0) {
            result.draws++;
        } else if (winner == static_cast<int>(contenderTeam)) {
            result.contenderWins++;
            contenderOutcome = 1;
        } else {
            result.baselineWins++;
            contenderOutcome = -1;
        }
        record_outcome(
            contenderTeam == Stockfish::WHITE ? result.asWhite : result.asBlack,
            contenderOutcome);
        record_outcome(
            contenderHasTimeAdvantage ? result.upTime : result.downTime,
            contenderOutcome);
        currentPairPoints += contenderOutcome > 0 ? 1.0 : contenderOutcome == 0 ? 0.5 : 0.0;
        if (gameIndex % 2 == 1) {
            result.pairScores.push_back(currentPairPoints / 2.0);
            currentPairPoints = 0.0;
        }
        if (termination == "checkmate") {
            result.checkmates++;
        } else if (termination == "no legal action") {
            result.noLegalActions++;
        } else if (termination == "draw") {
            result.drawnTerminations++;
        } else {
            result.macroPlyLimits++;
        }

        const std::string whiteTeam = contenderTeam == Stockfish::WHITE
            ? contenderName
            : baselineName;
        const std::string blackTeam = contenderTeam == Stockfish::BLACK
            ? contenderName
            : baselineName;
        append_game_pgn(
            pgnPath, gameIndex + 1, whiteTeam, blackTeam,
            winner, termination, actions);
        write_summary(
            summaryPath, contenderName, baselineName, config, result);
        std::cout << "tournament game " << (gameIndex + 1) << '/' << config.games
                  << " contender " << result.contenderWins
                  << " baseline " << result.baselineWins
                  << " draws " << result.draws
                  << " termination " << termination << '\n';
    }

    print_final_summary(contenderName, baselineName, config, result);

    return 0;
}
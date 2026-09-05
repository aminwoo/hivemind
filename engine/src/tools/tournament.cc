#include "tools/tournament.h"

#include <algorithm>
#include <chrono>
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

#include "search/agent.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "nn/engine.h"

namespace {

std::string trim_copy(std::string value) {
    const size_t first = value.find_first_not_of(" \r\n\t");
    if (first == std::string::npos) return {};
    const size_t last = value.find_last_not_of(" \r\n\t");
    return value.substr(first, last - first + 1);
}

const char* sprt_decision_name(SprtState::Decision decision) {
    switch (decision) {
        case SprtState::Decision::ACCEPT_H0: return "accept_h0";
        case SprtState::Decision::ACCEPT_H1: return "accept_h1";
        default: return "continue";
    }
}

uint64_t tournament_seed(uint64_t seed, uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return seed ^ (value ^ (value >> 31));
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
    const std::string& initialDualFen,
    Stockfish::Color initialTeam,
    bool initialTimeAdvantage,
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
           << "[DualFEN \"" << initialDualFen << "\"]\n"
           << "[TeamToPlay \""
           << (initialTeam == Stockfish::WHITE ? "white" : "black") << "\"]\n"
           << "[TeamTimeAdvantage \""
           << (initialTimeAdvantage ? "true" : "false") << "\"]\n"
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
           << "  \"contender_model_signature\": \""
           << config.contenderModelSignature << "\",\n"
           << "  \"baseline_model_signature\": \""
           << config.baselineModelSignature << "\",\n"
           << "  \"games\": " << result.games() << ",\n"
           << "  \"nodes_per_move\": " << config.nodes << ",\n"
           << "  \"move_time_ms\": " << config.moveTimeMs << ",\n"
           << "  \"contender_batch_size\": " << config.contenderBatchSize << ",\n"
           << "  \"baseline_batch_size\": " << config.baselineBatchSize << ",\n"
           << "  \"contender_threads\": " << config.contenderThreads << ",\n"
           << "  \"baseline_threads\": " << config.baselineThreads << ",\n"
           << "  \"seed\": " << config.seed << ",\n"
           << "  \"contender_supply_policy_weight\": " << config.contenderSupplyPolicyWeight << ",\n"
           << "  \"baseline_supply_policy_weight\": " << config.baselineSupplyPolicyWeight << ",\n"
           << "  \"contender_supply_value_weight\": " << config.contenderSupplyValueWeight << ",\n"
           << "  \"baseline_supply_value_weight\": " << config.baselineSupplyValueWeight << ",\n"
           << "  \"contender_pw_coefficient\": "
           << config.contenderPwCoefficient << ",\n"
           << "  \"baseline_pw_coefficient\": "
           << config.baselinePwCoefficient << ",\n"
           << "  \"contender_root_pw_coefficient\": "
           << config.contenderRootPwCoefficient << ",\n"
           << "  \"baseline_root_pw_coefficient\": "
           << config.baselineRootPwCoefficient << ",\n"
           << "  \"contender_mcgs\": " << (config.contenderMcgs ? "true" : "false") << ",\n"
           << "  \"baseline_mcgs\": " << (config.baselineMcgs ? "true" : "false") << ",\n"
           << "  \"contender_transpositions\": " << (config.contenderTranspositions ? "true" : "false") << ",\n"
           << "  \"baseline_transpositions\": " << (config.baselineTranspositions ? "true" : "false") << ",\n"
           << "  \"contender_root_mate_search\": " << (config.contenderRootMateSearch ? "true" : "false") << ",\n"
           << "  \"baseline_root_mate_search\": " << (config.baselineRootMateSearch ? "true" : "false") << ",\n"
           << "  \"contender_wdl_weight\": " << config.contenderWdlWeight << ",\n"
           << "  \"baseline_wdl_weight\": " << config.baselineWdlWeight << ",\n"
           << "  \"contender_moves_left_discount\": " << config.contenderMovesLeftDiscount << ",\n"
           << "  \"baseline_moves_left_discount\": " << config.baselineMovesLeftDiscount << ",\n"
           << "  \"contender_q_value_weight\": " << config.contenderQValueWeight << ",\n"
           << "  \"baseline_q_value_weight\": " << config.baselineQValueWeight << ",\n"
           << "  \"contender_q_veto_delta\": " << config.contenderQVetoDelta << ",\n"
           << "  \"baseline_q_veto_delta\": " << config.baselineQVetoDelta << ",\n"
           << "  \"positions_file\": \"" << config.positionsFile.string() << "\",\n"
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
    stream << ",\n  \"sprt\": {\"enabled\": "
           << (config.sprtEnabled() ? "true" : "false")
           << ", \"elo0\": " << config.sprtElo0
           << ", \"elo1\": " << config.sprtElo1
           << ", \"alpha\": " << config.sprtAlpha
           << ", \"beta\": " << config.sprtBeta
           << ", \"llr\": " << result.sprt.logLikelihoodRatio
           << ", \"lower\": " << result.sprt.lowerBoundary
           << ", \"upper\": " << result.sprt.upperBoundary
           << ", \"decision\": \""
           << sprt_decision_name(result.sprt.decision) << "\"},\n"
           << "  \"performance\": {\n"
           << "    \"contender\": {\"searches\": " << result.contenderPerformance.searches
           << ", \"nodes\": " << result.contenderPerformance.nodes
           << ", \"nps\": " << result.contenderPerformance.nps() << "},\n"
           << "    \"baseline\": {\"searches\": " << result.baselinePerformance.searches
           << ", \"nodes\": " << result.baselinePerformance.nodes
           << ", \"nps\": " << result.baselinePerformance.nps() << "}\n"
           << "  },\n"
           << "  \"contender_breakdown\": {\n";
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
              << "\nFull-search throughput:\n"
              << "  Contender: " << result.contenderPerformance.nps() << " NPS\n"
              << "  Baseline: " << result.baselinePerformance.nps() << " NPS\n";
    if (config.sprtEnabled()) {
        std::cout << "\nSPRT: " << sprt_decision_name(result.sprt.decision)
                  << ", LLR " << result.sprt.logLikelihoodRatio
                  << " [" << result.sprt.lowerBoundary << ", "
                  << result.sprt.upperBoundary << "]\n";
    }
    std::cout
                  << "\nConfidence method: " << result.confidenceMethod() << ".\n"
              << "Reports: " << (config.outputDirectory / "summary.json")
              << " and " << (config.outputDirectory / "games.pgn") << '\n'
              << "============================================================\n";
}

} // namespace

std::vector<TournamentStartPosition> load_tournament_positions(
    const std::filesystem::path& path) {
    if (path.empty()) {
        return {};
    }
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(
            "Unable to open tournament positions: " + path.string());
    }

    std::vector<TournamentStartPosition> positions;
    std::string line;
    size_t lineNumber = 0;
    while (std::getline(stream, line)) {
        ++lineNumber;
        line = trim_copy(line);
        if (line.empty() || line.front() == '#') continue;

        std::vector<std::string> fields;
        std::stringstream parser(line);
        std::string field;
        while (std::getline(parser, field, '\t')) {
            fields.push_back(trim_copy(field));
        }
        if (fields.empty() || std::count(fields[0].begin(), fields[0].end(), '|') != 1) {
            throw std::runtime_error(
                "Invalid dual FEN at " + path.string() + ":"
                + std::to_string(lineNumber));
        }
        TournamentStartPosition position;
        position.dualFen = fields[0];
        if (fields.size() >= 2) {
            if (fields[1] == "white" || fields[1] == "w") {
                position.teamToPlay = Stockfish::WHITE;
            } else if (fields[1] == "black" || fields[1] == "b") {
                position.teamToPlay = Stockfish::BLACK;
            } else {
                throw std::runtime_error(
                    "Invalid team at " + path.string() + ":"
                    + std::to_string(lineNumber));
            }
        }
        if (fields.size() >= 3) {
            if (fields[2] == "1" || fields[2] == "true") {
                position.teamHasTimeAdvantage = true;
            } else if (fields[2] == "0" || fields[2] == "false") {
                position.teamHasTimeAdvantage = false;
            } else {
                throw std::runtime_error(
                    "Invalid time-advantage flag at " + path.string() + ":"
                    + std::to_string(lineNumber));
            }
        }
        if (fields.size() > 3) {
            throw std::runtime_error(
                "Too many fields at " + path.string() + ":"
                + std::to_string(lineNumber));
        }
        positions.push_back(std::move(position));
    }
    if (positions.empty()) {
        throw std::runtime_error("Tournament positions file is empty");
    }
    return positions;
}

SprtState evaluate_paired_sprt(const std::vector<double>& pairScores,
                               double elo0, double elo1,
                               double alpha, double beta) {
    if (!(elo1 > elo0) || !(alpha > 0.0 && alpha < 1.0)
        || !(beta > 0.0 && beta < 1.0)) {
        throw std::invalid_argument("Invalid paired SPRT parameters");
    }
    const auto expected_score = [](double elo) {
        return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0));
    };
    const double p0 = expected_score(elo0);
    const double p1 = expected_score(elo1);
    SprtState state;
    state.lowerBoundary = std::log(beta / (1.0 - alpha));
    state.upperBoundary = std::log((1.0 - beta) / alpha);
    for (double pairScore : pairScores) {
        if (!std::isfinite(pairScore) || pairScore < 0.0 || pairScore > 1.0) {
            throw std::invalid_argument("Invalid paired SPRT score");
        }
        const double points = 2.0 * pairScore;
        state.logLikelihoodRatio +=
            points * std::log(p1 / p0)
            + (2.0 - points) * std::log((1.0 - p1) / (1.0 - p0));
    }
    if (state.logLikelihoodRatio <= state.lowerBoundary) {
        state.decision = SprtState::Decision::ACCEPT_H0;
    } else if (state.logLikelihoodRatio >= state.upperBoundary) {
        state.decision = SprtState::Decision::ACCEPT_H1;
    }
    return state;
}

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
    if (config.contenderThreads <= 0 || config.baselineThreads <= 0
        || config.contenderThreads > SearchParams::NUM_SEARCH_THREADS
        || config.baselineThreads > SearchParams::NUM_SEARCH_THREADS) {
        throw std::invalid_argument(
            "Tournament threads must be between 1 and the compiled worker count");
    }
    if (config.dirichletAlpha < 0.0f
        || config.dirichletEpsilon < 0.0f
        || config.dirichletEpsilon > 1.0f) {
        throw std::invalid_argument("Invalid tournament Dirichlet configuration");
    }
    if (!std::isfinite(config.contenderPwCoefficient)
        || !std::isfinite(config.baselinePwCoefficient)
        || !std::isfinite(config.contenderRootPwCoefficient)
        || !std::isfinite(config.baselineRootPwCoefficient)
        || config.contenderPwCoefficient <= 0.0f
        || config.baselinePwCoefficient <= 0.0f
        || config.contenderRootPwCoefficient <= 0.0f
        || config.baselineRootPwCoefficient <= 0.0f) {
        throw std::invalid_argument("Tournament PW coefficients must be positive and finite");
    }
    const auto finite_in_range = [](float value, float minimum, float maximum) {
        return std::isfinite(value) && value >= minimum && value <= maximum;
    };
    if (!finite_in_range(config.contenderSupplyPolicyWeight, 0.0f, 0.5f)
        || !finite_in_range(config.baselineSupplyPolicyWeight, 0.0f, 0.5f)
        || !finite_in_range(config.contenderSupplyValueWeight, 0.0f, 0.5f)
        || !finite_in_range(config.baselineSupplyValueWeight, 0.0f, 0.5f)
        || !finite_in_range(config.contenderWdlWeight, 0.0f, 1.0f)
        || !finite_in_range(config.baselineWdlWeight, 0.0f, 1.0f)
        || !finite_in_range(config.contenderMovesLeftDiscount, 0.0f, 1.0f)
        || !finite_in_range(config.baselineMovesLeftDiscount, 0.0f, 1.0f)
        || !finite_in_range(config.contenderQValueWeight, 0.0f, 100.0f)
        || !finite_in_range(config.baselineQValueWeight, 0.0f, 100.0f)
        || !finite_in_range(config.contenderQVetoDelta, 0.0f, 2.0f)
        || !finite_in_range(config.baselineQVetoDelta, 0.0f, 2.0f)) {
        throw std::invalid_argument("Invalid tournament supply/WDL/moves-left/Q parameter");
    }
    if (config.sprtEnabled()
        && (!(config.sprtAlpha > 0.0 && config.sprtAlpha < 1.0)
            || !(config.sprtBeta > 0.0 && config.sprtBeta < 1.0))) {
        throw std::invalid_argument("Tournament SPRT alpha and beta must be in (0, 1)");
    }

    std::filesystem::create_directories(config.outputDirectory);
    const std::filesystem::path pgnPath = config.outputDirectory / "games.pgn";
    const std::filesystem::path summaryPath = config.outputDirectory / "summary.json";
    std::filesystem::remove(pgnPath);
    TournamentResult result;
    double currentPairPoints = 0.0;
    const std::vector<TournamentStartPosition> startPositions =
        load_tournament_positions(config.positionsFile);
    Agent contenderAgent(config.contenderThreads);
    Agent baselineAgent(config.baselineThreads);

    for (size_t gameIndex = 0; gameIndex < config.games; ++gameIndex) {
        Board board;
        const size_t pairIndex = gameIndex / 2;
        if (!startPositions.empty()) {
            board.set(startPositions[pairIndex % startPositions.size()].dualFen);
        }
        const Stockfish::Color contenderTeam = gameIndex % 2 == 0
            ? Stockfish::WHITE
            : Stockfish::BLACK;
        Stockfish::Color team = startPositions.empty()
            ? (pairIndex % 2 == 0 ? Stockfish::WHITE : Stockfish::BLACK)
            : startPositions[pairIndex % startPositions.size()].teamToPlay;
        bool hasTimeAdvantage = startPositions.empty()
            ? false
            : startPositions[pairIndex % startPositions.size()].teamHasTimeAdvantage;
        const Stockfish::Color initialTeam = team;
        const bool initialTimeAdvantage = hasTimeAdvantage;
        const std::string initialDualFen =
            board.fen(BOARD_A) + "|" + board.fen(BOARD_B);
        const bool contenderHasTimeAdvantage = contenderTeam == team
            ? hasTimeAdvantage : !hasTimeAdvantage;
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
            Agent& agent = contenderActing ? contenderAgent : baselineAgent;
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
            const auto searchStart = std::chrono::steady_clock::now();
            const JointActionCandidate action = agent.run_search(
                board, engines, team, hasTimeAdvantage, options);
            const uint64_t searchNanos = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - searchStart).count());
            const std::vector<RootEdgeStats> edges = agent.root_edge_stats();
            TournamentPerformance& performance = contenderActing
                ? result.contenderPerformance : result.baselinePerformance;
            ++performance.searches;
            performance.nanoseconds += searchNanos;
            performance.nodes += std::accumulate(
                edges.begin(), edges.end(), uint64_t{0},
                [](uint64_t sum, const RootEdgeStats& edge) {
                    return sum + edge.visits;
                });
            if (edges.empty()) {
                winner = static_cast<int>(~team);
                termination = "no legal action";
                break;
            }
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
            if (config.sprtEnabled()) {
                result.sprt = evaluate_paired_sprt(
                    result.pairScores, config.sprtElo0, config.sprtElo1,
                    config.sprtAlpha, config.sprtBeta);
            }
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
            winner, termination, initialDualFen, initialTeam,
            initialTimeAdvantage, actions);
        write_summary(
            summaryPath, contenderName, baselineName, config, result);
        std::cout << "tournament game " << (gameIndex + 1) << '/' << config.games
                  << " contender " << result.contenderWins
                  << " baseline " << result.baselineWins
                  << " draws " << result.draws
                  << " termination " << termination << '\n';
        if (gameIndex % 2 == 1
            && result.sprt.decision != SprtState::Decision::CONTINUE) {
            std::cout << "SPRT stopped after " << result.pairScores.size()
                      << " pairs: "
                      << sprt_decision_name(result.sprt.decision)
                      << " (LLR " << result.sprt.logLikelihoodRatio << ")\n";
            break;
        }
    }

    print_final_summary(contenderName, baselineName, config, result);

    return 0;
}

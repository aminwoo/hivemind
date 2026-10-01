#include "tools/search_diag.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "environment/board.h"
#include "environment/constants.h"
#include "nn/engine.h"
#include "search/agent.h"

namespace {

/// m: ordinary move, c: capture, d: drop, s: sit (pass with a legal move
/// available), w: forced wait (board not on turn), p: pass with no legal move.
char move_kind(Board& board, int boardNumber, bool onTurn, Stockfish::Move move) {
    if (!onTurn) {
        return 'w';
    }
    if (move == Stockfish::MOVE_NONE) {
        return board.legal_moves(boardNumber).empty() ? 'p' : 's';
    }
    if (Stockfish::type_of(move) == Stockfish::DROP) {
        return 'd';
    }
    return board.is_capture(boardNumber, move) ? 'c' : 'm';
}

std::string move_text(Board& board, int boardNumber, Stockfish::Move move) {
    return move == Stockfish::MOVE_NONE ? "pass" : board.uci_move(boardNumber, move);
}

}  // namespace

int run_search_diag(Engine& engine, const SearchDiagConfig& config) {
    std::ifstream fens(config.fenFile);
    if (!fens) {
        throw std::runtime_error("Cannot open " + config.fenFile.string());
    }
    std::ofstream out(config.outputFile, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Cannot create " + config.outputFile.string());
    }
    const std::vector<Engine*> engines = {&engine};
    Agent agent;
    std::string line;
    size_t lineIndex = 0;
    size_t written = 0;
    while (written < config.positions && std::getline(fens, line)) {
        if (lineIndex++ % std::max<size_t>(1, config.every) != 0) {
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream stream(line);
        for (std::string field; std::getline(stream, field, ';');) {
            fields.push_back(field);
        }
        if (fields.size() != 4) {
            continue;
        }
        Board board;
        board.set(fields[0] + "|" + fields[1]);
        const Stockfish::Color team = fields[2] == "w" ? Stockfish::WHITE : Stockfish::BLACK;
        const bool advantage = fields[3] == "1";
        if (board.is_checkmate(team, advantage) || board.is_checkmate(~team, !advantage)
            || board.is_draw()) {
            continue;
        }
        const bool onTurnA = board.side_to_move(BOARD_A) == team;
        const bool onTurnB = board.side_to_move(BOARD_B) == ~team;

        std::ostringstream record;
        record << std::setprecision(5) << "{\"line\":" << (lineIndex - 1) << ",\"on_turn\":["
               << onTurnA << ',' << onTurnB << "],\"results\":[";
        bool empty = false;
        for (size_t budgetIndex = 0; budgetIndex < config.budgets.size(); ++budgetIndex) {
            agent.reset_search_state();
            SearchOptions options;
            options.targetNodes = config.budgets[budgetIndex];
            options.search.enableMateProbe = false;
            options.mateProbeNodes = 0;
            options.search.enableGumbelRootSearch = false;
            agent.run_search(board, engines, team, advantage, options);
            const std::vector<RootEdgeStats> edges = agent.root_edge_stats();
            if (edges.empty()) {
                empty = true;
                break;
            }
            record << (budgetIndex ? "," : "") << "{\"nodes\":" << config.budgets[budgetIndex]
                   << ",\"root_q\":" << agent.root_q() << ",\"edges\":[";
            for (size_t index = 0; index < edges.size(); ++index) {
                const JointActionCandidate& action = edges[index].action;
                record << (index ? "," : "") << "[\"" << move_text(board, BOARD_A, action.moveA)
                       << "\",\"" << move_text(board, BOARD_B, action.moveB) << "\","
                       << action.jointPrior << ',' << edges[index].visits << ','
                       << edges[index].q << ",\"" << move_kind(board, BOARD_A, onTurnA, action.moveA)
                       << move_kind(board, BOARD_B, onTurnB, action.moveB) << "\"]";
            }
            record << "]}";
        }
        if (empty) {
            continue;
        }
        record << "]}\n";
        out << record.str();
        if (++written % 500 == 0) {
            std::cout << "searchdiag " << written << '/' << config.positions << std::endl;
        }
    }
    std::cout << "searchdiag wrote " << written << " positions to " << config.outputFile << std::endl;
    return 0;
}

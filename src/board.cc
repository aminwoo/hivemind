#include "board.h"

const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string &s) {
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == std::string::npos) ? "" : s.substr(start);
}
 
std::string rtrim(const std::string &s) {
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}
 
std::string trim(const std::string &s) {
    return rtrim(ltrim(s));
}

void Board::set(std::string fen) {
    std::stringstream ss(fen);
    std::string line; 
    getline(ss, line, '|');
    line = trim(line); 

    if (line != pos[0]->fen()) {
        states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
        states[0]->emplace_back();
        pos[0]->set(Stockfish::variants.find("bughouse")->second, line, false, &states[0]->back(), Stockfish::Threads.main());
    }
    
    getline(ss, line, '|');
    line = trim(line);
    if (line != pos[1]->fen()) {
        states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
        states[1]->emplace_back();
        pos[1]->set(Stockfish::variants.find("bughouse")->second, line, false, &states[1]->back(), Stockfish::Threads.main());
    }
}

Board::Board() {
    pos[0] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);
    pos[1] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);

    states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    states[0]->emplace_back();
    pos[0]->set(Stockfish::variants.find("bughouse")->second, startingFen, false, &states[0]->back(), Stockfish::Threads.main());

    states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    states[1]->emplace_back();
    pos[1]->set(Stockfish::variants.find("bughouse")->second, startingFen, false, &states[1]->back(), Stockfish::Threads.main());
}

Board::Board(const Board& board) {
    pos[0] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);
    pos[1] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);

    states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    for (int i=1; i<(int)board.states[0]->size(); i++) {
        states[0]->emplace_back((*board.states[0])[i]);
    }

    states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    for (int i=1; i<(int)board.states[1]->size(); i++) {
        states[1]->emplace_back((*board.states[1])[i]);
    }

    pos[0]->set(Stockfish::variants.find("bughouse")->second, board.pos[0]->fen(), false, &states[0]->back(), Stockfish::Threads.main());
    pos[1]->set(Stockfish::variants.find("bughouse")->second, board.pos[1]->fen(), false, &states[1]->back(), Stockfish::Threads.main());

}

void Board::push_move(int board_num, Stockfish::Move move) {
    states[board_num]->emplace_back();
    pos[board_num]->do_move(move, states[board_num]->back());
    Stockfish::Piece p = states[board_num]->back().pieceToHand; 
    if (p) {
        pos[1 - board_num]->add_to_hand(p);
    }
}

void Board::pop_move(int board_num) {
    Stockfish::Move m = states[board_num]->back().move; 
    Stockfish::Piece p = states[board_num]->back().pieceToHand; 
    if (p) {
        pos[1 - board_num]->remove_from_hand(p);
    }
    pos[board_num]->undo_move(m); 
    states[board_num]->pop_back();
}

std::vector<Stockfish::Move> Board::legal_moves(int board_num) {
    std::vector<Stockfish::Move> legal_moves;
    for(const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[board_num])) {
        legal_moves.emplace_back(move);
    }
    return legal_moves;
}

std::vector<std::pair<int, Stockfish::Move>> Board::legal_moves(Stockfish::Color side) {
    std::vector<std::pair<int, Stockfish::Move>> moves;
    
    bool mated;
    if (pos[0]->side_to_move() == side) {
        mated = true; 
        for (const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[0])) {
            mated = false; 
            moves.emplace_back(0, move);
        }
        if (mated) {
            return {}; 
        }
    }

    if (pos[1]->side_to_move() == ~side) {
        mated = true; 
        for (const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[1])) {
            mated = false; 
            moves.emplace_back(1, move);
        }
        if (mated) {
            return {}; 
        }
    }
    return moves;
}

Board::~Board() {
    
}

#include "board.h"

// String containing whitespace characters.
const std::string WHITESPACE = " \n\r\t\f\v";

// Removes leading whitespace from the input string.
std::string ltrim(const std::string &s)
{
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == std::string::npos) ? "" : s.substr(start);
}
 
// Removes trailing whitespace from the input string.
std::string rtrim(const std::string &s)
{
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}
 
// Trims both leading and trailing whitespace.
std::string trim(const std::string &s)
{
    return rtrim(ltrim(s));
}

// Sets the board state using a FEN string.
// The FEN is expected to have two parts separated by a '|' character.
void Board::set(std::string fen)
{
    std::stringstream ss(fen);
    std::string line; 
    getline(ss, line, '|');
    line = trim(line); 

    // Update first board if its FEN differs.
    if (line != pos[0]->fen()) {
        states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
        states[0]->emplace_back();
        pos[0]->set(Stockfish::variants.find("bughouse")->second, line, false, &states[0]->back(), Stockfish::Threads.main());
    }
    
    getline(ss, line, '|');
    line = trim(line);
    
    // Update second board if its FEN differs.
    if (line != pos[1]->fen()) {
        states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
        states[1]->emplace_back();
        pos[1]->set(Stockfish::variants.find("bughouse")->second, line, false, &states[1]->back(), Stockfish::Threads.main());
    }
}

// Default constructor: initializes the board positions and sets them to the starting FEN.
Board::Board()
{
    pos[0] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);
    pos[1] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);

    states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    states[0]->emplace_back();
    pos[0]->set(Stockfish::variants.find("bughouse")->second, startingFen, false, &states[0]->back(), Stockfish::Threads.main());

    states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    states[1]->emplace_back();
    pos[1]->set(Stockfish::variants.find("bughouse")->second, startingFen, false, &states[1]->back(), Stockfish::Threads.main());
}

// Copy constructor: copies state history and reinitializes positions from the provided board.
Board::Board(const Board& board)
{
    pos[0] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);
    pos[1] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);

    states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    for (int i = 1; i < (int)board.states[0]->size(); i++) {
        states[0]->emplace_back((*board.states[0])[i]);
    }

    states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    for (int i = 1; i < (int)board.states[1]->size(); i++) {
        states[1]->emplace_back((*board.states[1])[i]);
    }

    pos[0]->set(Stockfish::variants.find("bughouse")->second, board.pos[0]->fen(), false, &states[0]->back(), Stockfish::Threads.main());
    pos[1]->set(Stockfish::variants.find("bughouse")->second, board.pos[1]->fen(), false, &states[1]->back(), Stockfish::Threads.main());
}

// Executes a move on the board and updates the corresponding state.
// Also adds a piece to the opponent's hand if necessary.
void Board::push_move(int board_num, Stockfish::Move move)
{
    states[board_num]->emplace_back();
    pos[board_num]->do_move(move, states[board_num]->back());
    Stockfish::Piece p = states[board_num]->back().pieceToHand; 
    if (p) {
        pos[1 - board_num]->add_to_hand(p);
    }
}

// Reverts the last move on the board and updates the state.
// Also removes a piece from the opponent's hand if necessary.
void Board::pop_move(int board_num)
{
    Stockfish::Move m = states[board_num]->back().move; 
    Stockfish::Piece p = states[board_num]->back().pieceToHand; 
    if (p) {
        pos[1 - board_num]->remove_from_hand(p);
    }
    pos[board_num]->undo_move(m); 
    states[board_num]->pop_back();
}

// Returns a list of legal moves for the specified board index.
std::vector<std::pair<int, Stockfish::Move>> Board::legal_moves(int board_num)
{
    std::vector<std::pair<int, Stockfish::Move>> legal_moves;
    for (const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[board_num])) {
        legal_moves.emplace_back(board_num, move);
    }
    return legal_moves;
}

// Returns a list of legal moves for the specified side by checking both boards.
std::vector<std::pair<int, Stockfish::Move>> Board::legal_moves(Stockfish::Color side)
{
    std::vector<std::pair<int, Stockfish::Move>> moves;

    // If checkmate, return an empty move list.
    if (is_checkmate(side)) {
        return {};
    }
    
    if (pos[0]->side_to_move() == side) {
        for (const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[0])) {
            moves.emplace_back(0, move);
        }
    }

    if (pos[1]->side_to_move() == ~side) {
        for (const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[1])) {
            moves.emplace_back(1, move);
        }
    }
    return moves;
}

// Determines if the board is in checkmate for the given side.
bool Board::is_checkmate(Stockfish::Color side)
{
    if (pos[0]->side_to_move() == side && pos[0]->checkers()) {
        Stockfish::MoveList<Stockfish::LEGAL> legalMoves(*pos[0]);
        if (!legalMoves.size())
            return true;
    }

    if (pos[1]->side_to_move() == ~side && pos[1]->checkers()) {
        Stockfish::MoveList<Stockfish::LEGAL> legalMoves(*pos[1]);
        if (!legalMoves.size())
            return true;
    }

    return false;
}

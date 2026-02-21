#include "board.h"

// String containing whitespace characters.
const std::string WHITESPACE = " \n\r\t\f\v";

// Removes leading whitespace from the input string.
std::string ltrim(const std::string &s) {
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == std::string::npos) ? "" : s.substr(start);
}
 
// Removes trailing whitespace from the input string.
std::string rtrim(const std::string &s) {
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}
 
// Trims both leading and trailing whitespace.
std::string trim(const std::string &s) {
    return rtrim(ltrim(s));
}

// Sets the board state using a FEN string.
// The FEN is expected to have two parts separated by a '|' character.
void Board::set(std::string fen) {
    std::stringstream ss(fen);
    std::string line; 
    getline(ss, line, '|');
    line = trim(line); 

    // Update first board if its FEN differs.
    if (line != pos[0]->fen()) {
        states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
        states[0]->emplace_back();
        pos[0]->set(Stockfish::variants.find("bughouse")->second, line, false, &states[0]->back(), Stockfish::Threads.main());
        // Reset position history for this board
        clear_position_history(0);
        record_position(0);
    }
    
    getline(ss, line, '|');
    line = trim(line);
    
    // Update second board if its FEN differs.
    if (line != pos[1]->fen()) {
        states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
        states[1]->emplace_back();
        pos[1]->set(Stockfish::variants.find("bughouse")->second, line, false, &states[1]->back(), Stockfish::Threads.main());
        // Reset position history for this board
        clear_position_history(1);
        record_position(1);
    }
}

// Default constructor: initializes the board positions and sets them to the starting FEN.
Board::Board() {
    pos[0] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);
    pos[1] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);

    states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    states[0]->emplace_back();
    pos[0]->set(Stockfish::variants.find("bughouse")->second, startingFen, false, &states[0]->back(), Stockfish::Threads.main());

    states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    states[1]->emplace_back();
    pos[1]->set(Stockfish::variants.find("bughouse")->second, startingFen, false, &states[1]->back(), Stockfish::Threads.main());
    
    // Initialize position history with starting positions
    record_position(0);
    record_position(1);
}

// Copy constructor: copies state history and reinitializes positions from the provided board.
Board::Board(const Board& board) {
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
    
    // Copy position history
    positionHistory[0] = board.positionHistory[0];
    positionHistory[1] = board.positionHistory[1];
}

// Executes a move on the board and updates the corresponding state.
// Also adds a piece to the opponent's hand if necessary.
void Board::push_move(int board_num, Stockfish::Move move) {
    states[board_num]->emplace_back();
    pos[board_num]->do_move(move, states[board_num]->back());
    Stockfish::Piece p = states[board_num]->back().pieceToHand; 
    if (p) {
        pos[1 - board_num]->add_to_hand(p);
    }
    // Record position for repetition detection
    record_position(board_num);
}

// Reverts the last move on the board and updates the state.
// Also removes a piece from the opponent's hand if necessary.
void Board::pop_move(int board_num) {
    Stockfish::Move m = states[board_num]->back().move; 
    Stockfish::Piece p = states[board_num]->back().pieceToHand; 
    if (p) {
        pos[1 - board_num]->remove_from_hand(p);
    }
    pos[board_num]->undo_move(m); 
    states[board_num]->pop_back();
    // Remove position from history
    unrecord_position(board_num);
}

// Returns a list of legal moves for the specified board index.
std::vector<Stockfish::Move> Board::legal_moves(int board_num) {
    std::vector<Stockfish::Move> legal_moves;
    for (const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[board_num])) {
        legal_moves.emplace_back(move);
    }
    return legal_moves;
}

// Returns a list of legal moves for the specified side by checking both boards.
std::vector<std::pair<int, Stockfish::Move>> Board::legal_moves(Stockfish::Color side, bool teamHasTimeAdvantage) {
    std::vector<std::pair<int, Stockfish::Move>> moves;

    // If checkmate, return an empty move list.
    if (is_checkmate(side, teamHasTimeAdvantage)) {
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

// Determines if the board is in checkmate for the given side in bughouse.
// In bughouse, checkmate is more complex because partner captures can provide
// pieces to drop and block a check. We must verify that:
// 1. The player has no legal moves (including drops with current pieces in hand)
// 2. The partner cannot capture any piece that could be used to block the check
bool Board::is_checkmate(Stockfish::Color side, bool teamHasTimeAdvantage) {
    // Check Board A (where 'side' plays)
    if (pos[BOARD_A]->side_to_move() == side && pos[BOARD_A]->checkers()) {
        Stockfish::MoveList<Stockfish::LEGAL> legalMoves(*pos[BOARD_A]);
        if (!legalMoves.size()) {
            // No legal moves - but can partner provide a blocking piece?
            if (!can_partner_provide_blocking_piece(BOARD_A, side, teamHasTimeAdvantage)) {
                return true;
            }
        }
    }

    // Check Board B (where partner of 'side' plays, so opponent color is ~side)
    if (pos[BOARD_B]->side_to_move() == ~side && pos[BOARD_B]->checkers()) {
        Stockfish::MoveList<Stockfish::LEGAL> legalMoves(*pos[BOARD_B]);
        if (!legalMoves.size()) {
            // No legal moves - but can partner provide a blocking piece?
            if (!can_partner_provide_blocking_piece(BOARD_B, ~side, teamHasTimeAdvantage)) {
                return true;
            }
        }
    }

    // Bughouse special case: if team has no legal moves at all (e.g., stalemate on their
    // board(s)) AND they don't have time advantage, they cannot pass/sit - this is a loss.
    // This handles positions like stalemate where the team cannot move but isn't in check.
    // Note: We check moves directly here to avoid infinite recursion with legal_moves().
    if (!teamHasTimeAdvantage) {
        bool hasMovesOnA = false;
        bool hasMovesOnB = false;
        bool isOnTurnOnA = (pos[BOARD_A]->side_to_move() == side);
        bool isOnTurnOnB = (pos[BOARD_B]->side_to_move() == ~side);
        
        // If neither board has the team on turn, they can't be checkmated this way
        // (opponent(s) are on turn, so it's not the team's problem yet)
        if (!isOnTurnOnA && !isOnTurnOnB) {
            return false;
        }
        
        // Check Board A (where 'side' plays)
        if (isOnTurnOnA) {
            Stockfish::MoveList<Stockfish::LEGAL> movesA(*pos[BOARD_A]);
            hasMovesOnA = (movesA.size() > 0);
        }
        
        // Check Board B (where partner of 'side' plays, so ~side)
        if (isOnTurnOnB) {
            Stockfish::MoveList<Stockfish::LEGAL> movesB(*pos[BOARD_B]);
            hasMovesOnB = (movesB.size() > 0);
        }
        
        // If on turn on at least one board but no moves on any, it's a loss
        if (!hasMovesOnA && !hasMovesOnB) {
            return true;
        }
    }

    return false;
}

// Helper function to check if partner can capture a piece that would allow blocking the check.
// board_in_check: the board index where the player is in check (0 or 1)
// checked_side: the color of the player being checked on that board
// teamHasTimeAdvantage: if true, partner may be able to capture in the future even if not their turn
bool Board::can_partner_provide_blocking_piece(int board_in_check, Stockfish::Color checked_side, bool teamHasTimeAdvantage) {
    int partner_board = (board_in_check == BOARD_A) ? BOARD_B : BOARD_A;
    Stockfish::Color partner_side = ~checked_side;  // Partner plays opposite color
    
    // Check if it's currently partner's turn
    bool is_partner_turn = (pos[partner_board]->side_to_move() == partner_side);
    
    // If it's not partner's turn and we don't have time advantage, they can't help
    // But if we have time advantage, they might capture something in the future
    if (!is_partner_turn && !teamHasTimeAdvantage) {
        return false;
    }
    
    // Get the king square and checker for the board in check
    Stockfish::Square ksq = pos[board_in_check]->square<Stockfish::KING>(checked_side);
    Stockfish::Bitboard checkers = pos[board_in_check]->checkers();
    
    // Double check - can only escape by king move, partner pieces won't help
    if (Stockfish::more_than_one(checkers)) {
        return false;
    }
    
    Stockfish::Square checker_sq = Stockfish::lsb(checkers);
    
    // Get squares between king and checker (where a drop could block)
    // For leaper attacks (knights), there are no blocking squares
    Stockfish::Bitboard blocking_squares = Stockfish::between_bb(ksq, checker_sq);
    
    // Knight, pawn, and king checks cannot be blocked by interposition
    // (between_bb returns empty for adjacent or knight-distance squares)
    if (!blocking_squares) {
        return false;
    }
    
    // Check if any blocking square is empty (available for a drop)
    Stockfish::Bitboard occupied = pos[board_in_check]->pieces();
    Stockfish::Bitboard available_blocks = blocking_squares & ~occupied;
    
    // If there are no empty blocking squares, a drop can't help
    if (!available_blocks) {
        return false;
    }
    
    // Check if there's at least one blocking square valid for pawns (ranks 2-7)
    Stockfish::Bitboard pawn_valid_blocks = available_blocks & ~(Stockfish::Rank1BB | Stockfish::Rank8BB);
    
    // If it's the partner's turn, check their current legal capture moves
    if (is_partner_turn) {
        Stockfish::MoveList<Stockfish::LEGAL> partner_moves(*pos[partner_board]);
        
        for (const Stockfish::ExtMove& ext_move : partner_moves) {
            Stockfish::Move move = ext_move;
            
            // Check if this is a capture move
            Stockfish::Square to = Stockfish::to_sq(move);
            Stockfish::Piece captured = Stockfish::type_of(move) == Stockfish::EN_PASSANT 
                ? Stockfish::make_piece(~partner_side, Stockfish::PAWN) 
                : pos[partner_board]->piece_on(to);
            
            if (captured == Stockfish::NO_PIECE) {
                continue;  // Not a capture
            }
            
            // Check if this piece type could be dropped on a blocking square
            Stockfish::PieceType captured_type = Stockfish::type_of(captured);
            
            // Pawns can only be dropped on ranks 2-7
            if (captured_type == Stockfish::PAWN) {
                if (pawn_valid_blocks) {
                    return true;
                }
            } else {
                // Non-pawn pieces can be dropped on any empty blocking square
                return true;
            }
        }
    } else {
        // Team has time advantage but it's not partner's turn yet
        // Check if opponent has any pieces that could potentially be captured in the future
        Stockfish::Color opponent_side = ~partner_side;
        
        // Check for each piece type if opponent has it and it could block if captured
        for (Stockfish::PieceType pt = Stockfish::PAWN; pt <= Stockfish::QUEEN; ++pt) {
            Stockfish::Bitboard opponent_pieces = pos[partner_board]->pieces(opponent_side, pt);
            
            if (opponent_pieces) {
                // Opponent has pieces of this type that could be captured in the future
                if (pt == Stockfish::PAWN) {
                    if (pawn_valid_blocks) {
                        return true;  // Pawn could block on valid squares
                    }
                } else {
                    return true;  // Non-pawn piece could block on any empty square
                }
            }
        }
    }
    
    return false;  // No partner capture can provide a blocking piece
}

void Board::make_moves(Stockfish::Move moveA, Stockfish::Move moveB) {
    Stockfish::Piece p;
    
    if (moveA != Stockfish::MOVE_NONE) {
        states[BOARD_A]->emplace_back();
        pos[BOARD_A]->do_move(moveA, states[BOARD_A]->back());
        p = states[BOARD_A]->back().pieceToHand; 
        if (p) {
            pos[BOARD_B]->add_to_hand(p);
        }
        // Record position for repetition detection
        record_position(BOARD_A);
    }
    
    if (moveB != Stockfish::MOVE_NONE) {
        states[BOARD_B]->emplace_back();
        pos[BOARD_B]->do_move(moveB, states[BOARD_B]->back());
        p = states[BOARD_B]->back().pieceToHand; 
        if (p) {
            pos[BOARD_A]->add_to_hand(p);
        }
        // Record position for repetition detection
        record_position(BOARD_B);
    }
}

void Board::unmake_moves(Stockfish::Move moveA, Stockfish::Move moveB) {
    if (moveB != Stockfish::MOVE_NONE) {
        Stockfish::Piece pB = states[BOARD_B]->back().pieceToHand;
        if (pB) {
            pos[BOARD_A]->remove_from_hand(pB);
        }
        pos[BOARD_B]->undo_move(moveB);
        states[BOARD_B]->pop_back();
        // Remove position from history
        unrecord_position(BOARD_B);
    }

    if (moveA != Stockfish::MOVE_NONE) {
        Stockfish::Piece pA = states[BOARD_A]->back().pieceToHand;
        if (pA) {
            pos[BOARD_B]->remove_from_hand(pA);
        }
        pos[BOARD_A]->undo_move(moveA);
        states[BOARD_A]->pop_back();
        // Remove position from history
        unrecord_position(BOARD_A);
    }
}
#include "environment/board.h"

#include <stdexcept>

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
    if (is_single_board()) {
        if (fen.find('|') != std::string::npos) {
            throw std::invalid_argument("Single-board variants expect one FEN");
        }
        set_fen(BOARD_A, trim(fen));
        set_fen(BOARD_B, startingFen);
        return;
    }
    if (fen.find('|') == std::string::npos) {
        throw std::invalid_argument("Bughouse expects two FENs separated by |");
    }
    std::stringstream ss(fen);
    std::string line; 
    getline(ss, line, '|');
    line = trim(line); 

    states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    pos[0]->set(Stockfish::variants.find("bughouse")->second, line, false, &states[0]->back(), Stockfish::Threads.main());
    clear_position_history(0);
    record_position(0);
    moveHistory[0].clear();
    
    getline(ss, line, '|');
    line = trim(line);
    
    states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    pos[1]->set(Stockfish::variants.find("bughouse")->second, line, false, &states[1]->back(), Stockfish::Threads.main());
    clear_position_history(1);
    record_position(1);
    moveHistory[1].clear();
}

// Default constructor: initializes the board positions and sets them to the starting FEN.
Board::Board() : Board(Variant::BUGHOUSE) {}

Board::Board(Variant selected) : variant(selected) {
    pos[0] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);
    pos[1] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);

    set_fen(BOARD_A, initial_fen());
    set_fen(BOARD_B, startingFen);
}

void Board::set_variant(Variant selected) {
    variant = selected;
    if (is_single_board()) {
        set(initial_fen());
    } else {
        set(startingFen + "|" + startingFen);
    }
}

bool Board::is_insufficient_material() const {
    using namespace Stockfish;
    if (!is_single_board()) return false;
    const auto& position = *pos[BOARD_A];
    if (variant == Variant::CRAZYHOUSE) {
        // Lichess Crazyhouse never adjudicates insufficient material.
        return false;
    }
    const auto occupied = position.pieces();
    const auto bishops = position.pieces(BISHOP);
    const auto pawns = position.pieces(PAWN);
    const auto kings = position.pieces(KING) | position.pieces(COMMONER);
    const bool oppositeBishops = (bishops & DarkSquares) && (bishops & ~DarkSquares);
    if (variant == Variant::THREE_CHECK) return occupied == kings;

    // Lichess also recognizes blocked pawn/bishop structures in Antichess
    // and Atomic. Test pawn mobility for both colors under the native rules.
    auto blockedPawns = [&](Bitboard subset) {
        if (!subset) return true;
        for (Color color : {WHITE, BLACK}) {
            auto remaining = subset & position.pieces(color);
            if (!remaining) continue;
            std::string fen = position.fen(false, true);
            fen[fen.find(' ') + 1] = color == WHITE ? 'w' : 'b';
            StateInfo state{};
            Position scratch;
            scratch.set(position.variant(), fen, false, &state, Threads.main());
            for (const auto& move : MoveList<LEGAL>(scratch)) {
                if (type_of(move) != DROP && (remaining & square_bb(from_sq(move)))) return false;
            }
            while (remaining) {
                const auto square = pop_lsb(remaining);
                const int next = static_cast<int>(square) + (color == WHITE ? 8 : -8);
                if (next < 0 || next >= 64 || !(pawns & square_bb(Square(next)))) return false;
            }
        }
        return true;
    };
    if (variant == Variant::ANTICHESS) {
        if (occupied != (bishops | pawns)) return false;
        const auto whiteBishops = bishops & position.pieces(WHITE);
        const auto blackBishops = bishops & position.pieces(BLACK);
        auto oneColor = [&](Bitboard pieces) {
            return pieces && (!(pieces & DarkSquares) || !(pieces & ~DarkSquares));
        };
        if (!oneColor(whiteBishops) || !oneColor(blackBishops)
            || bool(whiteBishops & DarkSquares) == bool(blackBishops & DarkSquares)) return false;
        for (Color color : {WHITE, BLACK}) {
            const auto enemyBishops = bishops & position.pieces(~color);
            const auto attackableColor = enemyBishops & DarkSquares ? DarkSquares : ~DarkSquares;
            if (pawns & position.pieces(color) & attackableColor) return false;
        }
        return blockedPawns(pawns);
    }
    if (variant == Variant::ATOMIC) {
        const int count = position.count<ALL_PIECES>();
        bool insufficient;
        if (position.count<ALL_PIECES>(WHITE) >= 2 && position.count<ALL_PIECES>(BLACK) >= 2) {
            insufficient = occupied == (kings | bishops) && count <= 4 && oppositeBishops;
        } else if (occupied == (kings | position.pieces(KNIGHT))) {
            insufficient = count <= 4;
        } else {
            insufficient = occupied == (kings | bishops | position.pieces(KNIGHT) | position.pieces(ROOK))
                && !oppositeBishops && count <= 3;
        }
        if (insufficient) return true;
        if (occupied != (kings | bishops | pawns) || !blockedPawns(pawns)) return false;
        if (!bishops) return true;
        const auto square = lsb(bishops);
        const Color side = color_of(position.piece_on(square));
        const bool dark = bool(square_bb(square) & DarkSquares);
        const auto bishopColor = dark ? DarkSquares : ~DarkSquares;
        return !(bishops & ~position.pieces(side)) && !(bishops & ~bishopColor)
            && !(pawns & position.pieces(~side) & bishopColor);
    }
    if (position.count<Stockfish::PAWN>() || position.count<Stockfish::ROOK>()
        || position.count<Stockfish::QUEEN>()) return false;
    if (position.count<Stockfish::ALL_PIECES>() <= 3) return true;
    if (position.count<Stockfish::KNIGHT>()) return false;
    return !(bishops & Stockfish::DarkSquares) || !(bishops & ~Stockfish::DarkSquares);
}

bool Board::single_board_terminal_value(float& value) {
    if (!is_single_board()) return false;
    auto& position = *pos[BOARD_A];
    Stockfish::Value nativeValue;
    if (position.is_immediate_game_end(nativeValue)) {
        value = nativeValue > Stockfish::VALUE_DRAW ? 1.0f
            : nativeValue < Stockfish::VALUE_DRAW ? -1.0f : 0.0f;
        return true;
    }
    if (!has_any_legal_move(BOARD_A)) {
        nativeValue = position.checkers() ? position.checkmate_value() : position.stalemate_value();
        value = nativeValue > Stockfish::VALUE_DRAW ? 1.0f
            : nativeValue < Stockfish::VALUE_DRAW ? -1.0f : 0.0f;
        return true;
    }
    if ((variant != Variant::CRAZYHOUSE && rule50_count(BOARD_A) >= 100)
        || repetition_count(BOARD_A) >= 3 || is_insufficient_material()) {
        value = 0.0f;
        return true;
    }
    return false;
}

// Copy constructor: copies state history and reinitializes positions from the provided board.
Board::Board(const Board& board) : variant(board.variant) {
    pos[0] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);
    pos[1] = std::unique_ptr<Stockfish::Position>(new Stockfish::Position);

    // A copied search board only needs the current Position state: all moves
    // made on the copy are undone back to this root, never into the source's
    // StateInfo chain. Rebuilding that entire deque was pure allocation/copy
    // overhead in the hottest setup path.
    states[0] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
    states[1] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));

    pos[0]->set(native_variant(0), board.pos[0]->fen(false, true), board.pos[0]->is_chess960(), &states[0]->back(), Stockfish::Threads.main());
    pos[1]->set(native_variant(1), board.pos[1]->fen(false, true), board.pos[1]->is_chess960(), &states[1]->back(), Stockfish::Threads.main());
    
    // Copy position history
    positionHistory[0] = board.positionHistory[0];
    positionHistory[1] = board.positionHistory[1];
    repetitionCounts[0] = board.repetitionCounts[0];
    repetitionCounts[1] = board.repetitionCounts[1];
    repetitionFingerprint[0] = board.repetitionFingerprint[0];
    repetitionFingerprint[1] = board.repetitionFingerprint[1];
    moveHistory[0] = board.moveHistory[0];
    moveHistory[1] = board.moveHistory[1];
}

// Executes a move on the board and updates the corresponding state.
// Also adds a piece to the opponent's hand if necessary.
void Board::push_move(int board_num, Stockfish::Move move) {
    states[board_num]->emplace_back();
    pos[board_num]->do_move(move, states[board_num]->back());
    Stockfish::Piece p = states[board_num]->back().pieceToHand; 
    if (p && !is_single_board()) {
        pos[1 - board_num]->add_to_hand_with_key(p);
    }
    // Record position for repetition detection
    record_position(board_num);
    moveHistory[board_num].push_back(move);
}

bool Board::is_legal_move(int board_num, Stockfish::Move move) const {
    if (is_single_board() && (board_num != BOARD_A || move == Stockfish::MOVE_NONE)) {
        return false;
    }
    if (move == Stockfish::MOVE_NONE) {
        return true;
    }
    const Stockfish::Position& position = *pos[board_num];
    return !position.is_immediate_game_end()
        && position.pseudo_legal(move)
        && !position.virtual_drop(move)
        && position.legal(move);
}

bool Board::has_any_legal_move(int board_num) const {
    if (is_single_board() && board_num != BOARD_A) {
        return false;
    }
    const Stockfish::Position& position = *pos[board_num];
    if (position.is_immediate_game_end()) {
        return false;
    }

    // In bughouse an available drop is immediately legal when the king is not
    // in check. Test the actual drop regions so pawn back-rank restrictions are
    // respected without invoking move generation.
    if (!position.checkers()
        && position.piece_drops()
        && position.count_in_hand(position.side_to_move(), Stockfish::ALL_PIECES) > 0) {
        const Stockfish::Color side = position.side_to_move();
        const Stockfish::Bitboard emptySquares = position.board_bb() & ~position.pieces();
        for (Stockfish::PieceType pieceType : position.piece_types()) {
            if (position.count_in_hand(side, pieceType) > 0
                && (position.drop_region(side, pieceType) & emptySquares)) {
                return true;
            }
        }
    }

    // Avoid LEGAL move generation, which checks and compacts every candidate.
    // We only need one legal, non-virtual candidate.
    Stockfish::ExtMove candidates[Stockfish::MAX_MOVES];
    Stockfish::ExtMove* end = position.checkers()
        ? Stockfish::generate<Stockfish::EVASIONS>(position, candidates)
        : Stockfish::generate<Stockfish::NON_EVASIONS>(position, candidates);
    for (Stockfish::ExtMove* candidate = candidates; candidate != end; ++candidate) {
        if (position.legal(*candidate) && !position.virtual_drop(*candidate)) {
            return true;
        }
    }
    return false;
}

// Reverts the last move on the board and updates the state.
// Also removes a piece from the opponent's hand if necessary.
void Board::pop_move(int board_num) {
    Stockfish::Move m = states[board_num]->back().move; 
    Stockfish::Piece p = states[board_num]->back().pieceToHand; 
    if (p && !is_single_board()) {
        pos[1 - board_num]->remove_from_hand_with_key(p);
    }
    pos[board_num]->undo_move(m); 
    states[board_num]->pop_back();
    // Remove position from history
    unrecord_position(board_num);
    if (!moveHistory[board_num].empty()) {
        moveHistory[board_num].pop_back();
    }
}

// Returns a list of legal moves for the specified board index.
namespace {

// Rook and bishop promotions have no policy index, so nothing the engine
// searches may play one: a queen does everything either can, mate-wise, and
// a knight covers the rest. Opponent moves arrive through is_legal_move and
// UCI parsing, which still accept them.
bool is_unplayable_promotion(Stockfish::Move move) {
    if (Stockfish::type_of(move) != Stockfish::PROMOTION) {
        return false;
    }
    const Stockfish::PieceType promoted = Stockfish::promotion_type(move);
    return promoted != Stockfish::QUEEN && promoted != Stockfish::KNIGHT;
}

}  // namespace

std::vector<Stockfish::Move> Board::legal_moves(int board_num) {
    if (is_single_board() && (board_num != BOARD_A || pos[board_num]->is_immediate_game_end())) {
        return {};
    }
    const Stockfish::MoveList<Stockfish::LEGAL> candidates(*pos[board_num]);
    std::vector<Stockfish::Move> legal_moves;
    legal_moves.reserve(candidates.size());
    for (const Stockfish::ExtMove& move : candidates) {
        if (is_single_board() || !is_unplayable_promotion(move)) {
            legal_moves.emplace_back(move);
        }
    }
    return legal_moves;
}

std::vector<Stockfish::Move> Board::checking_moves(int board_num) const {
    if (is_single_board() && board_num != BOARD_A) {
        return {};
    }
    const Stockfish::Position& position = *pos[board_num];
    std::vector<Stockfish::Move> checks;
    if (position.is_immediate_game_end()) {
        return checks;
    }

    // Keep all move types, including checking evasions, en passant, castling
    // and knight promotions. QUIET_CHECKS alone omits some of those. Filtering
    // before legality avoids validating hundreds of irrelevant pocket drops.
    Stockfish::ExtMove candidates[Stockfish::MAX_MOVES];
    const Stockfish::ExtMove* end = position.checkers()
        ? Stockfish::generate<Stockfish::EVASIONS>(position, candidates)
        : Stockfish::generate<Stockfish::NON_EVASIONS>(position, candidates);
    for (const Stockfish::ExtMove* candidate = candidates; candidate != end; ++candidate) {
        if (!position.virtual_drop(*candidate)
            && (is_single_board() || !is_unplayable_promotion(*candidate))
            && position.gives_check(*candidate)
            && position.legal(*candidate)) {
            checks.push_back(*candidate);
        }
    }
    return checks;
}

// Returns a list of legal moves for the specified side by checking both boards.
std::vector<std::pair<int, Stockfish::Move>> Board::legal_moves(Stockfish::Color side, bool teamHasTimeAdvantage) {
    if (is_single_board()) {
        std::vector<std::pair<int, Stockfish::Move>> moves;
        if (side == side_to_move(BOARD_A)) {
            for (Stockfish::Move move : legal_moves(BOARD_A)) {
                moves.emplace_back(BOARD_A, move);
            }
        }
        return moves;
    }
    std::vector<std::pair<int, Stockfish::Move>> moves;

    // If checkmate, return an empty move list.
    if (is_checkmate(side, teamHasTimeAdvantage)) {
        return {};
    }
    
    if (pos[0]->side_to_move() == side) {
        for (const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[0])) {
            if (!is_unplayable_promotion(move)) {
                moves.emplace_back(0, move);
            }
        }
    }

    if (pos[1]->side_to_move() == ~side) {
        for (const Stockfish::ExtMove& move : Stockfish::MoveList<Stockfish::LEGAL>(*pos[1])) {
            if (!is_unplayable_promotion(move)) {
                moves.emplace_back(1, move);
            }
        }
    }
    return moves;
}

// Determines if the board is in checkmate for the given side in bughouse.
// In bughouse, checkmate is more complex because partner captures can provide
// pieces to drop and block a check. We must verify that:
// 1. The player has no legal moves (including drops with current pieces in hand)
// 2. The partner cannot capture any piece that could be used to block the check
bool Board::is_checkmate(Stockfish::Color side,
                         bool teamHasTimeAdvantage,
                         LegalMoveCache* legalMoveCache,
                         bool assumePartnerCanBlock) {
    if (is_single_board()) {
        return side == side_to_move(BOARD_A) && is_in_check(BOARD_A)
            && !has_any_legal_move(BOARD_A);
    }
    const bool isOnTurnOnA = pos[BOARD_A]->side_to_move() == side;
    const bool isOnTurnOnB = pos[BOARD_B]->side_to_move() == ~side;

    LegalMoveCache localLegalMoveCache;
    LegalMoveCache& cache = legalMoveCache ? *legalMoveCache : localLegalMoveCache;
    auto has_legal_move = [&](int boardNum) {
        if (!cache[boardNum].has_value()) {
            cache[boardNum] = has_any_legal_move(boardNum);
        }
        return *cache[boardNum];
    };

    auto partner_capture_unfreezes = [&](int stuckBoard, int partnerBoard) {
        for (const Stockfish::ExtMove& extMove
             : Stockfish::MoveList<Stockfish::LEGAL>(*pos[partnerBoard])) {
            const Stockfish::Move move = extMove;
            if (!is_capture(partnerBoard, move)) {
                continue;
            }
            push_move(partnerBoard, move);
            const bool unfrozen = has_any_legal_move(stuckBoard);
            pop_move(partnerBoard);
            if (unfrozen) {
                return true;
            }
        }
        return false;
    };

    // Check Board A (where 'side' plays)
    if (isOnTurnOnA && pos[BOARD_A]->checkers() && !has_legal_move(BOARD_A)) {
        // No legal moves - but can partner provide a blocking piece?
        if (!can_partner_provide_blocking_piece(
                BOARD_A, side, teamHasTimeAdvantage, assumePartnerCanBlock)) {
            return true;
        }
    }

    // Check Board B (where partner of 'side' plays, so opponent color is ~side)
    if (isOnTurnOnB && pos[BOARD_B]->checkers() && !has_legal_move(BOARD_B)) {
        // No legal moves - but can partner provide a blocking piece?
        if (!can_partner_provide_blocking_piece(
                BOARD_B, ~side, teamHasTimeAdvantage, assumePartnerCanBlock)) {
            return true;
        }
    }

    // If the team has no legal move, it loses unless time advantage makes
    // double-sit legal. Double-sit is still forbidden when both boards are on turn.
    if (isOnTurnOnA || isOnTurnOnB) {
        const bool hasMovesOnA = isOnTurnOnA && has_legal_move(BOARD_A);
        const bool hasMovesOnB = isOnTurnOnB && has_legal_move(BOARD_B);
        if (!teamHasTimeAdvantage) {
            if (isOnTurnOnA && !hasMovesOnA && hasMovesOnB
                && !partner_capture_unfreezes(BOARD_A, BOARD_B)) {
                return true;
            }
            if (isOnTurnOnB && !hasMovesOnB && hasMovesOnA
                && !partner_capture_unfreezes(BOARD_B, BOARD_A)) {
                return true;
            }
        }
        if (!hasMovesOnA && !hasMovesOnB
            && (!teamHasTimeAdvantage || (isOnTurnOnA && isOnTurnOnB))) {
            return true;
        }
    }

    return false;
}

// Helper function to check if partner can capture a piece that would allow blocking the check.
// board_in_check: the board index where the player is in check (0 or 1)
// checked_side: the color of the player being checked on that board
// teamHasTimeAdvantage: if true, partner may be able to capture in the future even if not their turn
bool Board::can_partner_provide_blocking_piece(int board_in_check, Stockfish::Color checked_side, bool teamHasTimeAdvantage, bool assumePartnerCanBlock) {
    if (is_single_board()) {
        return false;
    }
    int partner_board = (board_in_check == BOARD_A) ? BOARD_B : BOARD_A;
    Stockfish::Color partner_side = ~checked_side;  // Partner plays opposite color
    
    // Check if it's currently partner's turn
    bool is_partner_turn = (pos[partner_board]->side_to_move() == partner_side);
    
    // If it's not partner's turn and we don't have time advantage, they can't help
    // But if we have time advantage, they might capture something in the future
    if (!is_partner_turn && !teamHasTimeAdvantage && !assumePartnerCanBlock) {
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
    
    auto has_useful_capture = [&](Board& candidateBoard) {
        Stockfish::MoveList<Stockfish::LEGAL> partner_moves(*candidateBoard.pos[partner_board]);

        for (const Stockfish::ExtMove& ext_move : partner_moves) {
            Stockfish::Move move = ext_move;
            Stockfish::Square to = Stockfish::to_sq(move);
            Stockfish::Piece captured = Stockfish::type_of(move) == Stockfish::EN_PASSANT 
                ? Stockfish::make_piece(~partner_side, Stockfish::PAWN) 
                : candidateBoard.pos[partner_board]->piece_on(to);

            if (captured == Stockfish::NO_PIECE) {
                continue;
            }

            Stockfish::PieceType captured_type =
                (candidateBoard.pos[partner_board]->promotedPieces & to)
                    ? Stockfish::PAWN
                    : Stockfish::type_of(captured);

            if (captured_type == Stockfish::PAWN) {
                if (pawn_valid_blocks) {
                    return true;
                }
            } else {
                return true;
            }
        }
        return false;
    };

    // Every test up to here reads the checked board's geometry alone, so this
    // is the point where the answer would start to depend on the partner
    // board. A caller that asked for the blockable-by-assumption model stops
    // here with "yes".
    if (assumePartnerCanBlock) {
        return true;
    }

    if (is_partner_turn) {
        return has_useful_capture(*this);
    }

    // With time advantage the checked team may wait, but the opponent chooses the
    // intervening move. A future blocker is guaranteed only if every reply leaves
    // the partner an immediate useful capture.
    if (teamHasTimeAdvantage) {
        Stockfish::MoveList<Stockfish::LEGAL> opponent_moves(*pos[partner_board]);
        if (!opponent_moves.size()) {
            return false;
        }

        // MoveList holds its own copy of the generated moves, so probing each
        // reply with make/unmake on this board is safe - and far cheaper than
        // copying the whole Board once per reply, which this predicate does for
        // every candidate move of every mate scan.
        for (const Stockfish::ExtMove& opponentMove : opponent_moves) {
            push_move(partner_board, opponentMove);
            const bool partnerCanCapture = has_useful_capture(*this);
            pop_move(partner_board);
            if (!partnerCanCapture) {
                return false;
            }
        }
        return true;
    }

    return false;
}

void Board::make_moves(Stockfish::Move moveA, Stockfish::Move moveB) {
#ifndef NDEBUG
    if (!is_legal_move(BOARD_A, moveA) || !is_legal_move(BOARD_B, moveB)) {
        throw std::logic_error(
            "Search tree supplied an illegal joint action (A="
            + std::to_string(static_cast<uint32_t>(moveA))
            + ", B=" + std::to_string(static_cast<uint32_t>(moveB)) + ")");
    }
#endif

    auto apply_move = [&](int boardNum, Stockfish::Move move) {
        if (move == Stockfish::MOVE_NONE) {
            return;
        }
        states[boardNum]->emplace_back();
        pos[boardNum]->do_move(move, states[boardNum]->back());
        Stockfish::Piece p = states[boardNum]->back().pieceToHand;
        if (p) {
            pos[1 - boardNum]->add_to_hand_with_key(p);
        }
        record_position(boardNum);
        moveHistory[boardNum].push_back(move);
    };

    apply_move(BOARD_A, moveA);
    apply_move(BOARD_B, moveB);
}

void Board::unmake_moves(Stockfish::Move moveA, Stockfish::Move moveB) {
    auto undo_move = [&](int boardNum, Stockfish::Move move) {
        if (move == Stockfish::MOVE_NONE) {
            return;
        }
        Stockfish::Piece p = states[boardNum]->back().pieceToHand;
        if (p) {
            pos[1 - boardNum]->remove_from_hand_with_key(p);
        }
        pos[boardNum]->undo_move(move);
        states[boardNum]->pop_back();
        unrecord_position(boardNum);
        moveHistory[boardNum].pop_back();
    };

    undo_move(BOARD_B, moveB);
    undo_move(BOARD_A, moveA);
}

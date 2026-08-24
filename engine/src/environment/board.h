#pragma once

#include "environment/constants.h"
#include "environment/zobrist.h"

#include <sstream>
#include <string>
#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <vector>
#include <functional>

#include "Fairy-Stockfish/src/apiutil.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Represents a chess board with dual perspectives.
 *
 * This class encapsulates the state of a chess board, including position data and move history,
 * and provides methods to manipulate and query the board state using FEN strings and UCI moves.
 */
class Board {
    public: 
        std::unique_ptr<Stockfish::Position> pos[2]; ///< Array of board positions.
        Stockfish::StateListPtr states[2];             ///< Array of state history pointers.
        const std::string startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"; ///< Standard starting position in FEN notation.
        
        /// History of repetition keys for on-board positions.
        std::vector<uint64_t> positionHistory[2];
        std::vector<uint64_t> positionHistoryPrefixes[2];
        std::vector<Stockfish::Move> moveHistory[2];

        Board();
        Board(const Board& board);

        /**
        * @brief Generates a hash key for the board.
        * @param teamHasTimeAdvantage Whether the team to move has time advantage (affects sitting rules)
        * @return long unsigned int Combined hash of the positions.
        */
        unsigned long hash_key(bool teamHasTimeAdvantage = false) {
            auto k0 = mix_hash(pos[0]->key(), static_cast<uint64_t>(rule50_count(0)));
            auto k1 = mix_hash(pos[1]->key(), static_cast<uint64_t>(rule50_count(1)));
            // Combines the two keys using a hash_combine technique.
            auto combined = k0 ^ (k1 + 0x9e3779b97f4a7c15UL + (k0 << 6) + (k0 >> 2));
            auto repetitionContext = history_key(0);
            repetitionContext ^= history_key(1)
                + 0x9e3779b97f4a7c15ULL
                + (repetitionContext << 6)
                + (repetitionContext >> 2);
            combined ^= repetitionContext
                + 0x9e3779b97f4a7c15ULL
                + (combined << 6)
                + (combined >> 2);
            // XOR in time advantage key if team is up on time
            return teamHasTimeAdvantage ? (combined ^ Stockfish::Zobrist::timeAdvantage) : combined;
        }

        /**
         * @brief Computes a repetition key for a single board.
         * Pocket pieces are ignored for bughouse repetition claims.
         * @param board_num The board index.
         * @return uint64_t Hash of the on-board position.
         */
        uint64_t board_only_key(int board_num) const {
            // Zobrist over the same fields the FEN form used to encode: piece
            // placement including promotion markers, side to move, castling
            // rights and the en passant file. Pockets stay excluded. This runs
            // on every move made in the search, so it must not build a string.
            const Stockfish::Position& position = *pos[board_num];
            uint64_t key = position.board_key();
            Stockfish::Bitboard promoted = position.promotedPieces;
            while (promoted) {
                key ^= Stockfish::Zobrist::boardPromoted[
                    Stockfish::pop_lsb(promoted)];
            }

            return key;
        }

        /**
         * @brief Adds current position to history for repetition tracking.
         * @param board_num The board index.
         */
        void record_position(int board_num) {
            const uint64_t positionKey = board_only_key(board_num);
            positionHistory[board_num].push_back(positionKey);
            const uint64_t prefix = positionHistoryPrefixes[board_num].empty()
                ? HISTORY_HASH_SEED
                : positionHistoryPrefixes[board_num].back();
            positionHistoryPrefixes[board_num].push_back(mix_hash(prefix, positionKey));
        }

        /**
         * @brief Removes the last position from history (for unmake_moves).
         * @param board_num The board index.
         */
        void unrecord_position(int board_num) {
            if (!positionHistory[board_num].empty()) {
                positionHistory[board_num].pop_back();
                positionHistoryPrefixes[board_num].pop_back();
            }
        }

        /**
         * @brief Clears position history for a board.
         * @param board_num The board index.
         */
        void clear_position_history(int board_num) {
            positionHistory[board_num].clear();
            positionHistoryPrefixes[board_num].clear();
        }

        static constexpr uint64_t HISTORY_HASH_SEED = 0xcbf29ce484222325ULL;

        static uint64_t mix_hash(uint64_t key, uint64_t value) {
            value += 0x9e3779b97f4a7c15ULL;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
            return key ^ (value ^ (value >> 31));
        }

        uint64_t history_key(int board_num) const {
            const uint64_t prefix = positionHistoryPrefixes[board_num].empty()
                ? HISTORY_HASH_SEED
                : positionHistoryPrefixes[board_num].back();
            return mix_hash(prefix, positionHistory[board_num].size());
        }

        void set(std::string fen); 
        void push_move(int board_num, Stockfish::Move move);
        void make_moves(Stockfish::Move moveA, Stockfish::Move moveB);
        void unmake_moves(Stockfish::Move moveA, Stockfish::Move moveB);
        void pop_move(int board_num);
        bool is_legal_move(int board_num, Stockfish::Move move) const;
        bool has_any_legal_move(int board_num) const;
        std::vector<Stockfish::Move> legal_moves(int board_num);
        std::vector<std::pair<int, Stockfish::Move>> legal_moves(Stockfish::Color side, bool teamHasTimeAdvantage = false);

        /**
         * @brief Adds a piece to the player's hand.
         * @param board_num The index of the board.
         * @param p The piece to add.
         */
        void add_to_hand(int board_num, Stockfish::Piece p) {
            pos[board_num]->add_to_hand_with_key(p);
        }

        /**
         * @brief Removes a piece from the player's hand.
         * @param board_num The index of the board.
         * @param p The piece to remove.
         */
        void remove_from_hand(int board_num, Stockfish::Piece p) {
            pos[board_num]->remove_from_hand_with_key(p);
        }

        /**
         * @brief Returns the FEN string for the specified board.
         * @param board_num The board index.
         * @return std::string FEN representation of the board.
         */
        std::string fen(int board_num) const {
            return pos[board_num]->fen(false, true); 
        }

        /**
         * @brief Sets the board state from a FEN string.
         * @param board_num The board index.
         * @param fen FEN string representing the board state.
         */
        void set_fen(int board_num, std::string fen) {
            states[board_num] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
            states[board_num]->emplace_back();
            pos[board_num]->set(Stockfish::variants.find("bughouse")->second, fen, false, &states[board_num]->back(), Stockfish::Threads.main());
            // Reset position history for this board
            clear_position_history(board_num);
            record_position(board_num);
            moveHistory[board_num].clear();
        }

        /**
         * @brief Returns a string representing the pieces held in hand.
         * @param board_num The board index.
         * @return std::string A string encoding the pieces in hand.
         */
        std::string get_hand(int board_num) {
            std::string hand; 
            for (Stockfish::Color color : {Stockfish::WHITE, Stockfish::BLACK}) {
                for (Stockfish::PieceType piece = Stockfish::QUEEN; piece >= Stockfish::PAWN; --piece) {
                    int pocket_cnt = count_in_hand(board_num, color, piece);
                    hand += std::string(pocket_cnt, pos[board_num]->piece_to_char()[Stockfish::make_piece(color, piece)]);
                }
            }
            return hand; 
        }
        
        /**
         * @brief Determines if a move is a capture.
         * @param board_num The board index.
         * @param move The move to evaluate.
         * @return true if the move is a capture, false otherwise.
         */
        bool is_capture(int board_num, Stockfish::Move move) {
            return pos[board_num]->capture(move);
        }

        /**
         * @brief Retrieves the character representing the captured piece.
         * @param board_num The board index.
         * @param move The move to evaluate.
         * @return char Character corresponding to the captured piece.
         */
        char get_captured_piece(int board_num, Stockfish::Move move) {
            Stockfish::Color us = side_to_move(board_num);
            Stockfish::Color them = ~us;
            Stockfish::Square to = to_sq(move);
            Stockfish::Piece captured = type_of(move) == Stockfish::EN_PASSANT ? make_piece(them, Stockfish::PAWN) : pos[board_num]->piece_on(to);
            
            return pos[board_num]->piece_to_char()[captured];
        }
        
        /**
         * @brief Counts the number of a specific piece in hand.
         * @param board_num The board index.
         * @param c The color of the piece.
         * @param pt The type of the piece.
         * @return int Number of pieces of the specified type in hand.
         */
        int count_in_hand(int board_num, Stockfish::Color c, Stockfish::PieceType pt) {
            return pos[board_num]->count_in_hand(c, pt);
        }

        /**
         * @brief Returns the side to move on the specified board.
         * @param board_num The board index.
         * @return Stockfish::Color The color of the side to move.
         */
        Stockfish::Color side_to_move(int board_num) {
            return pos[board_num]->side_to_move();
        }

        /**
         * @brief Returns a bitboard representing pieces of a specific type and color.
         * @param board_num The board index.
         * @param c The color of the pieces.
         * @param pt The piece type.
         * @return Stockfish::Bitboard Bitboard for the pieces.
         */
        Stockfish::Bitboard pieces(int board_num, Stockfish::Color c, Stockfish::PieceType pt) {
            return pos[board_num]->pieces(c, pt); 
        }

        /**
         * @brief Returns a bitboard representing all pieces of a specific color.
         * @param board_num The board index.
         * @param c The color of the pieces.
         * @return Stockfish::Bitboard Bitboard for the pieces.
         */
        Stockfish::Bitboard pieces(int board_num, Stockfish::Color c) {
            return pos[board_num]->pieces(c); 
        }

        /**
         * @brief Returns the en passant square for the specified board.
         * @param board_num The board index.
         * @return Stockfish::Square The en passant square.
         */
        Stockfish::Square ep_square(int board_num) {
            return pos[board_num]->ep_square();
        }

        /**
         * @brief Returns a bitboard of promoted pieces.
         * @param board_num The board index.
         * @return Stockfish::Bitboard Bitboard for promoted pieces.
         */
        Stockfish::Bitboard promoted_pieces(int board_num) { 
            return pos[board_num]->promotedPieces; 
        }

        /**
         * @brief Returns the current game ply.
         * @param board_num The board index.
         * @return int The game ply count.
         */
        int game_ply(int board_num) { 
            return pos[board_num]->game_ply(); 
        }

        /**
         * @brief Checks if castling is possible given the castling rights.
         * @param board_num The board index.
         * @param cr The castling rights.
         * @return true if castling is possible, false otherwise.
         */
        bool can_castle(int board_num, Stockfish::CastlingRights cr) { 
            return pos[board_num]->can_castle(cr); 
        }

        /**
         * @brief Returns the count for the fifty-move rule.
         * @param board_num The board index.
         * @return int The fifty-move rule counter.
         */
        int rule50_count(int board_num) { 
            return pos[board_num]->rule50_count(); 
        }

        Stockfish::Move last_move(int board_num) const {
            if (moveHistory[board_num].empty()) {
                return Stockfish::MOVE_NONE;
            }
            return moveHistory[board_num].back();
        }

        int repetition_count(int board_num) {
            if (positionHistory[board_num].empty()) {
                return 0;
            }
            const uint64_t currentKey = positionHistory[board_num].back();
            return static_cast<int>(std::count(
                positionHistory[board_num].begin(),
                positionHistory[board_num].end(),
                currentKey));
        }

        /**
         * @brief Converts a move to a UCI string.
         * @param board_num The board index.
         * @param move The move to convert.
         * @return std::string The UCI move string.
         */
        std::string uci_move(int board_num, Stockfish::Move move) { 
            if (move == Stockfish::MOVE_NONE) {
                return "pass";
            }

            // Get the UCI move string
            std::string move_str = Stockfish::UCI::move(*pos[board_num], move).c_str();

            // Concatenate board_num and move_str without a space
            return move_str;
        }

        /**
         * @brief Converts a move to SAN (Standard Algebraic Notation) string.
         * @param board_num The board index.
         * @param move The move to convert.
         * @return std::string The SAN move string (e.g., "e4", "Nf3", "O-O").
         */
        std::string san_move(int board_num, Stockfish::Move move) { 
            if (move == Stockfish::MOVE_NONE) {
                return "pass";
            }

            // Get the SAN move string using Fairy-Stockfish's move_to_san
            return Stockfish::SAN::move_to_san(*pos[board_num], move, Stockfish::NOTATION_SAN);
        }

        using LegalMoveCache = std::array<std::optional<bool>, 2>;

        bool is_checkmate(Stockfish::Color side,
                          bool teamHasTimeAdvantage = false,
                          LegalMoveCache* legalMoveCache = nullptr);
        
        /**
         * @brief Checks if partner can capture a piece that could block a check.
         * Used for proper bughouse checkmate detection.
         * @param board_in_check The board index where the player is in check (0 or 1)
         * @param checked_side The color of the player being checked on that board
         * @param teamHasTimeAdvantage If true, partner may capture in the future even if not their turn
         * @return true if partner can provide a blocking piece, false otherwise
         */
        bool can_partner_provide_blocking_piece(int board_in_check, Stockfish::Color checked_side, bool teamHasTimeAdvantage = false);

        bool is_in_check(int board_num) {
            return pos[board_num]->checkers();
        }

        bool gives_check(int board_num, Stockfish::Move move) const {
            return move != Stockfish::MOVE_NONE && pos[board_num]->gives_check(move);
        }

        /**
         * @brief Check if a SPECIFIC color's king is being attacked on a board.
         * This checks if the given color's king is in check, regardless of whose turn it is.
         * @param board_num The board index (0 or 1)
         * @param color The color whose king to check
         * @return true if that color's king is being attacked by opponent pieces
         */
        bool is_king_attacked(int board_num, Stockfish::Color color) {
            Stockfish::Square kingSquare = pos[board_num]->square<Stockfish::KING>(color);
            if (kingSquare == Stockfish::SQ_NONE) {
                return false;  // King not on board (shouldn't happen in normal chess)
            }
            // Check if any opponent pieces attack the king's square
            Stockfish::Bitboard attackers = pos[board_num]->attackers_to(kingSquare, ~color);
            return attackers != 0;
        }

        /**
         * @brief Check if either board is in a draw state.
         * @param ply The current search depth (used for repetition detection)
         *            When ply > 0, 2-fold repetition within search is treated as draw.
         *            When ply = 0, requires 3-fold repetition.
         * @return true if either board has reached a draw condition.
         */
        bool is_draw(int ply = 0) {
            return is_draw_on_board(0, ply) || is_draw_on_board(1, ply); 
        }

        /**
         * @brief Check for a draw using search depths tracked independently per board.
         *
         * A twofold repetition is only terminal when the repeated position was
         * reached by a move on that board inside the search. In bughouse, a
         * search ply may advance one board while the other board waits, so a
         * single combined ply would incorrectly promote an existing twofold on
         * the waiting board to a draw.
         */
        bool is_draw(const std::array<int, 2>& board_search_plies) {
            return is_draw_on_board(BOARD_A, board_search_plies[BOARD_A])
                || is_draw_on_board(BOARD_B, board_search_plies[BOARD_B]);
        }

        /**
         * @brief Check if a specific board is in a draw state.
         * @param board_num The board index.
         * @param ply The current search depth (used for repetition detection)
         *            When ply > 0, 2-fold repetition within search is treated as draw.
         *            When ply = 0, requires 3-fold repetition.
         * @return true if the board has reached a draw condition.
         */
        bool is_draw_on_board(int board_num, int ply = 0) {
            // Check 50-move rule using Fairy-Stockfish's built-in detection
            if (pos[board_num]->rule50_count() >= 100) {
                return true;
            }
            
            // Check for 3-fold (or 2-fold in search) repetition.
            const int threshold = ply > 0 ? 2 : 3;
            return repetition_count(board_num) >= threshold;
        }
};

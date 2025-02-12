#pragma once

#include <sstream>
#include <string>
#include <algorithm>
#include <iostream>

#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/uci.h"

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

        Board();
        Board(const Board& board);
        ~Board(); 

        /**
         * @brief Generates a hash key for the board.
         * @return std::string Concatenated position keys as a hash.
         */
        std::string hash_key() {
            return std::to_string(pos[0]->key()) + std::to_string(pos[1]->key()); 
        }

        /**
         * @brief Swaps the positions and states of the two boards.
         */
        void swap_boards() {
            std::swap(pos[0], pos[1]);
            std::swap(states[0], states[1]);
        }

        void set(std::string fen); 
        void push_move(int board_num, Stockfish::Move move);
        void pop_move(int board_num);
        std::vector<std::pair<int, Stockfish::Move>> legal_moves(int board_num);
        std::vector<std::pair<int, Stockfish::Move>> legal_moves(Stockfish::Color side);

        /**
         * @brief Adds a piece to the player's hand.
         * @param board_num The index of the board.
         * @param p The piece to add.
         */
        void add_to_hand(int board_num, Stockfish::Piece p) {
            pos[board_num]->add_to_hand(p);
        }

        /**
         * @brief Removes a piece from the player's hand.
         * @param board_num The index of the board.
         * @param p The piece to remove.
         */
        void remove_from_hand(int board_num, Stockfish::Piece p) {
            pos[board_num]->remove_from_hand(p);
        }

        /**
         * @brief Returns the FEN string for the specified board.
         * @param board_num The board index.
         * @return std::string FEN representation of the board.
         */
        std::string fen(int board_num) {
            return pos[board_num]->fen(); 
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

        /**
         * @brief Converts a move to a UCI string.
         * @param board_num The board index.
         * @param move The move to convert.
         * @return std::string The UCI move string.
         */
        std::string uci_move(int board_num, Stockfish::Move move) { 
            if (move == Stockfish::MOVE_NULL) {
                return "pass";
            }
            std::ostringstream oss;
            oss << board_num + 1;

            // Get the UCI move string
            std::string move_str = Stockfish::UCI::move(*pos[board_num], move).c_str();

            // Concatenate board_num and move_str without a space
            return oss.str() + move_str;
        }

        bool is_checkmate(Stockfish::Color side);
        bool check_mate_in_one(Stockfish::Color side);
};

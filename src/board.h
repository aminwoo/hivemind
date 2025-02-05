#ifndef BOARD_H
#define BOARD_H

#include <sstream> 
#include <string>
#include <algorithm>
#include <iostream> 
#include <sstream>

#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/uci.h"

class Board {
    private:

    public: 
        std::unique_ptr<Stockfish::Position> pos[2];
        Stockfish::StateListPtr states[2];
        const std::string startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"; 

        Board();
        Board(const Board& board);
        ~Board(); 

        std::string hash_key() {
            return std::to_string(pos[0]->key()) + std::to_string(pos[1]->key()); 
        }

        void swap_boards() {
            std::swap(pos[0], pos[1]);
            std::swap(states[0], states[1]);
        }

        void set(std::string fen); 
        void push_move(int board_num, Stockfish::Move move);
        void pop_move(int board_num);
        std::vector<std::pair<int, Stockfish::Move>> legal_moves(int board_num);
        std::vector<std::pair<int, Stockfish::Move>> legal_moves(Stockfish::Color side);

        void add_to_hand(int board_num, Stockfish::Piece p) {
            pos[board_num]->add_to_hand(p);
        }

        void remove_from_hand(int board_num, Stockfish::Piece p) {
            pos[board_num]->remove_from_hand(p);
        }

        std::string fen(int board_num) {
            return pos[board_num]->fen(); 
        }

        void set_fen(int board_num, std::string fen) {
            states[board_num] = Stockfish::StateListPtr(new std::deque<Stockfish::StateInfo>(1));
            states[board_num]->emplace_back();
            pos[board_num]->set(Stockfish::variants.find("bughouse")->second, fen, false, &states[board_num]->back(), Stockfish::Threads.main());
        }

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
        
        bool is_capture(int board_num, Stockfish::Move move) {
            return pos[board_num]->capture(move);
        }

        char get_captured_piece(int board_num, Stockfish::Move move) {
            Stockfish::Color us = side_to_move(board_num);
            Stockfish::Color them = ~us;
            Stockfish::Square to = to_sq(move);
            Stockfish::Piece captured = type_of(move) == Stockfish::EN_PASSANT ? make_piece(them, Stockfish::PAWN) : pos[board_num]->piece_on(to);
            
            return pos[board_num]->piece_to_char()[captured];
        }
        
        int count_in_hand(int board_num, Stockfish::Color c, Stockfish::PieceType pt) {
            return pos[board_num]->count_in_hand(c, pt);
        }

        Stockfish::Color side_to_move(int board_num) {
            return pos[board_num]->side_to_move();
        }

        Stockfish::Bitboard pieces(int board_num, Stockfish::Color c, Stockfish::PieceType pt) {
            return pos[board_num]->pieces(c, pt); 
        }

        Stockfish::Bitboard pieces(int board_num, Stockfish::Color c) {
            return pos[board_num]->pieces(c); 
        }

        Stockfish::Square ep_square(int board_num) {
            return pos[board_num]->ep_square();
        }

        Stockfish::Bitboard promoted_pieces(int board_num) { 
            return pos[board_num]->promotedPieces; 
        }

        int game_ply(int board_num) { 
            return pos[board_num]->game_ply(); 
        }

        bool can_castle(int board_num, Stockfish::CastlingRights cr) { 
            return pos[board_num]->can_castle(cr); 
        }

        int rule50_count(int board_num) { 
            return pos[board_num]->rule50_count(); 
        }

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

#endif

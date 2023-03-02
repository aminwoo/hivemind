#ifndef BUGBOARD_H
#define BUGBOARD_H

#include <sstream> 
#include <string>
#include <algorithm>
#include <iostream> 

#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/uci.h"

#include "clock.h"

class Bugboard {
    private:
        std::unique_ptr<Stockfish::Position> pos[2];
        Stockfish::StateListPtr states[2];
        Clock clock; 
        const std::string startingFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"; 
        int move_time = 1; 

    public: 
        Bugboard();
        Bugboard(const Bugboard& board);
        ~Bugboard(); 

        std::string hash_key() {
            return std::to_string(pos[0]->key()) + std::to_string(pos[1]->key()); 
        }

        void swap_boards() {
            std::swap(pos[0], pos[1]);
            std::swap(states[0], states[1]);
            clock.swap(); 
        }

        Clock get_clock() {
            return clock; 
        }

        void set_move_time(int value) {
            move_time = std::max(value, 1); 
        }

        void set_time(int idx, int time_left) {
            switch (idx) {
                case 0:
                    clock.set_time(0, Stockfish::WHITE, time_left); 
                    break; 
                case 1:
                    clock.set_time(0, Stockfish::BLACK, time_left); 
                    break; 
                case 2:
                    clock.set_time(1, Stockfish::WHITE, time_left); 
                    break; 
                case 3:
                    clock.set_time(1, Stockfish::BLACK, time_left); 
                    break; 
            }
        } 

        void set(std::string fen); 
        void do_move(int board_num, Stockfish::Move move);
        void undo_move(int board_num);
        std::vector<Stockfish::Move> legal_moves(int board_num);
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
            return Stockfish::UCI::move(*pos[board_num], move).c_str(); 
        }
};

#endif
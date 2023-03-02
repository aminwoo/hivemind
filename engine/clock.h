#ifndef CLOCK_H
#define CLOCK_H

#include "Fairy-Stockfish/src/types.h"

class Clock {
    private:
        int white_A = 1800;
        int black_A = 1800; 
        int white_B = 1800;
        int black_B = 1800;  
    public: 
        Clock();
        Clock(int white_A, int black_A, int white_B, int black_B); 
        Clock(const Clock& clock);
        int get_time(int board_num, Stockfish::Color color); 
        void set_time(int board_num, Stockfish::Color color, int value); 
        void add_time(int board_num, Stockfish::Color color, int value); 
        void remove_time(int board_num, Stockfish::Color color, int value); 
        void swap(); 
};

#endif
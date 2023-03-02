#include "clock.h"

Clock::Clock() {

} 

Clock::Clock(int white_A, int black_A, int white_B, int black_B) {
    this->white_A = white_A; 
    this->black_A = black_A; 
    this->white_B = white_B; 
    this->black_B = black_B; 
} 

Clock::Clock(const Clock& clock) {
    set_time(0, Stockfish::WHITE, clock.white_A); 
    set_time(1, Stockfish::WHITE, clock.white_B); 
    set_time(0, Stockfish::BLACK, clock.black_A); 
    set_time(1, Stockfish::BLACK, clock.black_B); 
}

int Clock::get_time(int board_num, Stockfish::Color color) {
    if (board_num == 0) {
        if (color == Stockfish::WHITE)
            return white_A; 
        else
            return black_A;
    }
    else {
        if (color == Stockfish::WHITE)
            return white_B; 
        else
            return black_B;
    }
}

void Clock::set_time(int board_num, Stockfish::Color color, int value) {
    if (board_num == 0) {
        if (color == Stockfish::WHITE)
            white_A = value; 
        else
            black_A = value; 
    }
    else {
        if (color == Stockfish::WHITE)
            white_B = value; 
        else
            black_B = value; 
    }
}

void Clock::remove_time(int board_num, Stockfish::Color color, int value) {
    if (board_num == 0) {
        if (color == Stockfish::WHITE)
            white_A -= value; 
        else
            black_A -= value; 
    }
    else {
        if (color == Stockfish::WHITE)
            white_B -= value; 
        else
            black_B -= value; 
    }
}

void Clock::add_time(int board_num, Stockfish::Color color, int value) {
    if (board_num == 0) {
        if (color == Stockfish::WHITE)
            white_A += value; 
        else
            black_A += value; 
    }
    else {
        if (color == Stockfish::WHITE)
            white_B += value; 
        else
            black_B += value; 
    }
}

void Clock::swap() {
    std::swap(white_A, white_B); 
    std::swap(black_A, black_B); 
}
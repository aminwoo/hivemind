/*
  Stand-in for the NNUE accumulator.

  StateInfo embeds one of these, so the real 2176-byte accumulator would ride on
  every search node's stack frame and on every state a Board keeps, for a
  network this build never loads: Eval::useNNUE is only set by NNUE::init()
  finding an eval file, and the evaluator itself is stubbed out. Keeping the
  member - rather than editing StateInfo - leaves position.cpp's NNUE branches
  compiling untouched against upstream.

  Restoring real NNUE means restoring nnue/ from before the engine was
  minimized, this header included.
*/

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include <cstddef>

#include "../types.h"

namespace Stockfish::Eval::NNUE {

  constexpr std::size_t CacheLineSize = 64;

  struct Accumulator {
    bool computed[COLOR_NB];
  };

}  // namespace Stockfish::Eval::NNUE

#endif // NNUE_ACCUMULATOR_H_INCLUDED

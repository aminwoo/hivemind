/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2022 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "position.h"
#include "thread_win32_osx.h"
#include "stubs.h"

namespace Stockfish {

class Thread {
  std::mutex mutex;
  std::condition_variable cv;
  size_t idx;
  bool exit = false, searching = true;
  NativeThread stdThread;

public:
  explicit Thread(size_t);
  virtual ~Thread();
  virtual void search();
  void clear();
  void idle_loop();
  void start_searching();
  void wait_for_search_finished();
  size_t id() const { return idx; }

  Pawns::Table pawnsTable;
  Material::Table materialTable;
  std::atomic<uint64_t> nodes{0}, tbHits{0}, bestMoveChanges{0};

  Position rootPos;
  StateInfo rootState;
  Search::RootMoves rootMoves;
  Depth rootDepth{0}, completedDepth{0};
};

struct MainThread : public Thread {
  using Thread::Thread;
  void search() override;
};

struct ThreadPool : public std::vector<Thread*> {
  void clear();
  void set(size_t);

  MainThread* main() const { return static_cast<MainThread*>(front()); }
  uint64_t nodes_searched() const { return accumulate(&Thread::nodes); }

private:
  uint64_t accumulate(std::atomic<uint64_t> Thread::* member) const {
    uint64_t sum = 0;
    for (Thread* th : *this)
        sum += (th->*member).load(std::memory_order_relaxed);
    return sum;
  }
};

extern ThreadPool Threads;

} // namespace Stockfish

#endif // #ifndef THREAD_H_INCLUDED

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

#include <cassert>

#include <algorithm> // For std::count
#include "movegen.h"
#include "thread.h"
#include "stubs.h"

namespace Stockfish {

ThreadPool Threads; // Global object

/// Thread::search() is the main search function
/// Stubbed for minimal move generation - we don't use the search

void Thread::search() {
  // Empty stub - search not implemented for minimal move generation
}

/// MainThread::search() is the main thread search function  
/// Stubbed for minimal move generation

void MainThread::search() {
  // Empty stub - search not implemented for minimal move generation
}

/// Thread constructor launches the thread and waits until it goes to sleep
/// in idle_loop(). Note that 'searching' and 'exit' should be already set.

Thread::Thread(size_t n) : idx(n), stdThread(&Thread::idle_loop, this) {

  wait_for_search_finished();
}


/// Thread destructor wakes up the thread in idle_loop() and waits
/// for its termination. Thread should be already waiting.

Thread::~Thread() {

  assert(!searching);

  exit = true;
  start_searching();
  stdThread.join();
}


/// Thread::clear() reset histories, usually before a new game
/// Simplified for minimal move generation

void Thread::clear() {
  // History tables removed for minimal move generation
  // counterMoves.fill(MOVE_NONE);
  // mainHistory.fill(0);
  // lowPlyHistory.fill(0);
  // captureHistory.fill(0);
  
  // Continuation history removed
}


/// Thread::start_searching() wakes up the thread that will start the search

void Thread::start_searching() {

  std::lock_guard<std::mutex> lk(mutex);
  searching = true;
  cv.notify_one(); // Wake up the thread in idle_loop()
}


/// Thread::wait_for_search_finished() blocks on the condition variable
/// until the thread has finished searching.

void Thread::wait_for_search_finished() {

  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [&]{ return !searching; });
}


/// Thread::idle_loop() is where the thread is parked, blocked on the
/// condition variable, when it has no work to do.
/// Simplified for minimal move generation

void Thread::idle_loop() {

  // Windows NUMA binding removed for minimal move generation
  // if (Options["Threads"] > 8)
  //     WinProcGroup::bindThisThread(idx);

  while (true)
  {
      std::unique_lock<std::mutex> lk(mutex);
      searching = false;
      cv.notify_one(); // Wake up anyone waiting for search finished
      // XBoard ponder search removed for minimal move generation
      cv.wait(lk, [&]{ return searching; });

      if (exit)
          return;

      lk.unlock();

      search();
  }
}

/// ThreadPool::set() creates/destroys threads to match the requested number.
/// Created and launched threads will immediately go to sleep in idle_loop.
/// Upon resizing, threads are recreated to allow for binding if necessary.
/// Simplified for minimal move generation

void ThreadPool::set(size_t requested) {

  if (size() > 0)   // destroy any existing thread(s)
  {
      main()->wait_for_search_finished();

      while (size() > 0)
          delete back(), pop_back();
  }

  if (requested > 0)   // create new thread(s)
  {
      push_back(new MainThread(0));

      while (size() < requested)
          push_back(new Thread(size()));
      clear();

      // Hash table and search init removed for minimal move generation
      // TT.resize(size_t(Options["Hash"]));
      // Search::init();
  }
}


/// ThreadPool::clear() sets threadPool data to initial values

void ThreadPool::clear() {

  for (Thread* th : *this)
      th->clear();

  main()->callsCnt = 0;
  main()->bestPreviousScore = VALUE_INFINITE;
  main()->previousTimeReduction = 1.0;
}


/// ThreadPool::start_thinking() wakes up main thread waiting in idle_loop() and
/// returns immediately. Main thread will wake up other threads and start the search.
/// Simplified for minimal move generation - we don't actually use this for search

void ThreadPool::start_thinking(Position& pos, StateListPtr& states,
                                const Search::LimitsType& limits, bool ponderMode) {

  main()->wait_for_search_finished();

  main()->stopOnPonderhit = stop = abort = false;
  increaseDepth = true;
  main()->ponder = ponderMode;
  
  // Search functionality removed for minimal move generation
  // We only keep the setup code if needed

  assert(states.get() || setupStates.get());

  if (states.get())
      setupStates = std::move(states); // Ownership transfer

  for (Thread* th : *this)
  {
      th->nodes = th->tbHits = th->nmpMinPly = th->bestMoveChanges = 0;
      th->rootDepth = th->completedDepth = 0;
      th->rootPos.set(pos.variant(), pos.fen(), pos.is_chess960(), &th->rootState, th);
      th->rootState = setupStates->back();
  }

  main()->start_searching();
}

/// Find best thread - simplified for minimal move generation
/// Just returns the main thread since we don't do real search

Thread* ThreadPool::get_best_thread() const {

    Thread* bestThread = front();
    // Voting and score comparison removed for minimal move generation
    return bestThread;
}


/// Start non-main threads

void ThreadPool::start_searching() {

    for (Thread* th : *this)
        if (th != front())
            th->start_searching();
}


/// Wait for non-main threads

void ThreadPool::wait_for_search_finished() const {

    for (Thread* th : *this)
        if (th != front())
            th->wait_for_search_finished();
}

} // namespace Stockfish

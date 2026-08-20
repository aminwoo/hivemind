#include <cassert>
#include "movegen.h"
#include "thread.h"
#include "stubs.h"

namespace Stockfish {

ThreadPool Threads; // Global object

void Thread::search() {}

void MainThread::search() {}

Thread::Thread(size_t n) : idx(n), stdThread(&Thread::idle_loop, this) {
  wait_for_search_finished();
}

Thread::~Thread() {
  assert(!searching);
  exit = true;
  start_searching();
  stdThread.join();
}

void Thread::clear() {}

void Thread::start_searching() {
  std::lock_guard<std::mutex> lk(mutex);
  searching = true;
  cv.notify_one();
}

void Thread::wait_for_search_finished() {
  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [&]{ return !searching; });
}

void Thread::idle_loop() {
  while (true)
  {
      std::unique_lock<std::mutex> lk(mutex);
      searching = false;
      cv.notify_one();
      cv.wait(lk, [&]{ return searching; });

      if (exit)
          return;

      lk.unlock();

      search();
  }
}

void ThreadPool::set(size_t requested) {
  if (size() > 0)
  {
      main()->wait_for_search_finished();

      while (size() > 0)
          delete back(), pop_back();
  }

  if (requested > 0)
  {
      push_back(new MainThread(0));

      while (size() < requested)
          push_back(new Thread(size()));
      clear();
  }
}

void ThreadPool::clear() {
  for (Thread* th : *this)
      th->clear();
}

} // namespace Stockfish

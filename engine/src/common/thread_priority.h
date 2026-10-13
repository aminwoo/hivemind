#pragma once

#if defined(__linux__) && !defined(__EMSCRIPTEN__)
#include <algorithm>
#include <cerrno>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

/**
 * How much lower than the tree's workers the mate searches that run beside
 * them are scheduled, as a nice increment.
 */
constexpr int MATE_SEARCH_THREAD_NICE = 10;

/**
 * Lowers the calling thread's scheduling priority by MATE_SEARCH_THREAD_NICE.
 *
 * The root scans and the Fairy-Stockfish probe run for the whole search,
 * beside the tree's workers. At a lower priority they take the cores the
 * tree leaves idle rather than the tree's own. Linux keeps a nice value per
 * thread, and a thread starts with the value of the thread that created it;
 * elsewhere this does nothing.
 */
inline void lower_mate_search_thread_priority() {
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
    const int self = static_cast<int>(syscall(SYS_gettid));
    errno = 0;
    const int current = getpriority(PRIO_PROCESS, self);
    if (errno == 0) {
        setpriority(PRIO_PROCESS, self,
                    std::min(19, current + MATE_SEARCH_THREAD_NICE));
    }
#endif
}

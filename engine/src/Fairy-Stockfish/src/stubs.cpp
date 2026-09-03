/*
  Stand-ins for the Fairy-Stockfish subsystems this build leaves out.

  hivemind links Fairy-Stockfish for bughouse move generation, evaluation and a
  bounded mate search. Four of the upstream subsystems cannot be reached from
  that: NNUE never loads a network (Eval::useNNUE stays false because nothing
  calls Eval::NNUE::init with a file), Syzygy never has a path so probing stays
  behind a zero cardinality, and the xboard state machine and its bughouse
  partner protocol only run under `Protocol xboard`, which hivemind never sets.

  Their call sites are all guarded, so the definitions below exist to satisfy
  the linker rather than to run. Anything that does reach one is a bug in the
  guard, so they fail loudly instead of returning a plausible value.
*/

#include <cassert>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

#include "evaluate.h"
#include "partner.h"
#include "position.h"
#include "types.h"
#include "search.h"
#include "xboard.h"
#include "syzygy/tbprobe.h"

namespace Stockfish {

namespace Eval::NNUE {

// init() and verify() live in evaluate.cpp, which is compiled: they look for a
// network file, find none, and leave useNNUE false. Only what a loaded network
// would have provided is stubbed here.
Value evaluate(const Position&, bool) {
    assert(false && "NNUE evaluation is not built in");
    return VALUE_ZERO;
}

std::string trace(Position&) { return {}; }

bool load_eval(std::string, std::istream&) { return false; }
bool save_eval(std::ostream&) { return false; }
bool save_eval(const std::optional<std::string>&) { return false; }

}  // namespace Eval::NNUE

namespace Tablebases {

// Zero cardinality is what keeps every probe site in the search from running,
// so init() has nothing to set up and the probes below are unreachable.
int MaxCardinality = 0;

void init(const std::string&) {}

WDLScore probe_wdl(Position&, ProbeState* result) {
    if (result) {
        *result = FAIL;
    }
    return WDLDraw;
}

int probe_dtz(Position&, ProbeState* result) {
    if (result) {
        *result = FAIL;
    }
    return 0;
}

bool root_probe(Position&, Search::RootMoves&) { return false; }
bool root_probe_wdl(Position&, Search::RootMoves&) { return false; }

}  // namespace Tablebases

namespace XBoard {

StateMachine* stateMachine = nullptr;

void StateMachine::ponder() { assert(false && "xboard is not built in"); }
void StateMachine::do_move(Move) { assert(false && "xboard is not built in"); }
void StateMachine::process_command(std::string, std::istringstream&) {
    assert(false && "xboard is not built in");
}

}  // namespace XBoard

// UCI::loop() offers Stockfish's own `bench` command. hivemind drives its own
// UCI loop and never calls it, so the benchmark position list is not built in.
std::vector<std::string> setup_bench(const Position&, std::istream&) { return {}; }

PartnerHandler Partner;

template<PartnerType> void PartnerHandler::ptell(const std::string&) {}
template void PartnerHandler::ptell<HUMAN>(const std::string&);
template void PartnerHandler::ptell<FAIRY>(const std::string&);
template void PartnerHandler::ptell<ALL_PARTNERS>(const std::string&);

}  // namespace Stockfish

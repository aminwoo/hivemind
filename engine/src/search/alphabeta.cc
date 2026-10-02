#include "search/alphabeta.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include "environment/constants.h"
#include "environment/joint_action.h"
#include "search/agent.h"
#include "Fairy-Stockfish/src/movegen.h"
#include "Fairy-Stockfish/src/position.h"

namespace ab {

namespace {

constexpr int MAX_BOARD_OPTIONS = 512;
constexpr int HISTORY_SLOTS = 72;  // 64 from-squares, drops by type, a pass
constexpr int PASS_SLOT = 71;
constexpr int HISTORY_MAX = 1 << 14;

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

// Ordering tiers. A move keeps only the best tier it qualifies for, plus a
// within-tier term (MVV-LVA for captures, history for quiet moves).
constexpr int ORDER_TT = 4'000'000;
constexpr int ORDER_CAPTURE = 2'000'000;
constexpr int ORDER_PROMOTION = 1'900'000;
constexpr int ORDER_KILLER = 1'500'000;
constexpr int ORDER_CHECK = 1'000'000;

int piece_value(Stockfish::PieceType type) {
    switch (type) {
        case Stockfish::PAWN: return 1;
        case Stockfish::KNIGHT: return 3;
        case Stockfish::BISHOP: return 3;
        case Stockfish::ROOK: return 5;
        case Stockfish::QUEEN: return 9;
        default: return 0;
    }
}

bool is_unplayable_promotion(Stockfish::Move move) {
    if (Stockfish::type_of(move) != Stockfish::PROMOTION) {
        return false;
    }
    const Stockfish::PieceType promoted = Stockfish::promotion_type(move);
    return promoted != Stockfish::QUEEN && promoted != Stockfish::KNIGHT;
}

int history_index(int boardNumber, Stockfish::Color color, Stockfish::Move move) {
    int from = PASS_SLOT;
    int to = 0;
    if (move != Stockfish::MOVE_NONE) {
        to = static_cast<int>(Stockfish::to_sq(move)) & 63;
        from = Stockfish::type_of(move) == Stockfish::DROP
            ? 64 + std::min(6, static_cast<int>(Stockfish::dropped_piece_type(move)))
            : static_cast<int>(Stockfish::from_sq(move)) & 63;
    }
    return ((boardNumber * 2 + static_cast<int>(color)) * HISTORY_SLOTS + from) * 64 + to;
}

void update_history(int32_t& entry, int bonus) {
    entry += bonus - entry * std::abs(bonus) / HISTORY_MAX;
}

int score_to_tt(int score, int ply) {
    if (score >= SCORE_MATE_BOUND) return score + ply;
    if (score <= -SCORE_MATE_BOUND) return score - ply;
    return score;
}

int score_from_tt(int score, int ply) {
    if (score >= SCORE_MATE_BOUND) return score - ply;
    if (score <= -SCORE_MATE_BOUND) return score + ply;
    return score;
}

/// Late-move reduction for a pair of combined rank `index`.
int reduction(int depth, int index, double divisor) {
    if (depth < 3 || index < 3) {
        return 0;
    }
    return static_cast<int>(0.5 + std::log(static_cast<double>(depth))
                                  * std::log(static_cast<double>(index)) / divisor);
}

}  // namespace

bool Options::set(const std::string& assignment) {
    const size_t equals = assignment.find('=');
    if (equals == std::string::npos) {
        return false;
    }
    const std::string name = assignment.substr(0, equals);
    const std::string value = assignment.substr(equals + 1);
    const bool flag = value == "1" || value == "true" || value == "on";
    if (name == "check_extension") checkExtension = flag;
    else if (name == "qsearch_checks") quietChecksInQsearch = flag;
    else if (name == "lmp_base") lmpBase = std::stoi(value);
    else if (name == "lmp_scale") lmpScale = std::stoi(value);
    else if (name == "pv_lmp_scale") pvLmpScale = std::stoi(value);
    else if (name == "root_lmp_scale") rootLmpScale = std::stoi(value);
    else if (name == "lmr_divisor") lmrDivisor = std::stod(value);
    else if (name == "qsearch_plies") qsearchPlies = std::stoi(value);
    else if (name == "rfp_margin") rfpMargin = std::stoi(value);
    else if (name == "futility_margin") futilityMargin = std::stoi(value);
    else if (name == "pair_ordering") pairOrdering = flag;
    else if (name == "threads") threads = std::max(1, std::stoi(value));
    else if (name == "mate_probe") rootMateProbe = flag;
    else return false;
    return true;
}

/**
 * Transposition table shared by all search threads. Each entry is three
 * relaxed atomic words and stores key ^ moves ^ meta as its check word, so an
 * entry torn by concurrent writers fails verification instead of being
 * trusted (the lockless scheme of Hyatt and Mann).
 */
struct Searcher::Table {
    struct Entry {
        std::atomic<uint64_t> check{0};
        std::atomic<uint64_t> moves{0};  // moveA | moveB << 32
        std::atomic<uint64_t> meta{0};   // score | depth << 16 | bound << 24 | generation << 32
    };

    explicit Table(size_t hashMb) {
        const size_t wanted = std::max<size_t>(1, hashMb) * 1024 * 1024 / sizeof(Entry);
        size_t power = 1;
        while (power * 2 <= wanted) {
            power *= 2;
        }
        entries.reset(new Entry[power]);
        mask = power - 1;
    }

    void clear() {
        for (uint64_t i = 0; i <= mask; ++i) {
            entries[i].check.store(0, std::memory_order_relaxed);
            entries[i].moves.store(0, std::memory_order_relaxed);
            entries[i].meta.store(0, std::memory_order_relaxed);
        }
    }

    std::unique_ptr<Entry[]> entries;
    uint64_t mask = 0;
    std::atomic<uint8_t> generation{0};
};

namespace {

uint64_t pack_meta(int score, int depth, int bound, uint8_t generation) {
    return static_cast<uint64_t>(static_cast<uint16_t>(static_cast<int16_t>(score)))
        | static_cast<uint64_t>(static_cast<uint8_t>(static_cast<int8_t>(std::clamp(depth, -1, 127)))) << 16
        | static_cast<uint64_t>(bound) << 24
        | static_cast<uint64_t>(generation) << 32;
}

}  // namespace

struct Searcher::BoardOptions {
    std::array<Stockfish::Move, MAX_BOARD_OPTIONS> moves;
    std::array<int, MAX_BOARD_OPTIONS> scores;
    std::array<uint8_t, MAX_BOARD_OPTIONS> capture;
    std::array<uint8_t, MAX_BOARD_OPTIONS> quiet;  // eligible for pruning and reductions
    int count = 0;
    bool canMove = false;  // on turn with at least one legal move
};

struct Searcher::PairCandidate {
    uint32_t rank;
    uint16_t iA;
    uint16_t iB;
};

struct Searcher::PlyOptions {
    std::array<BoardOptions, 2> boards;
    std::vector<PairCandidate> pairs;
};

Searcher::Searcher(const nnue::Network& network, size_t hashMb)
    : Searcher(network, std::make_shared<Table>(hashMb)) {}

Searcher::Searcher(const nnue::Network& network, std::shared_ptr<Table> table)
    : network_(network),
      table_(std::move(table)),
      accumulators_(new nnue::Accumulator[MAX_PLY + 2]),
      options_(new PlyOptions[MAX_PLY + 1]),
      history_(2 * 2 * HISTORY_SLOTS * 64, 0) {}

Searcher::~Searcher() = default;

void Searcher::resize(size_t hashMb) {
    table_ = std::make_shared<Table>(hashMb);
    helpers_.clear();
}

void Searcher::clear() {
    table_->clear();
    for (Searcher* searcher : all_searchers()) {
        std::fill(searcher->history_.begin(), searcher->history_.end(), 0);
        for (auto& perPly : searcher->killers_) {
            for (auto& perBoard : perPly) {
                perBoard.fill(Stockfish::MOVE_NONE);
            }
        }
    }
}

std::vector<Searcher*> Searcher::all_searchers() {
    std::vector<Searcher*> searchers = {this};
    for (auto& helper : helpers_) {
        searchers.push_back(helper.get());
    }
    return searchers;
}

Searcher::TTData Searcher::probe(uint64_t key) const {
    const Table::Entry& entry = table_->entries[key & table_->mask];
    const uint64_t moves = entry.moves.load(std::memory_order_relaxed);
    const uint64_t meta = entry.meta.load(std::memory_order_relaxed);
    const uint64_t check = entry.check.load(std::memory_order_relaxed);
    TTData data;
    const int bound = static_cast<int>((meta >> 24) & 0xff);
    if ((check ^ moves ^ meta) != key || bound == BOUND_NONE) {
        return data;
    }
    data.hit = true;
    data.move = {static_cast<Stockfish::Move>(static_cast<int32_t>(moves & 0xffffffffULL)),
                 static_cast<Stockfish::Move>(static_cast<int32_t>(moves >> 32))};
    data.score = static_cast<int16_t>(meta & 0xffff);
    data.depth = static_cast<int8_t>((meta >> 16) & 0xff);
    data.bound = bound;
    return data;
}

void Searcher::store(uint64_t key, int depth, int score, int bound,
                     const JointMove& move, int ply) {
    Table::Entry& entry = table_->entries[key & table_->mask];
    const uint64_t oldMoves = entry.moves.load(std::memory_order_relaxed);
    const uint64_t oldMeta = entry.meta.load(std::memory_order_relaxed);
    const uint64_t oldCheck = entry.check.load(std::memory_order_relaxed);
    const bool sameKey = (oldCheck ^ oldMoves ^ oldMeta) == key;
    const uint8_t generation = table_->generation.load(std::memory_order_relaxed);
    const uint8_t oldGeneration = static_cast<uint8_t>(oldMeta >> 32);
    const int oldDepth = static_cast<int8_t>((oldMeta >> 16) & 0xff);
    // Keep a deeper entry of the same search unless this one is exact.
    if (!sameKey && oldGeneration == generation
        && depth + (bound == BOUND_EXACT ? 2 : 0) < oldDepth) {
        return;
    }
    const uint64_t moves = sameKey && move == JointMove{}
        ? oldMoves
        : (static_cast<uint64_t>(static_cast<uint32_t>(move.a))
           | static_cast<uint64_t>(static_cast<uint32_t>(move.b)) << 32);
    const uint64_t meta = pack_meta(score_to_tt(score, ply), depth, bound, generation);
    entry.moves.store(moves, std::memory_order_relaxed);
    entry.meta.store(meta, std::memory_order_relaxed);
    entry.check.store(key ^ moves ^ meta, std::memory_order_relaxed);
}

bool Searcher::team_advantage(Stockfish::Color team) const {
    return team == rootTeam_ ? rootAdvantage_ : !rootAdvantage_;
}

uint64_t Searcher::key(Stockfish::Color team) const {
    uint64_t k = board_->pos[BOARD_A]->key();
    k ^= (board_->pos[BOARD_B]->key() + 0x9e3779b97f4a7c15ULL + (k << 6) + (k >> 2));
    k ^= board_->repetitionFingerprint[0] * 0xbf58476d1ce4e5b9ULL;
    k ^= board_->repetitionFingerprint[1] * 0x94d049bb133111ebULL;
    if (team == Stockfish::BLACK) {
        k ^= 0x5445414d5445414dULL;
    }
    if (team_advantage(team)) {
        k ^= 0x7a1e5c0de0ddba11ULL;
    }
    return k;
}

void Searcher::ensure_accumulator(int ply) {
    if (accumulatorReady_[ply]) {
        return;
    }
    // play() keeps every parent ready before its children exist, so a stale
    // accumulator always has one to update from.
    network_.update(*board_, whiteAdvantage_, accumulators_[ply - 1], accumulators_[ply]);
    accumulatorReady_[ply] = true;
}

int Searcher::evaluate(int ply, Stockfish::Color team) {
    ensure_accumulator(ply);
    const float logit = network_.forward(accumulators_[ply], team);
    return std::clamp(static_cast<int>(std::lround(logit * EVAL_SCALE)),
                      -SCORE_MATE_BOUND + 1, SCORE_MATE_BOUND - 1);
}

int Searcher::static_eval(Board& board, Stockfish::Color team, bool teamHasTimeAdvantage) const {
    const float logit = network_.evaluate(board, team, teamHasTimeAdvantage);
    return std::clamp(static_cast<int>(std::lround(logit * EVAL_SCALE)),
                      -SCORE_MATE_BOUND + 1, SCORE_MATE_BOUND - 1);
}

bool Searcher::in_check(Stockfish::Color team) {
    return (board_->side_to_move(BOARD_A) == team && board_->pos[BOARD_A]->checkers())
        || (board_->side_to_move(BOARD_B) == ~team && board_->pos[BOARD_B]->checkers());
}

int Searcher::terminal_score(int ply, Stockfish::Color team, bool& terminal) {
    terminal = true;
    const bool advantage = team_advantage(team);
    Board::LegalMoveCache cache;
    // Same order as classify_terminal_position: the other team being mated
    // wins before our own mate is considered.
    if (board_->is_checkmate(~team, !advantage, &cache)) {
        return SCORE_MATE - ply;
    }
    if (board_->is_checkmate(team, advantage, &cache)) {
        return -(SCORE_MATE - ply);
    }
    if (board_->is_draw(boardPlies_)) {
        return 0;
    }
    terminal = false;
    return 0;
}

void Searcher::generate(int boardNumber, bool onTurn, int ply, bool hasTTMove,
                        Stockfish::Move ttMove, BoardOptions& options) {
    options.count = 0;
    Stockfish::Position& position = *board_->pos[boardNumber];
    const Stockfish::Color color = position.side_to_move();
    if (onTurn) {
        for (const Stockfish::ExtMove& extMove : Stockfish::MoveList<Stockfish::LEGAL>(position)) {
            const Stockfish::Move move = extMove;
            if (is_unplayable_promotion(move) || options.count >= MAX_BOARD_OPTIONS - 1) {
                continue;
            }
            const int index = options.count++;
            const bool isCapture = board_->is_capture(boardNumber, move);
            const bool isPromotion = Stockfish::type_of(move) == Stockfish::PROMOTION;
            int score;
            bool tactical = true;
            if (hasTTMove && move == ttMove) {
                score = ORDER_TT;
            } else if (isCapture) {
                const Stockfish::Square to = Stockfish::to_sq(move);
                const int victim = Stockfish::type_of(move) == Stockfish::EN_PASSANT
                    ? 1 : piece_value(Stockfish::type_of(position.piece_on(to)));
                const int attacker = Stockfish::type_of(move) == Stockfish::DROP
                    ? 0 : piece_value(Stockfish::type_of(position.moved_piece(move)));
                score = ORDER_CAPTURE + 16 * victim - attacker;
            } else if (isPromotion) {
                score = ORDER_PROMOTION;
            } else if (move == killers_[ply][boardNumber][0]
                       || move == killers_[ply][boardNumber][1]) {
                score = ORDER_KILLER + (move == killers_[ply][boardNumber][0] ? 1 : 0);
            } else if (position.gives_check(move)) {
                score = ORDER_CHECK + history_[history_index(boardNumber, color, move)];
            } else {
                score = history_[history_index(boardNumber, color, move)];
                tactical = false;
            }
            options.moves[index] = move;
            options.scores[index] = score;
            options.capture[index] = isCapture ? 1 : 0;
            options.quiet[index] = tactical ? 0 : 1;
        }
    }
    options.canMove = options.count > 0;
    // The pass is always an option here; joint legality filters it later.
    const int passIndex = options.count++;
    options.moves[passIndex] = Stockfish::MOVE_NONE;
    options.capture[passIndex] = 0;
    options.quiet[passIndex] = options.canMove ? 1 : 0;
    options.scores[passIndex] = !options.canMove || (hasTTMove && ttMove == Stockfish::MOVE_NONE)
        ? ORDER_TT
        : history_[history_index(boardNumber, color, Stockfish::MOVE_NONE)];

    // Best first. With pieces in hand a board can have hundreds of drops, so
    // sort an index permutation rather than the arrays. Ties keep generation
    // order; std::sort with that tie-break avoids stable_sort's allocation.
    std::array<std::pair<int, uint16_t>, MAX_BOARD_OPTIONS> order;
    for (int i = 0; i < options.count; ++i) {
        order[i] = {options.scores[i], static_cast<uint16_t>(i)};
    }
    std::sort(order.begin(), order.begin() + options.count,
              [](const auto& left, const auto& right) {
                  return left.first != right.first ? left.first > right.first
                                                   : left.second < right.second;
              });
    std::array<Stockfish::Move, MAX_BOARD_OPTIONS> moves;
    std::array<uint8_t, MAX_BOARD_OPTIONS> capture;
    std::array<uint8_t, MAX_BOARD_OPTIONS> quiet;
    std::copy_n(options.moves.begin(), options.count, moves.begin());
    std::copy_n(options.capture.begin(), options.count, capture.begin());
    std::copy_n(options.quiet.begin(), options.count, quiet.begin());
    for (int i = 0; i < options.count; ++i) {
        const int from = order[i].second;
        options.moves[i] = moves[from];
        options.scores[i] = order[i].first;
        options.capture[i] = capture[from];
        options.quiet[i] = quiet[from];
    }
}

void Searcher::play(int ply, const JointMove& move) {
    // The child's accumulator is built only if it is evaluated: many children
    // end at a table cutoff, a mate or a repetition before that.
    ensure_accumulator(ply);
    accumulatorReady_[ply + 1] = false;
    if (move.a != Stockfish::MOVE_NONE) {
        board_->push_move(BOARD_A, move.a);
        ++boardPlies_[BOARD_A];
    }
    if (move.b != Stockfish::MOVE_NONE) {
        board_->push_move(BOARD_B, move.b);
        ++boardPlies_[BOARD_B];
    }
}

void Searcher::unplay(const JointMove& move) {
    if (move.b != Stockfish::MOVE_NONE) {
        board_->pop_move(BOARD_B);
        --boardPlies_[BOARD_B];
    }
    if (move.a != Stockfish::MOVE_NONE) {
        board_->pop_move(BOARD_A);
        --boardPlies_[BOARD_A];
    }
}

bool Searcher::should_stop() {
    if (stopped_) {
        return true;
    }
    if ((nodes_ & 1023) != 0) {
        return false;
    }
    if ((stopFlag_ && stopFlag_->load(std::memory_order_relaxed))
        || mateStop_.load(std::memory_order_relaxed)
        || (nodeLimit_ && nodes_ >= nodeLimit_)
        || (hasDeadline_ && std::chrono::steady_clock::now() >= deadline_)) {
        stopped_ = true;
    }
    return stopped_;
}

int Searcher::qsearch(int ply, int alpha, int beta, Stockfish::Color team) {
    ++nodes_;
    ++stats.qNodes;
    pvLength_[ply] = ply;
    selDepth_ = std::max(selDepth_, ply);
    if (should_stop()) {
        return 0;
    }
    bool terminal = false;
    const int terminalScore = terminal_score(ply, team, terminal);
    if (terminal) {
        return terminalScore;
    }
    if (ply >= MAX_PLY - 1 || ply >= qsearchPlyLimit_) {
        return evaluate(ply, team);
    }
    // A check cannot be stood on: try every evasion. Drops make check
    // sequences long, so only the first two are followed.
    if (qsearchCheckDepth_ < 2 && in_check(team)) {
        ++qsearchCheckDepth_;
        const int score = qsearch_evasions(ply, alpha, beta, team);
        --qsearchCheckDepth_;
        return score;
    }

    const int standPat = evaluate(ply, team);
    if (standPat >= beta) {
        return standPat;
    }
    alpha = std::max(alpha, standPat);
    int best = standPat;

    // Single-board captures. Passing on the other board is always legal next
    // to a capture, so every such pair is a legal joint action.
    for (int boardNumber : {BOARD_A, BOARD_B}) {
        Stockfish::Position& position = *board_->pos[boardNumber];
        const bool onTurn = boardNumber == BOARD_A
            ? position.side_to_move() == team
            : position.side_to_move() == ~team;
        if (!onTurn) {
            continue;
        }
        std::array<Stockfish::Move, 64> captures;
        std::array<int, 64> scores;
        int count = 0;
        for (const Stockfish::ExtMove& extMove
             : Stockfish::MoveList<Stockfish::CAPTURES>(position)) {
            const Stockfish::Move move = extMove;
            if (count >= 64 || is_unplayable_promotion(move)
                || !position.legal(move) || !board_->is_capture(boardNumber, move)
                || !position.see_ge(move, Stockfish::VALUE_ZERO)) {
                continue;
            }
            const Stockfish::Square to = Stockfish::to_sq(move);
            const int victim = Stockfish::type_of(move) == Stockfish::EN_PASSANT
                ? 1 : piece_value(Stockfish::type_of(position.piece_on(to)));
            const int attacker = piece_value(Stockfish::type_of(position.moved_piece(move)));
            captures[count] = move;
            scores[count] = 16 * victim - attacker;
            ++count;
        }
        // On the first turn past the horizon, also non-capturing checks that
        // do not lose material: drop attacks are bughouse's main tactic. They
        // need a legal pass on the other board, which a quiet move does not
        // earn when both boards are on turn without the time advantage.
        const bool otherOnTurn = boardNumber == BOARD_A
            ? board_->side_to_move(BOARD_B) == ~team
            : board_->side_to_move(BOARD_A) == team;
        if (options.quietChecksInQsearch && ply == qsearchEntryPly_
            && (team_advantage(team) || !otherOnTurn)) {
            for (Stockfish::Move move : board_->checking_moves(boardNumber)) {
                if (count >= 64) {
                    break;
                }
                if (board_->is_capture(boardNumber, move)
                    || !position.see_ge(move, Stockfish::VALUE_ZERO)) {
                    continue;
                }
                captures[count] = move;
                scores[count] = -100;  // after every capture
                ++count;
            }
        }
        for (int i = 0; i < count; ++i) {
            int bestIndex = i;
            for (int j = i + 1; j < count; ++j) {
                if (scores[j] > scores[bestIndex]) {
                    bestIndex = j;
                }
            }
            std::swap(captures[i], captures[bestIndex]);
            std::swap(scores[i], scores[bestIndex]);

            ++stats.qCaptures;
            JointMove move;
            (boardNumber == BOARD_A ? move.a : move.b) = captures[i];
            play(ply, move);
            const int score = -qsearch(ply + 1, -beta, -alpha, ~team);
            unplay(move);
            if (stopped_) {
                return 0;
            }
            if (score > best) {
                best = score;
                if (score > alpha) {
                    alpha = score;
                    if (alpha >= beta) {
                        return best;
                    }
                }
            }
        }
    }
    return best;
}

int Searcher::qsearch_evasions(int ply, int alpha, int beta, Stockfish::Color team) {
    ++stats.evasionNodes;
    const bool advantage = team_advantage(team);
    const bool boardAOnTurn = board_->side_to_move(BOARD_A) == team;
    const bool boardBOnTurn = board_->side_to_move(BOARD_B) == ~team;
    const bool checkedA = boardAOnTurn && board_->pos[BOARD_A]->checkers();
    const bool checkedB = boardBOnTurn && board_->pos[BOARD_B]->checkers();
    BoardOptions& optionsA = options_[ply].boards[BOARD_A];
    BoardOptions& optionsB = options_[ply].boards[BOARD_B];
    generate(BOARD_A, boardAOnTurn, ply, false, Stockfish::MOVE_NONE, optionsA);
    generate(BOARD_B, boardBOnTurn, ply, false, Stockfish::MOVE_NONE, optionsB);

    // Every option on a board in check. On a board not in check the team
    // passes where that is legal, and otherwise plays one of its three
    // best-ordered moves.
    auto pass_legal = [&](bool passOnA, int partnerIndex) {
        const BoardOptions& partner = passOnA ? optionsB : optionsA;
        if (partner.moves[partnerIndex] == Stockfish::MOVE_NONE) {
            return is_double_sit_legal(advantage, boardAOnTurn, boardBOnTurn);
        }
        return is_single_pass_legal(advantage, boardAOnTurn, boardBOnTurn,
                                    partner.capture[partnerIndex] != 0);
    };
    int best = -SCORE_INF;
    for (int iA = 0; iA < optionsA.count; ++iA) {
        for (int iB = 0; iB < optionsB.count; ++iB) {
            const JointMove move{optionsA.moves[iA], optionsB.moves[iB]};
            const bool sitsA = move.a == Stockfish::MOVE_NONE;
            const bool sitsB = move.b == Stockfish::MOVE_NONE;
            if ((sitsA && optionsA.canMove && !pass_legal(true, iB))
                || (sitsB && optionsB.canMove && !pass_legal(false, iA))
                || (sitsA && sitsB && !is_double_sit_legal(advantage, boardAOnTurn, boardBOnTurn))) {
                continue;
            }
            if (!checkedA && !sitsA && (iA >= 3 || pass_legal(true, iB))) {
                continue;
            }
            if (!checkedB && !sitsB && (iB >= 3 || pass_legal(false, iA))) {
                continue;
            }
            ++stats.evasionPairs;
            play(ply, move);
            const int score = -qsearch(ply + 1, -beta, -alpha, ~team);
            unplay(move);
            if (stopped_) {
                return 0;
            }
            if (score > best) {
                best = score;
                if (score > alpha) {
                    alpha = score;
                    if (alpha >= beta) {
                        return best;
                    }
                }
            }
        }
    }
    return best;
}

int Searcher::negamax(int depth, int ply, int alpha, int beta, bool pvNode,
                      Stockfish::Color team) {
    pvLength_[ply] = ply;
    if (depth <= 0) {
        // Captures continue for at most qsearchPlies team turns past the horizon.
        const int savedLimit = qsearchPlyLimit_;
        const int savedEntry = qsearchEntryPly_;
        qsearchPlyLimit_ = std::min(savedLimit, ply + options.qsearchPlies);
        qsearchEntryPly_ = ply;
        const int score = qsearch(ply, alpha, beta, team);
        qsearchPlyLimit_ = savedLimit;
        qsearchEntryPly_ = savedEntry;
        return score;
    }
    ++nodes_;
    ++stats.mainNodes;
    selDepth_ = std::max(selDepth_, ply);
    if (should_stop()) {
        return 0;
    }
    const bool root = ply == 0;
    if (!root) {
        bool terminal = false;
        const int terminalScore = terminal_score(ply, team, terminal);
        if (terminal) {
            return terminalScore;
        }
        // Mate distance pruning.
        alpha = std::max(alpha, -(SCORE_MATE - ply));
        beta = std::min(beta, SCORE_MATE - ply - 1);
        if (alpha >= beta) {
            return alpha;
        }
    }
    if (ply >= MAX_PLY - 1) {
        return evaluate(ply, team);
    }

    const uint64_t nodeKey = key(team);
    const TTData tt = probe(nodeKey);
    const bool ttHit = tt.hit;
    JointMove ttMove;
    if (ttHit) {
        ttMove = tt.move;
        const int ttScore = score_from_tt(tt.score, ply);
        if (!pvNode && tt.depth >= depth
            && ((tt.bound == BOUND_EXACT)
                || (tt.bound == BOUND_LOWER && ttScore >= beta)
                || (tt.bound == BOUND_UPPER && ttScore <= alpha))) {
            ++stats.ttCuts;
            return ttScore;
        }
    }

    const bool checked = in_check(team);
    // Extend a turn spent in check, at most doubling the nominal depth.
    if (checked && options.checkExtension && !root && ply < 2 * rootDepth_) {
        ++depth;
    }
    const int staticEval = evaluate(ply, team);
    if (!pvNode && !checked && depth <= 4 && std::abs(beta) < SCORE_MATE_BOUND
        && staticEval - options.rfpMargin * depth >= beta) {
        ++stats.rfpCuts;
        return staticEval;
    }

    const bool advantage = team_advantage(team);
    const bool boardAOnTurn = board_->side_to_move(BOARD_A) == team;
    const bool boardBOnTurn = board_->side_to_move(BOARD_B) == ~team;
    BoardOptions& optionsA = options_[ply].boards[BOARD_A];
    BoardOptions& optionsB = options_[ply].boards[BOARD_B];
    generate(BOARD_A, boardAOnTurn, ply, ttHit, ttMove.a, optionsA);
    generate(BOARD_B, boardBOnTurn, ply, ttHit, ttMove.b, optionsB);

    JointActionRules rules;
    rules.boardAOnTurn = boardAOnTurn;
    rules.boardBOnTurn = boardBOnTurn;
    rules.teamHasTimeAdvantage = advantage;
    rules.boardACanMove = optionsA.canMove;
    rules.boardBCanMove = optionsB.canMove;
    auto legal = [&](int iA, int iB) {
        const Stockfish::Move a = optionsA.moves[iA];
        const Stockfish::Move b = optionsB.moves[iB];
        // The seat requirement binds our own move only, not the tree.
        if (root) {
            return is_joint_action_legal(rules, a, b, optionsA.capture[iA], optionsB.capture[iB]);
        }
        const bool sitsA = a == Stockfish::MOVE_NONE;
        const bool sitsB = b == Stockfish::MOVE_NONE;
        if (sitsA && sitsB) {
            return is_double_sit_legal(advantage, boardAOnTurn, boardBOnTurn);
        }
        if (sitsA && rules.boardACanMove) {
            return is_single_pass_legal(advantage, boardAOnTurn, boardBOnTurn,
                                        optionsB.capture[iB] != 0);
        }
        if (sitsB && rules.boardBCanMove) {
            return is_single_pass_legal(advantage, boardAOnTurn, boardBOnTurn,
                                        optionsA.capture[iA] != 0);
        }
        return true;
    };

    // Pairs are pruned on the product of the two boards' ranks, so the
    // number searched grows like L log L instead of with |A| x |B|. Evasions
    // on a board in check all rank first; the other board still counts.
    const bool checkedA = boardAOnTurn && board_->pos[BOARD_A]->checkers();
    const bool checkedB = boardBOnTurn && board_->pos[BOARD_B]->checkers();
    const int pairLimit = (options.lmpBase + options.lmpScale * depth * depth)
        * (root ? options.rootLmpScale : (pvNode ? options.pvLmpScale : 1));
    int best = -SCORE_INF;
    JointMove bestMove;
    int searched = 0;
    const int originalAlpha = alpha;

    // Candidate pairs within the rank limit. Ordered by rank product (then
    // rank sum) the second-best move on one board paired with the best on the
    // other comes before the best paired with a poor one, which A-major loops
    // would search first.
    std::vector<PairCandidate>& pairs = options_[ply].pairs;
    pairs.clear();
    for (int iA = 0; iA < optionsA.count; ++iA) {
        const int rankA = checkedA ? 1 : iA + 1;
        if (rankA > pairLimit) {
            break;
        }
        for (int iB = 0; iB < optionsB.count; ++iB) {
            const int rank = rankA * (checkedB ? 1 : iB + 1);
            if (rank > pairLimit) {
                break;
            }
            if (legal(iA, iB)) {
                pairs.push_back({static_cast<uint32_t>(rank),
                                 static_cast<uint16_t>(iA), static_cast<uint16_t>(iB)});
            }
        }
    }
    if (pairs.empty()) {
        // Nothing legal inside the limit (a pass that needs a capture, say):
        // the first legal pair anywhere still has to be searched.
        for (int iA = 0; iA < optionsA.count && pairs.empty(); ++iA) {
            for (int iB = 0; iB < optionsB.count; ++iB) {
                if (legal(iA, iB)) {
                    pairs.push_back({UINT32_MAX, static_cast<uint16_t>(iA),
                                     static_cast<uint16_t>(iB)});
                    break;
                }
            }
        }
    }
    const int legalCount = static_cast<int>(pairs.size());
    stats.legalPairs += pairs.size();
    if (options.pairOrdering) {
        std::stable_sort(pairs.begin(), pairs.end(),
                         [](const PairCandidate& left, const PairCandidate& right) {
                             return left.rank != right.rank
                                 ? left.rank < right.rank
                                 : left.iA + left.iB < right.iA + right.iB;
                         });
    }

    // Returns true on a beta cutoff.
    auto search_pair = [&](const PairCandidate& candidate) {
        const int iA = candidate.iA;
        const int iB = candidate.iB;
        const Stockfish::Move a = optionsA.moves[iA];
        const Stockfish::Move b = optionsB.moves[iB];
        const int rank = static_cast<int>(std::min<uint32_t>(candidate.rank, 1u << 20));
        const bool quietPair = optionsA.quiet[iA] && optionsB.quiet[iB];
        if (searched > 0 && best > -SCORE_MATE_BOUND && !pvNode && !checked && quietPair
            && depth <= 2 && staticEval + options.futilityMargin * depth <= alpha) {
            return false;
        }

        const JointMove move{a, b};
        ++stats.mainPairs;
        play(ply, move);
        int score;
        if (searched == 0) {
            score = -negamax(depth - 1, ply + 1, -beta, -alpha, pvNode, ~team);
        } else {
            int r = 0;
            if (depth >= 3) {
                r = reduction(depth, rank, options.lmrDivisor);
                r -= (pvNode ? 1 : 0) + (quietPair ? 0 : 1) + (checked ? 1 : 0);
                r = std::clamp(r, 0, depth - 2);
            }
            score = -negamax(depth - 1 - r, ply + 1, -alpha - 1, -alpha, false, ~team);
            if (score > alpha && r > 0) {
                score = -negamax(depth - 1, ply + 1, -alpha - 1, -alpha, false, ~team);
            }
            if (score > alpha && pvNode && score < beta) {
                score = -negamax(depth - 1, ply + 1, -beta, -alpha, true, ~team);
            }
        }
        unplay(move);
        ++searched;
        if (stopped_) {
            return true;
        }

        if (score > best) {
            best = score;
            bestMove = move;
            if (score > alpha) {
                alpha = score;
                if (pvNode) {
                    pv_[ply][ply] = move;
                    for (int next = ply + 1; next < pvLength_[ply + 1]; ++next) {
                        pv_[ply][next] = pv_[ply + 1][next];
                    }
                    pvLength_[ply] = std::max(pvLength_[ply + 1], ply + 1);
                }
                if (alpha >= beta) {
                    const int bonus = std::min(depth * depth * 16, 1600);
                    const Stockfish::Color colorA = board_->side_to_move(BOARD_A);
                    const Stockfish::Color colorB = board_->side_to_move(BOARD_B);
                    if (optionsA.quiet[iA] || a == Stockfish::MOVE_NONE) {
                        update_history(history_[history_index(BOARD_A, colorA, a)], bonus);
                        if (a != Stockfish::MOVE_NONE && killers_[ply][BOARD_A][0] != a) {
                            killers_[ply][BOARD_A][1] = killers_[ply][BOARD_A][0];
                            killers_[ply][BOARD_A][0] = a;
                        }
                    }
                    if (optionsB.quiet[iB] || b == Stockfish::MOVE_NONE) {
                        update_history(history_[history_index(BOARD_B, colorB, b)], bonus);
                        if (b != Stockfish::MOVE_NONE && killers_[ply][BOARD_B][0] != b) {
                            killers_[ply][BOARD_B][1] = killers_[ply][BOARD_B][0];
                            killers_[ply][BOARD_B][0] = b;
                        }
                    }
                    return true;
                }
            }
        }
        return false;
    };

    bool cutoff = false;
    for (const PairCandidate& candidate : pairs) {
        if ((cutoff = search_pair(candidate))) {
            break;
        }
    }
    // Everything inside the limit loses to mate: look for an escape among the
    // pairs the limit left out.
    if (!cutoff && !stopped_ && best <= -SCORE_MATE_BOUND) {
        for (int iA = 0; iA < optionsA.count && !cutoff; ++iA) {
            const int rankA = checkedA ? 1 : iA + 1;
            for (int iB = 0; iB < optionsB.count; ++iB) {
                const int rank = rankA * (checkedB ? 1 : iB + 1);
                if (rank <= pairLimit || !legal(iA, iB)) {
                    continue;
                }
                if ((cutoff = search_pair({static_cast<uint32_t>(rank),
                                           static_cast<uint16_t>(iA),
                                           static_cast<uint16_t>(iB)}))
                    || best > -SCORE_MATE_BOUND) {
                    break;
                }
            }
            if (best > -SCORE_MATE_BOUND) {
                break;
            }
        }
    }
    if (stopped_) {
        return 0;
    }

    if (legalCount == 0) {
        // No legal joint action: the team cannot continue.
        return -(SCORE_MATE - ply);
    }
    if (searched == 0) {
        // Everything legal was pruned; fall back to the static evaluation.
        return staticEval;
    }
    const int bound = best >= beta ? BOUND_LOWER
        : (pvNode && best > originalAlpha ? BOUND_EXACT : BOUND_UPPER);
    store(nodeKey, depth, best, bound, bestMove, ply);
    return best;
}

Result Searcher::search(Board& board, Stockfish::Color team, bool teamHasTimeAdvantage,
                        const Limits& limits, const std::atomic<bool>* stop,
                        const std::function<void(const IterationInfo&)>& onIteration) {
    table_->generation.fetch_add(1, std::memory_order_relaxed);
    const size_t helperCount = static_cast<size_t>(std::max(1, options.threads)) - 1;
    while (helpers_.size() < helperCount) {
        helpers_.push_back(std::unique_ptr<Searcher>(new Searcher(network_, table_)));
    }
    // Lazy SMP: helpers search the same root on their own board copies and
    // share only the transposition table; staggered start depths keep them
    // from moving in lockstep. The main thread's result is returned.
    std::atomic<bool> helpersStop{false};
    // Copy the board before any thread starts: the main thread makes and
    // unmakes moves on `board` as soon as it begins searching.
    std::vector<std::unique_ptr<Board>> copies;
    for (size_t i = 0; i < helperCount; ++i) {
        copies.push_back(std::make_unique<Board>(board));
    }
    std::vector<std::thread> threads;
    for (size_t i = 0; i < helperCount; ++i) {
        Searcher& helper = *helpers_[i];
        helper.options = options;
        Board& copy = *copies[i];
        threads.emplace_back([&helper, &copy, &helpersStop, team, teamHasTimeAdvantage, limits, i] {
            Limits helperLimits = limits;
            helperLimits.nodes = 0;
            helperLimits.moveTimeMs = 0;
            helper.iterate(copy, team, teamHasTimeAdvantage, helperLimits, &helpersStop, {},
                           static_cast<int>(i) + 1);
        });
    }

    // Fairy-Stockfish mate probe. It searches each board on its own for up to
    // MATE_PROBE_MAX_MATE_MOVES, which reaches mates far beyond this search's
    // horizon, and only a team ahead on time can sit one board while playing
    // a mate out on the other. A hit that survives the two-board replay and
    // the mate-race veto stops the search and replaces its move.
    mateStop_.store(false);
    std::atomic<bool> mainDone{false};
    std::atomic<bool> searchFoundMate{false};
    JointActionCandidate mateAction;
    int matePly = 0;
    bool mateFound = false;
    std::unique_ptr<Board> mateBoard;
    std::thread mateThread;
    if (options.rootMateProbe && teamHasTimeAdvantage) {
        mateBoard = std::make_unique<Board>(board);
        // Without a move time the probe gets a fixed window, and gives up as
        // soon as the search is done rather than holding up its answer.
        const int probeMs = limits.moveTimeMs > 0 ? limits.moveTimeMs : 3000;
        const bool timed = limits.moveTimeMs > 0;
        mateThread = std::thread([&, probeMs, timed] {
            Board& probeBoard = *mateBoard;
            const auto race_safe = [&] {
                return !Agent::action_loses_mate_race(
                    probeBoard, mateAction, team, teamHasTimeAdvantage);
            };
            std::string principalVariation;
            const bool found = Agent::probe_root_mate(
                probeBoard, team, teamHasTimeAdvantage,
                SearchParams::MATE_PROBE_ROOT_NODE_BUDGET, probeMs,
                [&] {
                    return (stop && stop->load(std::memory_order_relaxed))
                        || searchFoundMate.load(std::memory_order_relaxed)
                        || (!timed && mainDone.load(std::memory_order_relaxed));
                },
                mateAction, matePly, principalVariation,
                [&] {
                    if (race_safe()) {
                        mateStop_.store(true);
                    }
                },
                true);
            mateFound = found && race_safe();
        });
    }

    Result result = iterate(board, team, teamHasTimeAdvantage, limits, stop, onIteration, 0);
    // A mate the search found itself needs no probe to confirm it.
    searchFoundMate.store(result.hasMove && result.score >= SCORE_MATE_BOUND);
    mainDone.store(true);
    helpersStop.store(true);
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (mateThread.joinable()) {
        mateThread.join();
    }
    for (size_t i = 0; i < helperCount; ++i) {
        result.nodes += helpers_[i]->nodes_;
    }
    if (mateFound && !(result.hasMove && result.score >= SCORE_MATE_BOUND
                       && result.score >= SCORE_MATE - std::max(1, matePly))) {
        result.best = {mateAction.moveA, mateAction.moveB};
        result.hasMove = true;
        result.score = SCORE_MATE - std::max(1, matePly);
        result.depth = std::max(result.depth, matePly);
        result.pv = {result.best};
        if (onIteration) {
            const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_).count();
            onIteration({result.depth, selDepth_, result.score, result.nodes, elapsed, result.pv});
        }
    }
    mateStop_.store(false);
    return result;
}

Result Searcher::iterate(Board& board, Stockfish::Color team, bool teamHasTimeAdvantage,
                         const Limits& limits, const std::atomic<bool>* stop,
                         const std::function<void(const IterationInfo&)>& onIteration,
                         int helperIndex) {
    board_ = &board;
    rootTeam_ = team;
    rootAdvantage_ = teamHasTimeAdvantage;
    whiteAdvantage_ = team == Stockfish::WHITE ? teamHasTimeAdvantage : !teamHasTimeAdvantage;
    boardPlies_ = {0, 0};
    nodes_ = 0;
    nodeLimit_ = limits.nodes;
    stopped_ = false;
    stopFlag_ = stop;
    selDepth_ = 0;
    start_ = std::chrono::steady_clock::now();
    hasDeadline_ = limits.moveTimeMs > 0;
    deadline_ = start_ + std::chrono::milliseconds(limits.moveTimeMs);
    for (auto& perPly : killers_) {
        for (auto& perBoard : perPly) {
            perBoard.fill(Stockfish::MOVE_NONE);
        }
    }
    for (int32_t& entry : history_) {
        entry /= 2;
    }
    qsearchCheckDepth_ = 0;
    qsearchPlyLimit_ = MAX_PLY;
    network_.refresh(board, whiteAdvantage_, accumulators_[0]);
    accumulatorReady_[0] = true;

    Result result;
    int previousScore = 0;
    const int maxDepth = std::clamp(limits.depth, 1, MAX_PLY - 2);
    for (int depth = 1; depth <= maxDepth; ++depth) {
        // Helpers skip depths in Stockfish's classic Lazy SMP pattern, so
        // they spread over the next few depths instead of repeating the main
        // thread's work.
        if (helperIndex > 0) {
            static constexpr int SKIP_SIZE[] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
            static constexpr int SKIP_PHASE[] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
            const int i = (helperIndex - 1) % 20;
            if (((depth + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 != 0 && depth < maxDepth) {
                continue;
            }
        }
        int delta = 60;
        int alpha = depth >= 4 ? std::max(-SCORE_INF, previousScore - delta) : -SCORE_INF;
        int beta = depth >= 4 ? std::min(SCORE_INF, previousScore + delta) : SCORE_INF;
        int score = 0;
        rootDepth_ = depth;
        for (;;) {
            selDepth_ = 0;
            score = negamax(depth, 0, alpha, beta, true, team);
            if (stopped_) {
                break;
            }
            if (score <= alpha) {
                alpha = std::max(-SCORE_INF, score - delta);
            } else if (score >= beta) {
                beta = std::min(SCORE_INF, score + delta);
            } else {
                break;
            }
            delta *= 2;
            if (delta > 1000) {
                alpha = -SCORE_INF;
                beta = SCORE_INF;
            }
        }
        if (stopped_ && result.hasMove) {
            break;
        }
        if (pvLength_[0] > 0) {
            result.best = pv_[0][0];
            result.hasMove = true;
            result.pv.assign(pv_[0].begin(), pv_[0].begin() + pvLength_[0]);
        }
        if (stopped_) {
            break;
        }
        result.score = score;
        result.depth = depth;
        previousScore = score;
        if (onIteration) {
            const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_).count();
            onIteration({depth, selDepth_, score, nodes_, elapsed, result.pv});
        }
        if (std::abs(score) >= SCORE_MATE_BOUND && depth >= 2 * (SCORE_MATE - std::abs(score)) + 2) {
            break;
        }
        if (hasDeadline_) {
            const auto elapsed = std::chrono::steady_clock::now() - start_;
            if (elapsed * 2 >= deadline_ - start_) {
                break;
            }
        }
    }
    result.nodes = nodes_;
    board_ = nullptr;
    return result;
}

std::string format_joint_move(Board& board, const JointMove& move) {
    return "(" + (move.a == Stockfish::MOVE_NONE ? std::string("pass") : board.uci_move(BOARD_A, move.a))
        + "," + (move.b == Stockfish::MOVE_NONE ? std::string("pass") : board.uci_move(BOARD_B, move.b))
        + ")";
}

}  // namespace ab

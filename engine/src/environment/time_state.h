#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>

#include "Fairy-Stockfish/src/types.h"

/**
 * @brief The clock situation of a bughouse match, as the search needs it.
 *
 * Without increment each board always has exactly one clock running, so the
 * two boards' clock sums stay equal and a team's lead over its diagonal
 * opponents is one number: Aw - Bw = Bb - Ab for the white team, and the black
 * team's lead is its negation. Absolute clock values never enter.
 *
 * The lead moves only while one team is on turn on both boards. Both of that
 * team's clocks run and neither of the other team's, so the team pays its own
 * move time for the action it is deciding. In every other configuration one
 * clock of each team runs and the lead holds. A team may sit while its lead is
 * positive; at a lead of exactly zero neither team may.
 *
 * Legacy mode reproduces the TimeAdvantage flag: one team's lead is unbounded,
 * never moves, and the other team never has one.
 */
class TimeState {
public:
    /// Chess.com's fastest possible move, the default for both teams.
    static constexpr int32_t DEFAULT_MOVE_TIME_MS = 100;
    /// Bound on a clocked lead, far beyond any real clock.
    static constexpr int32_t MAX_LEAD_MS = 1 << 30;

    /** A permanent flag: @p team may always sit if it has the advantage. */
    static TimeState legacy(Stockfish::Color team, bool teamHasTimeAdvantage) {
        TimeState state;
        const bool whiteAhead = (team == Stockfish::WHITE) == teamHasTimeAdvantage;
        state.whiteLeadMs_ = whiteAhead ? MAX_LEAD_MS : -MAX_LEAD_MS;
        return state;
    }

    /**
     * @param team The team the lead and the first move time belong to.
     * @param teamLeadMs Its lead over its diagonal opponents; negative when behind.
     * @param teamMoveTimeMs How long @p team takes for one action.
     * @param opponentMoveTimeMs How long the other team takes for one action.
     * @param teamNextMoveTimeMs What @p team's next action alone costs, when it
     *        is known to differ from its usual move time (0 = the usual): a
     *        front end that has budgeted the move it is about to search.
     */
    static TimeState clocked(Stockfish::Color team,
                             int64_t teamLeadMs,
                             int32_t teamMoveTimeMs = DEFAULT_MOVE_TIME_MS,
                             int32_t opponentMoveTimeMs = DEFAULT_MOVE_TIME_MS,
                             int32_t teamNextMoveTimeMs = 0) {
        TimeState state;
        state.clocked_ = true;
        const int64_t whiteLead = team == Stockfish::WHITE ? teamLeadMs : -teamLeadMs;
        state.whiteLeadMs_ = clamp_lead(whiteLead);
        state.moveTimeMs_[team] = std::max<int32_t>(1, teamMoveTimeMs);
        state.moveTimeMs_[~team] = std::max<int32_t>(1, opponentMoveTimeMs);
        if (teamNextMoveTimeMs > 0) {
            state.nextMoveTeam_ = team;
            state.nextMoveTimeMs_ = teamNextMoveTimeMs;
        }
        return state;
    }

    bool is_clocked() const { return clocked_; }

    int32_t lead_ms(Stockfish::Color team) const {
        return team == Stockfish::WHITE ? whiteLeadMs_ : -whiteLeadMs_;
    }

    /** How long @p team usually takes for one action. */
    int32_t move_time_ms(Stockfish::Color team) const {
        return moveTimeMs_[team];
    }

    /** This state with no next-move time: every action at its team's usual time. */
    TimeState without_next_move_time() const {
        TimeState plain = *this;
        plain.nextMoveTeam_ = Stockfish::WHITE;
        plain.nextMoveTimeMs_ = 0;
        return plain;
    }

    /** What @p team's next action costs when it is charged. */
    int32_t next_move_time_ms(Stockfish::Color team) const {
        return nextMoveTimeMs_ > 0 && team == nextMoveTeam_
            ? nextMoveTimeMs_ : moveTimeMs_[team];
    }

    /** Whether @p team may sit: wait on a board instead of moving there. */
    bool can_sit(Stockfish::Color team) const {
        return lead_ms(team) > 0;
    }

    /**
     * @brief Whether @p team can pay for @p actions more actions on turn on
     * both boards and still not be behind.
     *
     * Proofs that let one team sit a board for a whole line rest on this:
     * while it holds, that team may sit before each of those actions and the
     * other team never gains the right to. A legacy state answers can_sit().
     */
    bool can_sit_through(Stockfish::Color team, int actions) const {
        if (!clocked_) {
            return can_sit(team);
        }
        return can_sit(team)
            && static_cast<int64_t>(lead_ms(team))
                   >= next_move_time_ms(team)
                       + static_cast<int64_t>(std::max(actions, 1) - 1) * moveTimeMs_[team];
    }

    /**
     * @brief Charge the action @p actingTeam is about to play.
     *
     * Call it with the position before the action. Only a team on turn on both
     * boards pays; legacy leads never move. A next-move time set for the
     * acting team is used up by this action, whether or not it pays.
     */
    void advance(Stockfish::Color actingTeam, bool actingTeamOnBothBoards) {
        const int64_t cost = next_move_time_ms(actingTeam);
        if (actingTeam == nextMoveTeam_) {
            // Back to the plain clock, owner included, so equal states compare equal.
            nextMoveTeam_ = Stockfish::WHITE;
            nextMoveTimeMs_ = 0;
        }
        if (!clocked_ || !actingTeamOnBothBoards) {
            return;
        }
        whiteLeadMs_ = clamp_lead(
            static_cast<int64_t>(whiteLeadMs_)
            + (actingTeam == Stockfish::WHITE ? -cost : cost));
    }

    /** The state after @p actingTeam plays one action; see advance(). */
    TimeState after(Stockfish::Color actingTeam, bool actingTeamOnBothBoards) const {
        TimeState next = *this;
        next.advance(actingTeam, actingTeamOnBothBoards);
        return next;
    }

    /** Exact identity of a clocked state: the lead, both move times and any next-move time. */
    uint64_t exact_key() const {
        const uint64_t lead = static_cast<uint32_t>(whiteLeadMs_);
        const uint64_t moveTimes =
            (static_cast<uint64_t>(static_cast<uint32_t>(moveTimeMs_[Stockfish::WHITE])) << 32)
            | static_cast<uint32_t>(moveTimeMs_[Stockfish::BLACK]);
        uint64_t key = (0x434c4f434b000000ULL ^ (lead * 0x9e3779b97f4a7c15ULL))
            ^ (moveTimes * 0xbf58476d1ce4e5b9ULL) ^ (clocked_ ? 1 : 0);
        if (nextMoveTimeMs_ > 0) {
            const uint64_t next = (static_cast<uint64_t>(nextMoveTeam_) << 32)
                | static_cast<uint32_t>(nextMoveTimeMs_);
            key ^= (next + 0x6e657874ULL) * 0x94d049bb133111ebULL;
        }
        return key;
    }

    /**
     * @brief Coarse lead class: which team may sit, and the size of its lead
     * on a log2 scale in its own move times.
     *
     * Nothing in the search keys on it; nodes are identified by the exact
     * clock (exact_key), since leads in one class can differ in sign a few
     * actions later. A front end can use it to judge whether a lead that has
     * drifted is worth a new search.
     */
    uint32_t lead_class(Stockfish::Color team) const {
        const int32_t lead = lead_ms(team);
        if (lead == 0) {
            return 0;
        }
        const Stockfish::Color ahead = lead > 0 ? team : ~team;
        const uint32_t magnitude = static_cast<uint32_t>(std::abs(lead));
        const uint32_t moveTime = static_cast<uint32_t>(moveTimeMs_[ahead]);
        const uint32_t moves = (magnitude + moveTime - 1) / moveTime;
        const uint32_t scale = std::min<uint32_t>(std::bit_width(moves), 31);
        return lead > 0 ? scale : 32 + scale;
    }

    bool operator==(const TimeState&) const = default;

private:
    static int32_t clamp_lead(int64_t lead) {
        return static_cast<int32_t>(std::clamp<int64_t>(lead, -MAX_LEAD_MS, MAX_LEAD_MS));
    }

    bool clocked_ = false;
    // Legacy default: white may never sit and black always may. Callers build
    // states through legacy() or clocked(); this only keeps the type regular.
    int32_t whiteLeadMs_ = -MAX_LEAD_MS;
    std::array<int32_t, 2> moveTimeMs_{DEFAULT_MOVE_TIME_MS, DEFAULT_MOVE_TIME_MS};
    // The cost of nextMoveTeam_'s next action when it differs from its usual
    // move time; 0 when it does not. Used up by that team's next action.
    Stockfish::Color nextMoveTeam_ = Stockfish::WHITE;
    int32_t nextMoveTimeMs_ = 0;
};

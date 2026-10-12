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
     */
    static TimeState clocked(Stockfish::Color team,
                             int64_t teamLeadMs,
                             int32_t teamMoveTimeMs = DEFAULT_MOVE_TIME_MS,
                             int32_t opponentMoveTimeMs = DEFAULT_MOVE_TIME_MS) {
        TimeState state;
        state.clocked_ = true;
        const int64_t whiteLead = team == Stockfish::WHITE ? teamLeadMs : -teamLeadMs;
        state.whiteLeadMs_ = clamp_lead(whiteLead);
        state.moveTimeMs_[team] = std::max<int32_t>(1, teamMoveTimeMs);
        state.moveTimeMs_[~team] = std::max<int32_t>(1, opponentMoveTimeMs);
        return state;
    }

    bool is_clocked() const { return clocked_; }

    int32_t lead_ms(Stockfish::Color team) const {
        return team == Stockfish::WHITE ? whiteLeadMs_ : -whiteLeadMs_;
    }

    int32_t move_time_ms(Stockfish::Color team) const {
        return moveTimeMs_[team];
    }

    /** Whether @p team may sit: wait on a board instead of moving there. */
    bool can_sit(Stockfish::Color team) const {
        return lead_ms(team) > 0;
    }

    /**
     * @brief Charge the action @p actingTeam is about to play.
     *
     * Call it with the position before the action. Only a team on turn on both
     * boards pays; legacy leads never move.
     */
    void advance(Stockfish::Color actingTeam, bool actingTeamOnBothBoards) {
        if (!clocked_ || !actingTeamOnBothBoards) {
            return;
        }
        const int64_t cost = moveTimeMs_[actingTeam];
        whiteLeadMs_ = clamp_lead(
            static_cast<int64_t>(whiteLeadMs_)
            + (actingTeam == Stockfish::WHITE ? -cost : cost));
    }

    /**
     * @brief Coarse lead class for a search node's identity.
     *
     * The side that may sit is exact, since it decides which actions are
     * legal. The size of its lead counts on a log2 scale in its own move
     * times, so the few hundred milliseconds that pass between two searches
     * rarely move a retained position to a different node.
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
};

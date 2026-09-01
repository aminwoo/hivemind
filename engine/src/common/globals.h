#pragma once

#include <string>
#include <unordered_map>

/**
 * @brief Global log levels for debug output.
 */
enum LogLevel {
    LOG_NONE = 0,   // No debug output
    LOG_INFO = 1,   // Info-level output
    LOG_DEBUG = 2   // Verbose debug output
};

/**
 * @brief Global log level setting (default: LOG_NONE).
 */
extern LogLevel g_logLevel;

/**
 * @brief Board this team must move on when it has a legal move there.
 *
 * Bughouse actions are joint: a team may pass on one board while its partner
 * board moves. A client that only drives one seat needs a move for that seat,
 * so this constraint drops every joint action that passes on the named board
 * while a move there is available.
 */
enum RequiredMoveBoard {
    REQUIRE_MOVE_NONE = 0,      // Passing is allowed wherever the rules allow it
    REQUIRE_MOVE_BOARD_A = 1,   // Never pass on board A when it can move
    REQUIRE_MOVE_BOARD_B = 2    // Never pass on board B when it can move
};

/**
 * @brief Board the team must move on (default: REQUIRE_MOVE_NONE).
 */
extern RequiredMoveBoard g_requiredMoveBoard;

/**
 * @brief Parse a required-move board string ("none", "a"/"1", "b"/"2").
 * @return The corresponding RequiredMoveBoard, REQUIRE_MOVE_NONE if unrecognized.
 */
RequiredMoveBoard parseRequiredMoveBoard(const std::string& str);

/**
 * @brief Parse a log level string to LogLevel enum.
 * @param str The string to parse ("none", "info", "debug")
 * @return The corresponding LogLevel, defaults to LOG_NONE for unrecognized strings.
 */
LogLevel parseLogLevel(const std::string& str);

// Policy index mapping
extern std::unordered_map<std::string, int> POLICY_INDEX;
extern int POLICY_TABLE_NORMAL[2][64][64][2];
extern int POLICY_TABLE_DROP[2][64][8];

void init_policy_index();

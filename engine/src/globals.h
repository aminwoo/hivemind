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
 * @brief Parse a log level string to LogLevel enum.
 * @param str The string to parse ("none", "info", "debug")
 * @return The corresponding LogLevel, defaults to LOG_NONE for unrecognized strings.
 */
LogLevel parseLogLevel(const std::string& str);

// Policy index mapping (existing globals)
extern std::unordered_map<std::string, int> POLICY_INDEX;
void init_policy_index();

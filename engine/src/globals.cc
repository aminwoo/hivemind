#include "globals.h"
#include "constants.h"
#include <string>
#include <algorithm>

// Global log level (default: no debug output)
LogLevel g_logLevel = LOG_NONE;

std::unordered_map<std::string, int> POLICY_INDEX;

LogLevel parseLogLevel(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "debug") return LOG_DEBUG;
    if (lower == "info") return LOG_INFO;
    return LOG_NONE;  // Default
}

void init_policy_index() {
    for (size_t i = 0; i < NB_POLICY_VALUES(); i++) {
        if (POLICY_INDEX.find(UCI_MOVES[i]) == POLICY_INDEX.end()) {
            POLICY_INDEX[UCI_MOVES[i]] = i; 
        }
    }
}

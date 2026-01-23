#include "constants.h"
#include <string>

std::unordered_map<std::string, int> POLICY_INDEX;

void init_policy_index() {
    for (int i = 0; i < NB_POLICY_VALUES(); i++) {
        if (POLICY_INDEX.find(UCI_MOVES[i]) == POLICY_INDEX.end()) {
            POLICY_INDEX[UCI_MOVES[i]] = i; 
        }
    }
}

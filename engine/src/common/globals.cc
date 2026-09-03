#include "common/globals.h"
#include "environment/constants.h"
#include <string>
#include <algorithm>
#include <mutex>

#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/psqt.h"
#include "Fairy-Stockfish/src/search.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/tt.h"
#include "Fairy-Stockfish/src/tune.h"
#include "Fairy-Stockfish/src/uci.h"
#include "Fairy-Stockfish/src/variant.h"

void init_fairy_stockfish() {
    static std::once_flag initialised;
    std::call_once(initialised, [] {
        Stockfish::pieceMap.init();
        Stockfish::variants.init();
        Stockfish::UCI::init(Stockfish::Options);
        Stockfish::Tune::init();
        Stockfish::PSQT::init(Stockfish::variants.find("bughouse")->second);
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
        Stockfish::Bitbases::init();
        // Endgames::init() is deliberately skipped: it builds its probe
        // positions through Position::set(code, ...), which resolves the
        // "fairy" variant this build no longer ships. Material::probe() falls
        // back to the generic evaluation when the endgame maps are empty, and
        // no specialised endgame is reachable from a bughouse position anyway.
        // One idle main thread, plus the single worker the mate probe drives.
        Stockfish::Threads.set(2);
        Stockfish::TT.resize(size_t(Stockfish::Options["Hash"]));
        Stockfish::Search::clear();
    });
}

// Global log level (default: no debug output)
LogLevel g_logLevel = LOG_NONE;

// Board the team must move on (default: neither, so passing stays legal)
RequiredMoveBoard g_requiredMoveBoard = REQUIRE_MOVE_NONE;

std::unordered_map<std::string, int> POLICY_INDEX;
int POLICY_TABLE_NORMAL[2][64][64][2];
int POLICY_TABLE_DROP[2][64][8];

LogLevel parseLogLevel(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "debug") return LOG_DEBUG;
    if (lower == "info") return LOG_INFO;
    return LOG_NONE;  // Default
}

RequiredMoveBoard parseRequiredMoveBoard(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "a" || lower == "1") return REQUIRE_MOVE_BOARD_A;
    if (lower == "b" || lower == "2") return REQUIRE_MOVE_BOARD_B;
    return REQUIRE_MOVE_NONE;  // Default
}

static inline std::string fast_square_name(int sq) {
    int file = sq & 7;
    int rank = sq >> 3;
    std::string s;
    s.push_back(static_cast<char>('a' + file));
    s.push_back(static_cast<char>('1' + rank));
    return s;
}

static inline std::string fast_mirror_uci(const std::string& uci) {
    if (uci == "pass") return "pass";
    if (uci.size() >= 4 && uci[1] == '@') {
        std::string res = uci;
        int rank = res[3] - '0';
        int rank_mirrored = 8 - rank + 1;
        res[3] = static_cast<char>(rank_mirrored + '0');
        return res;
    }
    std::string res = uci;
    if (res.size() >= 4) {
        int r1 = res[1] - '0';
        int r2 = res[3] - '0';
        res[1] = static_cast<char>(8 - r1 + 1 + '0');
        res[3] = static_cast<char>(8 - r2 + 1 + '0');
    }
    return res;
}

void init_policy_index() {
    POLICY_INDEX.clear();
    for (size_t i = 0; i < NB_POLICY_VALUES(); i++) {
        if (POLICY_INDEX.find(UCI_MOVES[i]) == POLICY_INDEX.end()) {
            POLICY_INDEX[UCI_MOVES[i]] = static_cast<int>(i); 
        }
    }

    // Initialize fast lookup tables
    std::fill(&POLICY_TABLE_NORMAL[0][0][0][0],
              &POLICY_TABLE_NORMAL[0][0][0][0] + sizeof(POLICY_TABLE_NORMAL) / sizeof(int),
              -1);
    std::fill(&POLICY_TABLE_DROP[0][0][0],
              &POLICY_TABLE_DROP[0][0][0] + sizeof(POLICY_TABLE_DROP) / sizeof(int),
              -1);

    const char ptChars[8] = {' ', 'P', 'N', 'B', 'R', 'Q', 'K', ' '};

    for (int color = 0; color < 2; ++color) {
        for (int from_sq = 0; from_sq < 64; ++from_sq) {
            std::string from_name = fast_square_name(from_sq);
            for (int to_sq = 0; to_sq < 64; ++to_sq) {
                std::string to_name = fast_square_name(to_sq);

                // Normal move / Queen promotion
                std::string uci = from_name + to_name;
                std::string policy_uci = (color == 1) ? fast_mirror_uci(uci) : uci;
                auto it = POLICY_INDEX.find(policy_uci);
                if (it != POLICY_INDEX.end()) {
                    POLICY_TABLE_NORMAL[color][from_sq][to_sq][0] = it->second;
                }

                // Knight promotion ('n')
                std::string uci_n = uci + "n";
                std::string policy_uci_n = (color == 1) ? fast_mirror_uci(uci_n) : uci_n;
                auto it_n = POLICY_INDEX.find(policy_uci_n);
                if (it_n != POLICY_INDEX.end()) {
                    POLICY_TABLE_NORMAL[color][from_sq][to_sq][1] = it_n->second;
                }
            }
        }

        for (int to_sq = 0; to_sq < 64; ++to_sq) {
            std::string to_name = fast_square_name(to_sq);
            for (int pt = 1; pt <= 5; ++pt) {
                std::string drop_uci = std::string(1, ptChars[pt]) + "@" + to_name;
                std::string policy_drop_uci = (color == 1) ? fast_mirror_uci(drop_uci) : drop_uci;
                auto it = POLICY_INDEX.find(policy_drop_uci);
                if (it != POLICY_INDEX.end()) {
                    POLICY_TABLE_DROP[color][to_sq][pt] = it->second;
                }
            }
        }
    }
}

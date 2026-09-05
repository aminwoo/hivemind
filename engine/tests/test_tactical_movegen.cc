#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>
#include <thread>

#include "common/globals.h"
#include "environment/board.h"
#include "search/agent.h"
#include "Fairy-Stockfish/src/movegen.h"

class TacticalMovegenTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    static void compare_with_full_movegen(Board& board, int boardNum) {
        const auto legal = board.legal_moves(boardNum);
        std::set<Stockfish::Move> expected;
        for (auto move : legal) {
            EXPECT_TRUE(board.is_legal_move(boardNum, move));
            if (board.gives_check(boardNum, move)) {
                expected.insert(move);
            }
        }
        const auto checks = board.checking_moves(boardNum);
        EXPECT_EQ(std::set<Stockfish::Move>(checks.begin(), checks.end()), expected);
        EXPECT_EQ(checks.size(), expected.size());
        EXPECT_TRUE(board.is_legal_move(boardNum, Stockfish::MOVE_NONE));

        const auto& position = *board.pos[boardNum];
        Stockfish::ExtMove candidates[Stockfish::MAX_MOVES];
        const auto* end = position.checkers()
            ? Stockfish::generate<Stockfish::EVASIONS>(position, candidates)
            : Stockfish::generate<Stockfish::NON_EVASIONS>(position, candidates);
        for (const auto* candidate = candidates; candidate != end; ++candidate) {
            EXPECT_EQ(board.is_legal_move(boardNum, *candidate),
                      std::find(legal.begin(), legal.end(), candidate->move) != legal.end());
        }
    }
};

TEST_F(TacticalMovegenTest, IncludesSpecialChecksAndExcludesIllegalEvasions) {
    const std::vector<std::pair<std::string, std::string>> positions{
        // En passant opens the rook's file.
        {"4k3/8/8/3pP3/8/8/8/K3R3 w - d6 0 1", "e5d6"},
        // Castling puts the rook on the checking file.
        {"5k2/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1"},
        // Only the knight underpromotion gives check.
        {"8/4P1k1/8/8/8/8/8/K7 w - - 0 1", "e7e8n"},
        // A checking drop that also blocks a rook check.
        {"4r3/8/8/8/2k5/8/8/4K3[N] w - - 0 1", "N@e3"},
        {"r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2", "B@f7"},
    };
    for (const auto& [fen, requiredCheck] : positions) {
        for (int boardNum : {BOARD_A, BOARD_B}) {
            SCOPED_TRACE(fen);
            Board board;
            board.set_fen(boardNum, fen);
            compare_with_full_movegen(board, boardNum);
            const auto checks = board.checking_moves(boardNum);
            EXPECT_TRUE(std::any_of(checks.begin(), checks.end(), [&](auto move) {
                return board.uci_move(boardNum, move) == requiredCheck;
            })) << requiredCheck;
        }
    }
}

TEST_F(TacticalMovegenTest, MatchesFullMovegenAcrossPocketRichGames) {
    std::mt19937 random(20260905);
    for (int game = 0; game < 8; ++game) {
        Board board;
        for (int boardNum : {BOARD_A, BOARD_B}) {
            board.set_fen(boardNum,
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[PNBRQpnbrq] w KQkq - 0 1");
        }
        for (int ply = 0; ply < 100; ++ply) {
            const int boardNum = ply % 2;
            compare_with_full_movegen(board, boardNum);
            const auto moves = board.legal_moves(boardNum);
            if (moves.empty()) {
                break;
            }
            board.push_move(boardNum, moves[random() % moves.size()]);
        }
    }
}

TEST_F(TacticalMovegenTest, RejectsStaleMovesAndUnavailableDrops) {
    Board board;
    const auto move = board.legal_moves(BOARD_A).front();
    board.push_move(BOARD_A, move);
    EXPECT_FALSE(board.is_legal_move(BOARD_A, move));
    EXPECT_FALSE(board.is_legal_move(BOARD_A,
        Stockfish::make_drop(Stockfish::SQ_E4, Stockfish::QUEEN, Stockfish::QUEEN)));
}

TEST_F(TacticalMovegenTest, ProvesDeepMatesWithinTenThousandNodeAllowance) {
    struct Case {
        std::string boardA;
        std::string boardB;
        Stockfish::Color team;
        std::string move;
        int plies;
    };
    const std::vector<Case> cases{
        {"r4r2/1ppbn1pk/p3p2q/3pRn1B/5pP1/1BPP4/P1PB1PPP/1R4K1[QNp] b - - 0 1",
         "rnbN1b1r/pppk1Ppp/4pp2/4p3/4B3/8/PPPn1PPP/R3K1NR[QNPqb] w - - 0 1",
         Stockfish::BLACK, "h6h5", 7},
        {"r2q3r/ppp5/2n1Npkp/3p4/3P4/2P1P3/P1P2PPP/R2QK1NR[PNpppnbbrq] w KQ - 0 2",
         "r1bk4/ppp1npNp/2nb3B/3B2B1/3P4/2P5/P1P2PPP/R2QK2R[bpp] b KQ",
         Stockfish::WHITE, "d1h5", 9},
    };
    for (const auto& test : cases) {
        for (bool flipped : {false, true}) {
            SCOPED_TRACE(test.move + (flipped ? " flipped" : ""));
            Board board;
            board.set(flipped ? test.boardB + "|" + test.boardA
                              : test.boardA + "|" + test.boardB);
            const auto team = flipped ? ~test.team : test.team;
            const auto hash = board.search_hash_key(team, true);
            const auto signature = Agent::board_signature(board);
            JointActionCandidate action;
            int plies = 0;
            ASSERT_TRUE(Agent::find_root_mate(board, team, true, action, plies, 10000));
            EXPECT_EQ(plies, test.plies);
            EXPECT_EQ(board.uci_move(flipped ? BOARD_B : BOARD_A,
                flipped ? action.moveB : action.moveA), test.move);
            EXPECT_EQ(flipped ? action.moveA : action.moveB, Stockfish::MOVE_NONE);
            EXPECT_EQ(board.search_hash_key(team, true), hash);
            EXPECT_EQ(Agent::board_signature(board), signature);
        }
    }
}

TEST_F(TacticalMovegenTest, ReverseScanStopsWhenWorkersProvePosition16) {
    const std::string boardA =
        "2rq1rk1/pppnb1p1/4p1p1/3pP1pp/8/2N1PPB1/PPP2NPP/R2Q1RK1/Nn b - - 1 2";
    const std::string boardB =
        "r4rk1/ppp2p1p/4bB1p/8/6b1/2P5/P1PB1PPP/R3R1K1/qbbnnppPB w";
    for (bool flipped : {false, true}) {
        SCOPED_TRACE(flipped);
        Board board;
        board.set(flipped ? boardB + "|" + boardA : boardA + "|" + boardB);
        const auto team = flipped ? Stockfish::WHITE : Stockfish::BLACK;
        const auto signature = Agent::board_signature(board);
        const auto hash = board.search_hash_key(team, true);
        Node root(team, hash);

        // Reproduce a worker proving the root while the calling thread is
        // inside the reverse scan, without needing a neural network or GPU.
        std::jthread worker([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            root.mark_as_win(5);
        });
        const auto start = Agent::MateSearchBudget::Clock::now();
        std::vector<RootLossProof> proofs;
        EXPECT_FALSE(Agent::find_root_loss_proofs(
            board, team, true, proofs, 100000000,
            start + std::chrono::seconds(1), nullptr, nullptr, &root));
        const auto elapsed = Agent::MateSearchBudget::Clock::now() - start;
        std::cout << "Position 16" << (flipped ? "F" : "")
                  << " reverse scan returned in "
                  << std::chrono::duration<double, std::milli>(elapsed).count()
                  << " ms after worker proof at 20 ms\n";
        EXPECT_LT(elapsed, std::chrono::milliseconds(500));
        worker.join();
        EXPECT_EQ(root.get_node_type(), NodeType::WIN);
        EXPECT_EQ(Agent::board_signature(board), signature);
        EXPECT_EQ(board.search_hash_key(team, true), hash);
    }
}

TEST_F(TacticalMovegenTest, MateBudgetStopsOnProofRatherThanHighEvaluation) {
    for (NodeType outcome : {NodeType::WIN, NodeType::LOSS, NodeType::DRAW}) {
        Node root(Stockfish::WHITE);
        root.set_value(1.0f);
        Agent::MateSearchBudget budget;
        budget.stopOnSolvedRoot = &root;
        EXPECT_FALSE(budget.out_of_time());
        EXPECT_TRUE(budget.consume());

        if (outcome == NodeType::WIN) root.mark_as_win(5);
        else if (outcome == NodeType::LOSS) root.mark_as_loss(5);
        else root.mark_as_draw(1);
        EXPECT_TRUE(budget.out_of_time());
        for (uint32_t poll = 0; poll < SearchParams::MATE_SEARCH_TIME_CHECK_INTERVAL;
             ++poll) {
            budget.consume();
        }
        EXPECT_TRUE(budget.exhausted);
        EXPECT_FALSE(budget.consume());
    }
}

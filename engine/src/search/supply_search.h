#pragma once

#include <vector>
#include "environment/board.h"

// Cheap, bounded tactical candidate discovery for the ordinary neural search.
// These are exploration weights, never evaluations or mate certificates.
namespace SupplySearch {

std::vector<float> weights(const Board& board, int boardNumber,
                          Stockfish::Color team,
                          const std::vector<Stockfish::Move>& legalActions);

// Mix at most `fraction` of the policy mass into tactical candidates. No-op
// without candidates, preserving the network policy in quiet positions.
void mix_policy(std::vector<float>& policy, const std::vector<float>& weights,
                float fraction);

// Signed, bounded attack potential from `team`'s perspective. Uses current
// pockets and possible real transfers; does not add pieces or claim a mate.
float pressure(const Board& board, Stockfish::Color team);

} // namespace SupplySearch

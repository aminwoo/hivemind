#include <string>
#include <iostream>
#include <vector>

#include "parser.h"
#include "piece.h"
#include "variant.h"

using std::string;

namespace Stockfish {

VariantMap variants; // Global object

namespace {
    // Base variant
    Variant* variant_base() {
        Variant* v = new Variant();
        return v;
    }
    // Base for all fairy variants
    Variant* chess_variant_base() {
        Variant* v = variant_base()->init();
        v->pieceToCharTable = "PNBRQ................Kpnbrq................k";
        return v;
    }
    // Standard chess
    Variant* chess_variant() {
        Variant* v = chess_variant_base()->init();
        v->nnueAlias = "nn-";
        return v;
    }
    // Crazyhouse
    Variant* crazyhouse_variant() {
        Variant* v = chess_variant_base()->init();
        v->variantTemplate = "crazyhouse";
        v->startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1";
        v->pieceDrops = true;
        v->capturesToHand = true;
        return v;
    }
    // Bughouse
    Variant* bughouse_variant() {
        Variant* v = crazyhouse_variant()->init();
        v->variantTemplate = "bughouse";
        v->twoBoards = true;
        v->capturesToHand = false;
        v->stalemateValue = -VALUE_MATE;
        v->nFoldRule = 3;  // Threefold repetition
        return v;
    }
} // namespace

void VariantMap::init() {
    add("bughouse", bughouse_variant());
    add("crazyhouse", crazyhouse_variant());
    add("chess", chess_variant());
    add("normal", chess_variant());
}

void VariantMap::add(std::string s, Variant* v) {
    insert(std::pair<std::string, const Variant*>(s, v->conclude()));
}

void VariantMap::clear_all() {
    for (auto it = begin(); it != end(); ++it) {
        delete it->second;
    }
    clear();
}

std::vector<std::string> VariantMap::get_keys() {
    std::vector<std::string> keys;
    for (auto it = begin(); it != end(); ++it) {
        keys.push_back(it->first);
    }
    return keys;
}

} // namespace Stockfish

use crate::engine::transposition::{HashFlag, SearchData};

use super::{defs::SearchRefs, Search};
use shakmaty::{
    zobrist::{Zobrist64, ZobristHash},
    Chess, EnPassantMode, MoveList, Position,
};
use shakmaty_syzygy::{Tablebase, Wdl};

impl Search {
    pub fn alpha_beta(
        refs: &mut SearchRefs,
        mut depth: i8,
        mut alpha: i16,
        beta: i16,
        pv: &mut MoveList,
    ) -> i16 {
        let is_root = refs.search_info.ply == 0;

        if refs.search_info.elapsed() > refs.search_params.search_time {
            refs.search_info.terminated = true;
            return 0;
        }
        /*if refs.pos.board().occupied().count() <= 5 {
            println!("test");
            let mut tables: Tablebase<Chess> = Tablebase::new();
            tables.add_directory("tables/chess").unwrap();

            let wdl = tables.probe_wdl_after_zeroing(refs.pos).unwrap();
            match wdl {
                Wdl::Win => return 10000,
                Wdl::Loss => return -10000,
                Wdl::Draw => return 0,
                Wdl::BlessedLoss => return 0,
                Wdl::CursedWin => return 0,
            }
        }*/

        let is_check = refs.pos.is_check();
        if is_check {
            depth += 1;
        }

        if depth <= 0 {
            return Search::qsearch(refs, alpha, beta);
        }

        refs.search_info.nodes += 1;

        let mut tt_value: Option<i16> = None;

        if refs.tt_enabled {
            if let Some(data) = refs
                .tt
                .probe(refs.pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal))
            {
                tt_value = data.get(depth, refs.search_info.ply, alpha, beta);
            }
        }

        if let Some(v) = tt_value {
            if !is_root {
                refs.search_info.tt_hits += 1;
                return v;
            }
        }

        let mut hash_flag = HashFlag::Alpha;

        let mut legal_moves = refs.pos.legal_moves();
        let scores = Search::score_moves(&mut legal_moves, &refs.search_info.best_move, refs);

        for i in 0..legal_moves.len() {
            Search::pick_move(&mut legal_moves, i, scores);
            let mv = legal_moves.get(i).unwrap();

            let prev_pos = refs.pos.clone();
            refs.pos.play_unchecked(mv);
            refs.search_info.ply += 1;

            let mut node_pv: MoveList = MoveList::new();
            let score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha, &mut node_pv);

            *refs.pos = prev_pos;
            refs.search_info.ply -= 1;

            if score >= beta {
                refs.tt.insert(
                    refs.pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal),
                    SearchData::create(depth, refs.search_info.ply, HashFlag::Beta, beta),
                );

                if !mv.is_capture() {
                    let ply = refs.search_info.ply as usize;
                    if let Some(first_killer) = &refs.search_info.killer_moves1[ply] {
                        if mv != first_killer {
                            refs.search_info.killer_moves2[ply] =
                                refs.search_info.killer_moves1[ply].clone();
                            refs.search_info.killer_moves1[ply] = Some(mv.clone());
                        }
                    }

                    refs.search_info
                        .update_history(mv.role(), mv.to(), (depth * depth).into());
                }
                return beta;
            }

            if score > alpha {
                alpha = score;
                hash_flag = HashFlag::Exact;

                pv.clear();
                pv.push(mv.clone());
                pv.extend(node_pv.iter().cloned());
            }
        }
        if legal_moves.is_empty() {
            if is_check {
                return -9690 + (refs.search_info.ply as i16);
            } else {
                return 0;
            }
        }
        refs.tt.insert(
            refs.pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal),
            SearchData::create(depth, refs.search_info.ply, hash_flag, alpha),
        );
        alpha
    }
}

use super::sorting::see;
use super::{defs::SearchRefs, Search};
use crate::search::eval;
use crate::transposition::Bound;
use crate::types::Score;
use shakmaty::{Move, Position};

impl Search {
    pub fn qsearch(refs: &mut SearchRefs, mut alpha: i16, beta: i16) -> i16 {
        if (refs.search_info.nodes & 2047) == 0
            && refs.search_info.elapsed() > refs.search_params.search_time
        {
            refs.search_info.terminated = true;
            return 0;
        }
        if refs.search_info.elapsed() > refs.search_params.search_time {
            refs.search_info.terminated = true;
            return 0;
        }
        refs.search_info.nodes += 1;

        let ply = refs.search_info.ply as usize;
        let mut tt_move: Option<Move> = None;
        let hash = refs.get_hash();
        let hit = refs.tt.read(hash.0, ply);
        let is_check = refs.pos.is_check();

        let mut best_score = -Score::INFINITY;
        let mut best_move: Option<Move> = None;

        let mut score;
        if let Some(hit) = hit {
            match hit.bound {
                Bound::Exact => return hit.score,
                Bound::Alpha if hit.score <= alpha => return alpha,
                Bound::Beta if hit.score >= beta => return beta,
                _ => (),
            }
            score = hit.score;
            tt_move = hit.m;
        } else {
            score = eval::evaluate(refs.pos);
        }

        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }

        let mut captures = refs.pos.capture_moves();
        Search::sort_moves(&mut captures, &None, &tt_move, refs);

        for m in &captures {
            if !is_check && see(refs, m) < 0 {
                continue;
            }

            let prev_pos = refs.pos.clone();
            refs.pos.play_unchecked(m);
            score = -Search::qsearch(refs, -beta, -alpha);
            *refs.pos = prev_pos;

            if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
            }

            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        let bound = if best_score >= beta {
            Bound::Beta
        } else {
            Bound::Alpha
        };
        refs.tt.write(hash.0, 0, best_score, bound, best_move, ply);
        alpha
    }
}

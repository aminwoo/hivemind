use super::{defs::SearchRefs, Search};
use shakmaty::{Chess, Move, MoveList, Position};
use shakmaty_syzygy::{Tablebase, Wdl};

impl Search {
    pub fn iterative_deepening(refs: &mut SearchRefs) -> Move {
        let mut depth = 1;
        let mut root_pv = MoveList::new();

        let alpha = -10000;
        let beta = 10000;

        refs.search_info.start();
        while depth <= refs.search_params.depth {
            /*if refs.pos.board().occupied().count() <= 5 {
                let wdl = refs.tb.probe_wdl_after_zeroing(refs.pos).unwrap();
                match wdl {
                    Wdl::Win => refs.search_info.cp = 10000,
                    Wdl::Loss => refs.search_info.cp = -10000,
                    Wdl::Draw => refs.search_info.cp = 0,
                    Wdl::BlessedLoss => refs.search_info.cp = 0,
                    Wdl::CursedWin => refs.search_info.cp = 0,
                }
                let cp = refs.search_info.cp;
                let elapsed = refs.search_info.elapsed();
                let mut uci = String::from("");
                match refs.tb.best_move(refs.pos) {
                    Ok(Some((best_move, _))) => {
                        refs.search_info.best_move = Some(best_move.clone());
                        uci = best_move.to_uci(refs.pos.castles().mode()).to_string();
                    }
                    Ok(None) => {}
                    Err(e) => {
                        println!("Error querying Syzygy tablebase: {:?}", e)
                    }
                }
                println!(
                    "info depth {} score cp {} nodes {} nps {} tbhits {} time {} pv {}",
                    depth, cp, 1, 0, refs.search_info.tt_hits, elapsed, uci
                );
                return refs.search_info.best_move.clone().unwrap();
            }*/

            refs.search_info.cp = Search::alpha_beta(refs, depth, alpha, beta, &mut root_pv);

            if refs.search_info.terminated {
                break;
            }

            let cp = refs.search_info.cp;
            let nodes = refs.search_info.nodes;
            let elapsed = refs.search_info.elapsed();
            let nps = if elapsed > 0 {
                nodes / elapsed as usize * 1000
            } else {
                0
            };

            print!(
                "info depth {} score cp {} nodes {} nps {} tbhits {} time {} pv ",
                depth, cp, nodes, nps, refs.search_info.tt_hits, elapsed
            );

            for mv in &root_pv {
                let uci = mv.to_uci(refs.pos.castles().mode());
                print!("{} ", uci);
            }
            println!();

            refs.search_info.best_move = Some(root_pv[0].clone());
            depth += 1;

            if elapsed > (refs.search_params.search_time as f64 * 0.6).round() as u128 {
                break;
            }
        }

        refs.search_info.best_move.clone().unwrap()
    }
}

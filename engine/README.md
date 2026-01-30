./hivemind param-eval \  
 --model /home/ben/hivemind/engine/networks/model-0.97878-0.683-0224-v3.0.onnx \  
 --p1-name "Batch1" --p1-time 1000 --p1-batch 1 --p1-threads 1 \  
 --p2-name "Batch16" --p2-time 1000 --p2-batch 16 --p2-threads 2 \  
 --games 10

./hivemind param-eval \
 --model /home/ben/hivemind/engine/networks/model-0.97878-0.683-0224-v3.0.onnx \
 --p1-name "MCGS-ON" --p1-time 1000 --p1-batch 8 --p1-threads 2 --p1-mcgs 1 \
 --p2-name "MCGS-OFF" --p2-time 1000 --p2-batch 8 --p2-threads 2 --p2-mcgs 0 \
 --games 20

./hivemind param-eval \
 --model /home/ben/hivemind/engine/networks/model-0.97878-0.683-0224-v3.0.onnx \
 --p1-name "PW-1.0-0.3" --p1-time 1000 --p1-batch 8 --p1-threads 2 --p1-pw-coef 1.0 --p1-pw-exp 0.3 \
 --p2-name "PW-2.0-0.7" --p2-time 1000 --p2-batch 8 --p2-threads 2 --p2-pw-coef 2.0 --p2-pw-exp 0.7 \
 --games 20

./hivemind param-eval \
 --model /home/ben/hivemind/engine/networks/model-0.97878-0.683-0224-v3.0.onnx \
 --p1-time 500 --p1-batch 8 --p1-threads 2 --p1-contempt 0.0 \
 --p2-time 500 --p2-batch 8 --p2-threads 2 --p2-contempt 1.0 \
 --games 100

./hivemind param-eval \
 --model /home/ben/hivemind/engine/networks/model-0.97878-0.683-0224-v3.0.onnx \
 --p1-name "FPU-0.2" --p1-time 500 --p1-batch 16 --p1-threads 2 --p1-fpu 0.2 \
 --p2-name "FPU-0.6" --p2-time 500 --p2-batch 16 --p2-threads 2 --p2-fpu 0.6 \
 --games 20

./hivemind eval \
 --new /home/ben/hivemind/src/training/weights/model-rl-final-v3.0.onnx \
 --old /home/ben/hivemind/src/training/weights/model-0.97878-0.683-0224-v3.0.onnx \
 -g 100 -n 1600 -t 0.6 -v --pgn ./eval_games.pgn

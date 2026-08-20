#pragma once

#include "environment/board.h"
#include "nn/engine.h"

void benchmark_inference(Engine& engine, int iterations = 1000);
void benchmark_movegen(int depth = 5);

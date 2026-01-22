#pragma once

#include "board.h"
#include "engine.h"

void benchmark_inference(Engine& engine, int iterations = 1000);
void benchmark_movegen(int depth = 5);

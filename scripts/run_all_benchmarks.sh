#!/bin/bash
export PYTHONPATH=./src
mkdir -p results/mc_static_benchmark/
mkdir -p results/mc_streaming_replay/

# Function to run sequence for a chain
run_chain() {
    local chain=$1
    local max_samples=$2
    local extra_args=""
    if [ ! -z "$max_samples" ]; then
        extra_args="--max_samples $max_samples"
    fi

    echo ">>> Starting Static Benchmark for $chain..."
    .venv/bin/python ./src/gog_fraud/pipelines/run_mc_benchmark.py \
        --config configs/mc/static_benchmark.yaml \
        --chain $chain \
        --output results/mc_static_benchmark/$chain/ \
        --bootstrap \
        --stages l1,l1_l2 $extra_args > results/${chain}_static.log 2>&1

    echo ">>> Starting Streaming Replay for $chain..."
    .venv/bin/python ./src/gog_fraud/pipelines/run_streaming_replay.py \
        --config configs/mc/streaming_replay.yaml \
        --chain $chain \
        --output results/mc_streaming_replay/$chain/ $extra_args > results/${chain}_replay.log 2>&1
}

# 1. Polygon (Already partially done, but re-run replay for consistency)
run_chain polygon

# 2. BSC
run_chain bsc

# 3. Ethereum (Subsetting to 5000 for ROI)
run_chain ethereum 5000

echo ">>> All benchmarks completed."

#!/bin/sh

export LOGURU_LEVEL=ERROR

echo "Running Figures 4.a and 4.b.."
python3 experiments/runner.cli.caching.py --exp microbenchmark_varying_knob1

echo "Running Figures 4.c and 4.d.."
python3 experiments/runner.cli.caching.py --exp microbenchmark_varying_knob2

echo "Running Figures 5.a, 5.b and 5.c.."
python3 experiments/runner.cli.caching.py --exp patcg_varying_epoch_granularity

echo "Running Figures 6.a, 6.b and 6.c.."
python3 experiments/runner.cli.py --exp criteo_run

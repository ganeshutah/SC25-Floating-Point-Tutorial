#!/bin/bash
# Run experiments with nixnan GPU tracing using LOGFILE env variable

if [ $# -lt 1 ]; then
    echo "Usage: ./run_with_nixnan.sh <experiment> [args]"
    echo ""
    echo "Examples:"
    echo "  ./run_with_nixnan.sh baseline"
    echo "  ./run_with_nixnan.sh bfloat16"
    echo "  ./run_with_nixnan.sh attention_scale --scale 0.5"
    exit 1
fi

# Get experiment name (first argument)
EXPERIMENT="$1"
shift

# Find nixnan.so
NIXNAN_SO="${NIXNAN_SO:-}"
if [ -z "$NIXNAN_SO" ]; then
    if [ -f "/usr/local/lib/libnixnan.so" ]; then
        NIXNAN_SO="/usr/local/lib/libnixnan.so"
    fi
fi

# Check if nixnan found
if [ -z "$NIXNAN_SO" ] || [ ! -f "$NIXNAN_SO" ]; then
    echo "nixnan not found - running without SASS traces"
    echo "Set: export NIXNAN_SO=/path/to/nixnan.so"
    echo ""
    python3 single_experiment.py --experiment "$EXPERIMENT" "$@"
    exit 0
fi

# Create output directory
mkdir -p nan_experiments/nixnan_traces

# Set up paths
LOGFILE="nan_experiments/nixnan_traces/claude_${EXPERIMENT}.nixnan"

echo "========================================================================"
echo "Running: $EXPERIMENT"
echo "nixnan: $NIXNAN_SO"
echo "SASS log: $LOGFILE"
echo "========================================================================"

# Run with nixnan
LD_PRELOAD="$NIXNAN_SO" LOGFILE="$LOGFILE" \
    python3 single_experiment.py --experiment "$EXPERIMENT" "$@"

echo ""
echo "Done! SASS traces: $LOGFILE"

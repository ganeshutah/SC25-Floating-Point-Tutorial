# BioMistral NaN Experiments

## Quick Start
```bash
./install.sh
make priority
make analyze
```

## With SASS traces (nixnan)
```bash
# Set nixnan path
export NIXNAN_SO=/path/to/nixnan.so

# Run with SASS tracing
make nixnan-priority

# Analyze Python + SASS
make analyze-sass
```

## How nixnan LOGFILE Works

nixnan uses the `LOGFILE` environment variable to specify output:

```bash
# The script does this:
export LD_PRELOAD=/path/to/nixnan.so
export LOGFILE=nan_experiments/nixnan_traces/baseline.nixnan
python3 single_experiment.py --experiment baseline

# nixnan automatically writes to $LOGFILE
```

## Manual nixnan Usage

```bash
# Basic usage with LOGFILE
export NIXNAN_SO=/path/to/nixnan.so
export LD_PRELOAD=$NIXNAN_SO
export LOGFILE=my_sass_trace.log
python3 single_experiment.py --experiment baseline

# One-liner
LD_PRELOAD=/path/to/nixnan.so LOGFILE=trace.log python3 single_experiment.py --experiment baseline

# Script wrapper (recommended)
./run_with_nixnan.sh baseline
```

## Commands
- `make priority` - Run 4 key tests (Python-level)
- `make analyze` - Python-level analysis  
- `make analyze-sass` - Python + SASS analysis
- `make nixnan-priority` - Run with GPU SASS traces
- `./run_with_nixnan.sh <exp>` - Single experiment with SASS

## Finding nixnan.so

```bash
# Search for it
find ~ -name "nixnan.so" 2>/dev/null

# Common locations:
# ~/NVBit/tools/nixnan/nixnan.so
# /usr/local/lib/libnixnan.so
# /opt/nvidia/nvbit/tools/nixnan/nixnan.so

# Set permanently
echo 'export NIXNAN_SO=/path/to/nixnan.so' >> ~/.bashrc
```

## Output Files

**Python-level:**
- `nan_experiments/claude_baseline_4bit_fp16.log` - Summary
- `nan_experiments/claude_baseline_4bit_fp16.json` - Full data

**SASS-level (with nixnan):**
- `nan_experiments/nixnan_traces/claude_baseline.nixnan` - GPU instruction traces

## Analyzing SASS Traces

The `analyze_results.py` script automatically parses .nixnan files:

```bash
# Full analysis
python3 analyze_results.py --sass

# Shows:
# - Python-level NaN counts
# - SASS-level error counts  
# - Kernel breakdown (Flash Attention, GEMM, etc.)
# - Instruction breakdown (HMMA, FMUL, etc.)
# - Combined correlation
```

Manual analysis:

```bash
# View all NaN errors
grep 'error.*NaN' nan_experiments/nixnan_traces/*.nixnan | head -20

# Focus on Flash Attention
grep 'pytorch_flash' nan_experiments/nixnan_traces/claude_baseline.nixnan

# Count errors by type
grep '#nixnan: error' baseline.nixnan | grep -o '\[.*\]' | sort | uniq -c

# Top problematic instructions
grep 'instruction' baseline.nixnan | sed 's/.*instruction //' | cut -d' ' -f1 | sort | uniq -c | sort -rn | head -10

# Top problematic kernels/functions
grep 'in function' baseline.nixnan | sed 's/.*in function //' | cut -d' ' -f1 | sort | uniq -c | sort -rn | head -10
```

## SASS Trace Format

nixnan outputs errors like:
```
#nixnan: error [NaN,subnormal] detected in operand 2 of instruction 
HMMA.16816.F32 R0, R96.reuse, R128, R0 ; 
in function pytorch_flash::flash_fwd_kernel of type f16
```

The analyzer parses:
- **Error type**: NaN, subnormal, infinity
- **Instruction**: HMMA, FMUL, MUFU, etc.
- **Function**: Kernel name (e.g., pytorch_flash::flash_fwd_kernel)

## Troubleshooting

**"nixnan not found"**
- Set NIXNAN_SO: `export NIXNAN_SO=/path/to/nixnan.so`
- Or edit run_with_nixnan.sh with hardcoded path

**Empty .nixnan files**
- Check LOGFILE is set: `echo $LOGFILE`
- Verify nixnan.so loads: `LD_PRELOAD=$NIXNAN_SO python3 -c "print('test')"`
- Check GPU compatibility

**No SASS traces generated**
- Ensure LOGFILE env variable is exported
- Check file permissions on nan_experiments/nixnan_traces/
- Verify nixnan.so path is correct

**LOGFILE not being created**
- nixnan creates file automatically when set
- No need to pre-create the file
- Parent directory must exist (scripts create it)

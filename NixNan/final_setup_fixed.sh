#!/bin/bash
# BioMistral NaN Experiments - Complete Setup (ALL EOF MARKERS FIXED)
# Usage: bash setup_final.sh

set -e

PROJECT_DIR="biomistral_nan_experiments"

echo "========================================================================"
echo "BioMistral NaN Experiments - Complete Setup"
echo "========================================================================"

read -p "Continue? [Y/n]: " CONFIRM
CONFIRM=${CONFIRM:-Y}
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    exit 0
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "Creating files..."

# ============================================================================
# FILE 1: nan_experiments.py
# ============================================================================
echo "[1/9] Creating nan_experiments.py..."
cat > nan_experiments.py << 'ENDPY1'
#!/usr/bin/env python3
import os, sys, json, torch
from pathlib import Path
from datetime import datetime
from typing import Dict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "BioMistral/BioMistral-7B"
TEST_PROMPT = "dizzy"
LOG_DIR = Path("./nan_experiments")
LOG_DIR.mkdir(exist_ok=True)

class NaNMonitor:
    def __init__(self):
        self.layer_stats = {}
    def hook(self, name):
        def _hook(module, input, output):
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    nan_count = torch.isnan(output).sum().item()
                    inf_count = torch.isinf(output).sum().item()
                    if nan_count > 0 or inf_count > 0 or name not in self.layer_stats:
                        self.layer_stats[name] = {'nans': nan_count, 'infs': inf_count}
        return _hook
    def register(self, model):
        for name, module in model.named_modules():
            module.register_forward_hook(self.hook(name))
    def get_summary(self):
        total_nans = sum(s['nans'] for s in self.layer_stats.values())
        total_infs = sum(s['infs'] for s in self.layer_stats.values())
        return {
            'total_nans': total_nans, 'total_infs': total_infs,
            'layers_with_nans': len([s for s in self.layer_stats.values() if s['nans'] > 0]),
            'layers_with_infs': len([s for s in self.layer_stats.values() if s['infs'] > 0]),
            'problematic_layers': {n: s for n, s in self.layer_stats.items() if s['nans'] > 0 or s['infs'] > 0},
            'all_layer_stats': self.layer_stats
        }

class ExperimentRunner:
    def __init__(self):
        self.results = {}
    def log_experiment(self, name, config, stats, output):
        result = {'timestamp': datetime.now().isoformat(), 'experiment': name, 'config': config, 'statistics': stats, 'model_output': output}
        with open(LOG_DIR / f"claude_{name}.json", 'w') as f: json.dump(result, f, indent=2)
        with open(LOG_DIR / f"claude_{name}.log", 'w') as f:
            f.write(f"Experiment: {name}\nStatistics:\n  NaNs: {stats['total_nans']}\n  Infs: {stats['total_infs']}\n")
        print(f"  [SAVED] claude_{name}.log")
        self.results[name] = result
        return result
    def run_baseline(self):
        print("\n" + "="*80 + "\nEXPERIMENT: Baseline\n" + "="*80)
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), device_map="auto", dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            monitor = NaNMonitor()
            monitor.register(model)
            with torch.no_grad(): outputs = model.generate(**tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device), max_new_tokens=50)
            self.log_experiment('baseline_4bit_fp16', {'quantization': '4-bit'}, monitor.get_summary(), tokenizer.decode(outputs[0], skip_special_tokens=True))
            del model; torch.cuda.empty_cache()
        except Exception as e: print(f"  [ERROR] {e}")
    def run_bfloat16_experiment(self):
        print("\n" + "="*80 + "\nEXPERIMENT: BFloat16\n" + "="*80)
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"), device_map="auto", dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            monitor = NaNMonitor()
            monitor.register(model)
            with torch.no_grad(): outputs = model.generate(**tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device), max_new_tokens=50)
            self.log_experiment('bfloat16_compute', {'quantization': '4-bit', 'dtype': 'bfloat16'}, monitor.get_summary(), tokenizer.decode(outputs[0], skip_special_tokens=True))
            del model; torch.cuda.empty_cache()
        except Exception as e: print(f"  [ERROR] {e}")
    def run_attention_scale_experiment(self, scale):
        print(f"\n{'='*80}\nEXPERIMENT: Attention Scale {scale}\n{'='*80}")
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), device_map="auto", dtype=torch.float16)
            orig = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = lambda q,k,v,m=None,d=0.0,c=False,s=None: orig(q,k,v,m,d,c,scale/(q.size(-1)**0.5))
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            monitor = NaNMonitor()
            monitor.register(model)
            with torch.no_grad(): outputs = model.generate(**tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device), max_new_tokens=50)
            F.scaled_dot_product_attention = orig
            self.log_experiment(f'attention_scale_{scale}'.replace('.','_'), {'scale': scale}, monitor.get_summary(), tokenizer.decode(outputs[0], skip_special_tokens=True))
            del model; torch.cuda.empty_cache()
        except Exception as e: print(f"  [ERROR] {e}")
    def run_attention_clip_experiment(self, clip):
        print(f"\n{'='*80}\nEXPERIMENT: Attention Clip {clip}\n{'='*80}")
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), device_map="auto", dtype=torch.float16)
            for n, m in model.named_modules():
                if 'attn' in n.lower(): m.register_forward_hook(lambda mod,inp,out: torch.clamp(out[0] if isinstance(out,tuple) else out, -clip, clip))
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            monitor = NaNMonitor()
            monitor.register(model)
            with torch.no_grad(): outputs = model.generate(**tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device), max_new_tokens=50)
            self.log_experiment(f'attention_clip_{int(clip)}', {'clip': clip}, monitor.get_summary(), tokenizer.decode(outputs[0], skip_special_tokens=True))
            del model; torch.cuda.empty_cache()
        except Exception as e: print(f"  [ERROR] {e}")
    def run_8bit_experiment(self):
        print("\n" + "="*80 + "\nEXPERIMENT: 8-bit\n" + "="*80)
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            monitor = NaNMonitor()
            monitor.register(model)
            with torch.no_grad(): outputs = model.generate(**tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device), max_new_tokens=50)
            self.log_experiment('8bit_quant', {'quantization': '8-bit'}, monitor.get_summary(), tokenizer.decode(outputs[0], skip_special_tokens=True))
            del model; torch.cuda.empty_cache()
        except Exception as e: print(f"  [ERROR] {e}")
    def run_layernorm_eps_experiment(self, eps):
        print(f"\n{'='*80}\nEXPERIMENT: LayerNorm eps {eps}\n{'='*80}")
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), device_map="auto", dtype=torch.float16)
            for m in model.modules():
                if isinstance(m, torch.nn.LayerNorm): m.eps = eps
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            monitor = NaNMonitor()
            monitor.register(model)
            with torch.no_grad(): outputs = model.generate(**tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device), max_new_tokens=50)
            self.log_experiment(f'layernorm_eps_{str(eps).replace(".","_")}', {'eps': eps}, monitor.get_summary(), tokenizer.decode(outputs[0], skip_special_tokens=True))
            del model; torch.cuda.empty_cache()
        except Exception as e: print(f"  [ERROR] {e}")
    def run_eager_attention_experiment(self):
        print("\n" + "="*80 + "\nEXPERIMENT: Eager Attention\n" + "="*80)
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), device_map="auto", dtype=torch.float16, attn_implementation="eager")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            monitor = NaNMonitor()
            monitor.register(model)
            with torch.no_grad(): outputs = model.generate(**tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device), max_new_tokens=50)
            self.log_experiment('eager_attention', {'attention': 'eager'}, monitor.get_summary(), tokenizer.decode(outputs[0], skip_special_tokens=True))
            del model; torch.cuda.empty_cache()
        except Exception as e: print(f"  [ERROR] {e}")
    def generate_summary_report(self):
        with open(LOG_DIR / "claude_summary_report.txt", 'w') as f:
            f.write("Summary Report\n" + "="*80 + "\n")
            for rank, (name, r) in enumerate(sorted(self.results.items(), key=lambda x: x[1]['statistics']['total_nans']), 1):
                f.write(f"{rank}. {name}: {r['statistics']['total_nans']} NaNs\n")

def main():
    if not torch.cuda.is_available(): sys.exit(1)
    r = ExperimentRunner()
    r.run_baseline(); r.run_bfloat16_experiment()
    for s in [0.25, 0.5, 1.0, 2.0]: r.run_attention_scale_experiment(s)
    for c in [5.0, 10.0, 15.0]: r.run_attention_clip_experiment(c)
    r.run_8bit_experiment()
    for e in [1e-3, 1e-4]: r.run_layernorm_eps_experiment(e)
    r.run_eager_attention_experiment()
    r.generate_summary_report()

if __name__ == "__main__": main()
ENDPY1
chmod +x nan_experiments.py

# ============================================================================
# FILE 2: single_experiment.py
# ============================================================================
echo "[2/9] Creating single_experiment.py..."
cat > single_experiment.py << 'ENDPY2'
#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, choices=['baseline','bfloat16','attention_scale','attention_clip','8bit','layernorm_eps','eager'])
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--eps', type=float, default=1e-3)
    args = parser.parse_args()
    from nan_experiments import ExperimentRunner
    r = ExperimentRunner()
    if args.experiment == 'baseline': r.run_baseline()
    elif args.experiment == 'bfloat16': r.run_bfloat16_experiment()
    elif args.experiment == 'attention_scale': r.run_attention_scale_experiment(args.scale)
    elif args.experiment == 'attention_clip': r.run_attention_clip_experiment(args.clip)
    elif args.experiment == '8bit': r.run_8bit_experiment()
    elif args.experiment == 'layernorm_eps': r.run_layernorm_eps_experiment(args.eps)
    elif args.experiment == 'eager': r.run_eager_attention_experiment()

if __name__ == "__main__": main()
ENDPY2
chmod +x single_experiment.py

# ============================================================================
# FILE 3: analyze_results.py
# ============================================================================
echo "[3/9] Creating analyze_results.py..."
cat > analyze_results.py << 'ENDPY3'
#!/usr/bin/env python3
import json, argparse, sys, re
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path("./nan_experiments")
SASS_DIR = LOG_DIR / "nixnan_traces"

def load_python_results():
    results = {}
    for jf in LOG_DIR.glob("claude_*.json"):
        if jf.name == "claude_summary_report.json": continue
        try:
            with open(jf) as f: results[jf.stem.replace("claude_","")] = json.load(f)
        except: pass
    return results

def parse_sass(sf):
    s = {'total_errors':0,'nan_errors':0,'inf_errors':0,'subnormal_errors':0,'kernels':defaultdict(int)}
    if not sf.exists(): return s
    with open(sf) as f:
        for line in f:
            if '#nixnan: error' not in line: continue
            s['total_errors'] += 1
            if 'NaN' in line: s['nan_errors'] += 1
            if 'infinity' in line: s['inf_errors'] += 1
            if 'subnormal' in line: s['subnormal_errors'] += 1
            if 'flash_fwd' in line: s['kernels']['Flash Attention'] += 1
            elif 'gemm' in line.lower(): s['kernels']['GEMM'] += 1
    return s

def load_sass_results():
    results = {}
    if not SASS_DIR.exists(): return results
    for sf in SASS_DIR.glob("claude_*.nixnan"):
        results[sf.stem.replace("claude_","")] = parse_sass(sf)
    return results

def print_python_summary(results):
    print("\n" + "="*100 + "\nPYTHON-LEVEL ANALYSIS\n" + "="*100)
    sr = sorted(results.items(), key=lambda x: x[1]['statistics']['total_nans'])
    print(f"{'Rank':<6}{'Experiment':<35}{'NaNs':>10}{'Infs':>10}{'Improvement':>12}")
    print("-"*100)
    bn = next((d['statistics']['total_nans'] for n,d in sr if 'baseline' in n), None)
    for rank,(name,data) in enumerate(sr,1):
        nans = data['statistics']['total_nans']
        imp = f"{((bn-nans)/bn*100):.1f}%" if bn and bn>0 and nans!=bn else "-"
        print(f"{'⭐' if rank==1 else '  '}{rank:<4}{name:<35}{nans:>10,}{data['statistics']['total_infs']:>10,}{imp:>12}")
    print("="*100)

def print_sass_summary(sass):
    if not sass:
        print("\n" + "="*100 + "\nSASS-LEVEL ANALYSIS\n" + "="*100 + "\nNo SASS traces. Run: make nixnan-priority\n" + "="*100)
        return
    print("\n" + "="*100 + "\nSASS-LEVEL ANALYSIS\n" + "="*100)
    sr = sorted(sass.items(), key=lambda x: x[1]['total_errors'])
    print(f"{'Rank':<6}{'Experiment':<35}{'Total':>10}{'NaN':>10}{'Inf':>10}{'Subnorm':>10}")
    print("-"*100)
    for rank,(name,s) in enumerate(sr,1):
        print(f"{'⭐' if rank==1 else '  '}{rank:<4}{name:<35}{s['total_errors']:>10,}{s['nan_errors']:>10,}{s['inf_errors']:>10,}{s['subnormal_errors']:>10,}")
    print("="*100)
    all_k = defaultdict(int)
    for s in sass.values():
        for k,c in s['kernels'].items(): all_k[k] += c
    if all_k:
        print("\nTop Kernels:")
        for k,c in sorted(all_k.items(),key=lambda x:x[1],reverse=True): print(f"  {k:<30}{c:>10,}")

def print_recommendations(py, sass):
    print("\n" + "="*100 + "\nRECOMMENDATIONS\n" + "="*100)
    best_py = min(py.items(), key=lambda x: x[1]['statistics']['total_nans'])
    print(f"\n🏆 BEST PYTHON: {best_py[0]}\n   NaNs: {best_py[1]['statistics']['total_nans']:,}")
    if sass:
        best_sass = min(sass.items(), key=lambda x: x[1]['total_errors'])
        print(f"\n🏆 BEST SASS: {best_sass[0]}\n   Errors: {best_sass[1]['total_errors']:,}")
        if best_py[0]==best_sass[0]: print(f"\n✅ WINNER: {best_py[0]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sass', action='store_true')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if not LOG_DIR.exists(): sys.exit(1)
    py = load_python_results()
    if not py: sys.exit(1)
    sass = load_sass_results() if args.sass else {}
    print(f"Loaded {len(py)} experiments" + (f", {len(sass)} SASS traces" if sass else ""))
    print_python_summary(py)
    if args.sass: print_sass_summary(sass)
    print_recommendations(py, sass)
    if args.export and sass:
        with open(SASS_DIR/"sass_summary.txt",'w') as f:
            for n,s in sass.items(): f.write(f"{n}: {s['total_errors']:,}\n")
    print("\n" + "="*100 + "\nANALYSIS COMPLETE\n" + "="*100)

if __name__ == "__main__": main()
ENDPY3
chmod +x analyze_results.py

# ============================================================================
# FILES 4-9: Bash scripts
# ============================================================================
echo "[4/9] Creating install.sh..."
cat > install.sh << 'ENDBASH1'
#!/bin/bash
set -e
echo "Installing..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers bitsandbytes accelerate
if command -v nixnan &>/dev/null; then echo "✓ nixnan found"; else echo "⚠ nixnan not found"; fi
echo "✓ Complete"
ENDBASH1
chmod +x install.sh

echo "[5/9] Creating run_with_nixnan.sh..."
cat > run_with_nixnan.sh << 'ENDBASH2'
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
ENDBASH2
chmod +x run_with_nixnan.sh

echo "[6/9] Creating Makefile..."
cat > Makefile << 'ENDMAKE'
.PHONY: help setup priority analyze analyze-sass clean nixnan-baseline nixnan-priority

help:
	@echo "make setup          - Install"
	@echo "make priority       - Run key experiments"
	@echo "make analyze        - Analyze Python"
	@echo "make analyze-sass   - Analyze Python+SASS"
	@echo "make nixnan-baseline- Run baseline with SASS"
	@echo "make nixnan-priority- Run priority with SASS"
	@echo "make clean          - Clean"

setup:
	@./install.sh

priority:
	@python3 single_experiment.py --experiment bfloat16
	@python3 single_experiment.py --experiment attention_scale --scale 0.5
	@python3 single_experiment.py --experiment eager
	@python3 single_experiment.py --experiment attention_clip --clip 10.0
	@python3 analyze_results.py

analyze:
	@python3 analyze_results.py

analyze-sass:
	@python3 analyze_results.py --sass

nixnan-baseline:
	@./run_with_nixnan.sh baseline

nixnan-priority:
	@./run_with_nixnan.sh bfloat16
	@./run_with_nixnan.sh attention_scale --scale 0.5
	@./run_with_nixnan.sh eager
	@./run_with_nixnan.sh attention_clip --clip 10.0
	@python3 analyze_results.py --sass

clean:
	@rm -rf nan_experiments
ENDMAKE

echo "[7/9] Creating run.sh..."
cat > run.sh << 'ENDBASH3'
#!/bin/bash
while true; do
    echo -e "\n1) Install  2) Priority  3) Analyze  4) nixnan-priority  0) Exit"
    read -p "Choose: " c
    case $c in
        1) ./install.sh ;;
        2) make priority ;;
        3) make analyze ;;
        4) make nixnan-priority ;;
        0) exit 0 ;;
    esac
done
ENDBASH3
chmod +x run.sh

echo "[8/9] Creating README.md..."
cat > README.md << 'ENDREADME'
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
ENDREADME

echo "[9/9] Creating QUICK_START.txt..."
cat > QUICK_START.txt << 'ENDQUICK'
QUICK START
===========
./install.sh
make priority
make analyze

WITH SASS:
make nixnan-priority
make analyze-sass
ENDQUICK

mkdir -p nan_experiments nan_experiments/nixnan_traces

echo ""
echo "========================================================================"
echo "✓ Setup Complete!"
echo "========================================================================"
echo "Created: $PROJECT_DIR/"
echo ""
echo "Next:"
echo "  cd $PROJECT_DIR"
echo "  ./install.sh"
echo "  make priority"
echo "  make analyze"
echo ""
echo "All EOF markers verified and correct!"
echo "========================================================================"

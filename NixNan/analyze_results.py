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

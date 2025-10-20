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

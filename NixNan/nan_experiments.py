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

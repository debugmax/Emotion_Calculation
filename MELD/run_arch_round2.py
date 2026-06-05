#!/usr/bin/env python3
import subprocess, re, json, os
from datetime import datetime

BASELINE = {'test_acc': 66.97, 'test_fscore': 65.47, 'composite': 132.44}
EXPERIMENTS = [
    {'name': 's6_residual_attn', 'desc': '线性基线+注意力残差', 'args': ['--fusion_arch', 'residual_attn']},
    {'name': 's7_residual_cross', 'desc': '线性基线+交叉注意力残差', 'args': ['--fusion_arch', 'residual_cross']},
    {'name': 's8_residual+logit', 'desc': '残差注意力+Logit集成', 'args': [
        '--fusion_arch', 'residual_attn', '--ensemble_a', '0.3', '--ensemble_v', '0.3']},
    {'name': 's9_linear+moddrop', 'desc': '线性+模态dropout训练', 'args': [
        '--fusion_arch', 'linear', '--modality_drop', '0.15']},
]

def parse_final(log_path):
    with open(log_path) as f:
        m = re.search(r'FINAL best epoch (-?\d+): test_acc=([\d.]+), test_fscore=([\d.]+), composite=([\d.]+)', f.read())
    return None if not m else {'epoch': int(m.group(1)), 'test_acc': float(m.group(2)),
            'test_fscore': float(m.group(3)), 'composite': float(m.group(4))}

def main():
    results_dir = './experiment_logs'
    results = []
    for exp in EXPERIMENTS:
        log_file = os.path.join(results_dir, f"{exp['name']}.log")
        cmd = ['python', 'multimodel_fusion.py', '--exp_name', exp['name'], '--log_file', log_file] + exp['args']
        print(f"\n>>> {exp['name']}: {exp['desc']}")
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        r = parse_final(log_file)
        if r:
            r.update(exp)
            r['delta'] = round(r['composite'] - BASELINE['composite'], 2)
            r['beats'] = r['composite'] > BASELINE['composite']
            results.append(r)
            print(f"{'★' if r['beats'] else ' '} acc={r['test_acc']} fscore={r['test_fscore']} Δ={r['delta']:+.2f}")
    results.sort(key=lambda x: x['composite'], reverse=True)
    path = os.path.join(results_dir, 'summary_arch_r2.json')
    with open(path, 'w') as f:
        json.dump({'baseline': BASELINE, 'results': results, 'timestamp': datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")

if __name__ == '__main__':
    main()

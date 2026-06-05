#!/usr/bin/env python3
import subprocess, re, json, os
from datetime import datetime

BEST = {'test_acc': 66.97, 'test_fscore': 65.47, 'composite': 132.44}
EXPERIMENTS = [
    {'name': 's10_rescross+logit', 'desc': '残差交叉注意力+Logit集成', 'args': [
        '--fusion_arch', 'residual_cross', '--ensemble_a', '0.3', '--ensemble_v', '0.3']},
    {'name': 's11_rescross+moddrop', 'desc': '残差交叉注意力+模态dropout', 'args': [
        '--fusion_arch', 'residual_cross', '--modality_drop', '0.1']},
    {'name': 's12_rescross+all', 'desc': '残差交叉+Logit+moddrop', 'args': [
        '--fusion_arch', 'residual_cross', '--ensemble_a', '0.25', '--ensemble_v', '0.25', '--modality_drop', '0.1']},
    {'name': 's13_rescross+ens35', 'desc': '残差交叉+更强logit集成0.35', 'args': [
        '--fusion_arch', 'residual_cross', '--ensemble_a', '0.35', '--ensemble_v', '0.35']},
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
            r['delta'] = round(r['composite'] - BEST['composite'], 2)
            results.append(r)
            print(f"acc={r['test_acc']} fscore={r['test_fscore']} comp={r['composite']} Δ={r['delta']:+.2f}")
    results.sort(key=lambda x: x['composite'], reverse=True)
    path = os.path.join(results_dir, 'summary_arch_r3.json')
    with open(path, 'w') as f:
        json.dump({'best_prev': BEST, 'results': results, 'timestamp': datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")

if __name__ == '__main__':
    main()

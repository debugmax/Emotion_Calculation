#!/usr/bin/env python3
"""Internal fusion architecture ablations."""
import subprocess, re, json, os
from datetime import datetime

BASELINE = {'test_acc': 66.97, 'test_fscore': 65.47, 'composite': 132.44}

EXPERIMENTS = [
    {'name': 's0_linear', 'desc': '线性拼接基线', 'args': ['--fusion_arch', 'linear']},
    {'name': 's1_trimodal_attn', 'desc': '内部三模态Token注意力', 'args': ['--fusion_arch', 'trimodal_attn']},
    {'name': 's2_speaker_gate', 'desc': '说话人感知门控融合', 'args': ['--fusion_arch', 'speaker_gate']},
    {'name': 's3_cross_enhance', 'desc': '文本跨模态交叉注意力', 'args': ['--fusion_arch', 'cross_enhance']},
    {'name': 's4_trimodal+logit', 'desc': 'Token注意力+Logit集成', 'args': [
        '--fusion_arch', 'trimodal_attn', '--ensemble_a', '0.3', '--ensemble_v', '0.3']},
    {'name': 's5_speaker+logit', 'desc': '说话人门控+Logit集成', 'args': [
        '--fusion_arch', 'speaker_gate', '--ensemble_a', '0.3', '--ensemble_v', '0.3']},
]

def parse_final(log_path):
    with open(log_path) as f:
        m = re.search(r'FINAL best epoch (-?\d+): test_acc=([\d.]+), test_fscore=([\d.]+), composite=([\d.]+)', f.read())
    return None if not m else {
        'epoch': int(m.group(1)), 'test_acc': float(m.group(2)),
        'test_fscore': float(m.group(3)), 'composite': float(m.group(4)),
    }

def main():
    results_dir = './experiment_logs'
    os.makedirs(results_dir, exist_ok=True)
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
    path = os.path.join(results_dir, 'summary_arch.json')
    with open(path, 'w') as f:
        json.dump({'baseline': BASELINE, 'timestamp': datetime.now().isoformat(), 'results': results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {path}")

if __name__ == '__main__':
    main()

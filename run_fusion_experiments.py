#!/usr/bin/env python3
"""Compare structural fusion innovations vs fixed-weight baseline."""
import subprocess, re, json, os
from datetime import datetime

BASELINE = {'test_acc': 66.97, 'test_fscore': 65.47, 'composite': 132.44}

EXPERIMENTS = [
    {'name': 'f0_fixed', 'desc': '固定权重基线', 'args': ['--fusion_mode', 'fixed']},
    {'name': 'f1_global', 'desc': '可学习全局权重+融合辅助损失', 'args': [
        '--fusion_mode', 'global', '--loss_target', 'text+fused', '--aux_fused_weight', '0.15']},
    {'name': 'f2_utterance', 'desc': 'utterance自适应+融合辅助损失', 'args': [
        '--fusion_mode', 'utterance', '--loss_target', 'text+fused', '--aux_fused_weight', '0.15']},
    {'name': 'f3_dialogue', 'desc': '对话级均值池化+融合辅助损失', 'args': [
        '--fusion_mode', 'dialogue', '--loss_target', 'text+fused', '--aux_fused_weight', '0.15']},
    {'name': 'f4_dialogue_attn', 'desc': '对话级注意力池化+融合辅助损失', 'args': [
        '--fusion_mode', 'dialogue_attn', '--loss_target', 'text+fused', '--aux_fused_weight', '0.15']},
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
            print(f"{'✓' if r['beats'] else '✗'} acc={r['test_acc']} fscore={r['test_fscore']} Δ={r['delta']:+.2f}")

    results.sort(key=lambda x: x['composite'], reverse=True)
    path = os.path.join(results_dir, 'summary_fusion.json')
    with open(path, 'w') as f:
        json.dump({'baseline': BASELINE, 'timestamp': datetime.now().isoformat(), 'results': results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {path}")
    for r in results:
        print(f"  {'★' if r['beats'] else ' '} {r['name']:18s} acc={r['test_acc']:5.2f} fscore={r['test_fscore']:5.2f}  {r['desc']}")

if __name__ == '__main__':
    main()

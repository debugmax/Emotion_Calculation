#!/usr/bin/env python3
"""Batch experiment runner for multimodal fusion ablations."""
import subprocess
import re
import json
import os
from datetime import datetime

BASELINE = {'test_acc': 66.86, 'test_fscore': 65.32, 'composite': 132.18}
ORIGINAL = {'test_acc': 66.02, 'test_fscore': 64.87, 'composite': 130.89}

EXPERIMENTS = [
    {
        'name': 'r4_baseline_repro',
        'desc': '复现第2轮基线',
        'args': [],
    },
    {
        'name': 'r5_ensemble_asym',
        'desc': '非对称集成权重 a=0.15 v=0.25',
        'args': ['--ensemble_a', '0.15', '--ensemble_v', '0.25'],
    },
    {
        'name': 'r6_ensemble_strong',
        'desc': '更强音视频集成 a=0.3 v=0.3',
        'args': ['--ensemble_a', '0.3', '--ensemble_v', '0.3'],
    },
    {
        'name': 'r7_text_fused_aux',
        'desc': '文本主损失 + 融合辅助损失',
        'args': ['--loss_target', 'text+fused', '--aux_fused_weight', '0.2'],
    },
    {
        'name': 'r8_focal_g15',
        'desc': 'Focal Loss gamma=1.5',
        'args': ['--focal_gamma', '1.5'],
    },
    {
        'name': 'r9_focal_g20',
        'desc': 'Focal Loss gamma=2.0',
        'args': ['--focal_gamma', '2.0'],
    },
    {
        'name': 'r10_high_dropout',
        'desc': '更高 dropout=0.6 抑制过拟合',
        'args': ['--dropout', '0.6'],
    },
    {
        'name': 'r11_low_lr',
        'desc': '更低学习率 lr=5e-5',
        'args': ['--lr', '5e-5'],
    },
    {
        'name': 'r12_less_video_kd',
        'desc': '降低视频KD权重 kd_v=0.05',
        'args': ['--kd_weight_v', '0.05'],
    },
    {
        'name': 'r13_combo_strong_dropout',
        'desc': 'r6+r10: ensemble0.3 + dropout0.6',
        'args': ['--ensemble_a', '0.3', '--ensemble_v', '0.3', '--dropout', '0.6'],
    },
    {
        'name': 'r14_combo_strong_lowlr',
        'desc': 'r6+r11: ensemble0.3 + lr5e-5',
        'args': ['--ensemble_a', '0.3', '--ensemble_v', '0.3', '--lr', '5e-5'],
    },
    {
        'name': 'r15_combo_all',
        'desc': 'r6+r10+r11: ensemble0.3 + dropout0.6 + lr5e-5',
        'args': ['--ensemble_a', '0.3', '--ensemble_v', '0.3', '--dropout', '0.6', '--lr', '5e-5'],
    },
    {
        'name': 'r16_ensemble_035',
        'desc': '更强集成 a=0.35 v=0.35',
        'args': ['--ensemble_a', '0.35', '--ensemble_v', '0.35'],
    },
]


def parse_final(log_path):
    with open(log_path) as f:
        content = f.read()
    m = re.search(
        r'FINAL best epoch (-?\d+): test_acc=([\d.]+), test_fscore=([\d.]+), composite=([\d.]+)',
        content
    )
    if not m:
        return None
    return {
        'epoch': int(m.group(1)),
        'test_acc': float(m.group(2)),
        'test_fscore': float(m.group(3)),
        'composite': float(m.group(4)),
    }


def run_one(exp, results_dir):
    log_file = os.path.join(results_dir, f"{exp['name']}.log")
    cmd = [
        'python', 'multimodel_fusion.py',
        '--exp_name', exp['name'],
        '--log_file', log_file,
    ] + exp['args']
    print(f"\n{'='*60}\n>>> {exp['name']}: {exp['desc']}\n>>> {' '.join(cmd)}\n{'='*60}")
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    result = parse_final(log_file)
    if result:
        result['name'] = exp['name']
        result['desc'] = exp['desc']
        result['args'] = exp['args']
        delta = result['composite'] - BASELINE['composite']
        result['delta_vs_r2'] = round(delta, 2)
        result['beats_baseline'] = result['composite'] > BASELINE['composite']
    return result


def main():
    results_dir = './experiment_logs'
    os.makedirs(results_dir, exist_ok=True)
    all_results = []

    for exp in EXPERIMENTS:
        try:
            r = run_one(exp, results_dir)
            if r:
                all_results.append(r)
                tag = '✓ BEAT' if r['beats_baseline'] else '✗'
                print(f"{tag} {r['name']}: acc={r['test_acc']} fscore={r['test_fscore']} "
                      f"composite={r['composite']} (Δ{r['delta_vs_r2']:+.2f})")
        except Exception as e:
            print(f"FAILED {exp['name']}: {e}")
            all_results.append({'name': exp['name'], 'error': str(e)})

    all_results.sort(key=lambda x: x.get('composite', 0), reverse=True)
    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'baseline_r2': BASELINE,
            'original': ORIGINAL,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}\n实验汇总 (按 composite 排序)\n{'='*60}")
    for r in all_results:
        if 'error' in r:
            print(f"  FAIL  {r['name']}")
            continue
        mark = '★' if r['beats_baseline'] else ' '
        print(f"  {mark} {r['name']:22s} acc={r['test_acc']:5.2f} fscore={r['test_fscore']:5.2f} "
              f"comp={r['composite']:6.2f} Δ={r['delta_vs_r2']:+.2f}  {r['desc']}")
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Second-phase experiments: combine best innovations from round 1."""
import subprocess, re, json, os
from datetime import datetime

BASELINE = {'test_acc': 66.86, 'test_fscore': 65.32, 'composite': 132.18}

EXPERIMENTS = [
    {'name': 'r13_combo_strong_dropout', 'desc': 'ensemble0.3 + dropout0.6',
     'args': ['--ensemble_a', '0.3', '--ensemble_v', '0.3', '--dropout', '0.6']},
    {'name': 'r14_combo_strong_lowlr', 'desc': 'ensemble0.3 + lr5e-5',
     'args': ['--ensemble_a', '0.3', '--ensemble_v', '0.3', '--lr', '5e-5']},
    {'name': 'r15_combo_all', 'desc': 'ensemble0.3 + dropout0.6 + lr5e-5',
     'args': ['--ensemble_a', '0.3', '--ensemble_v', '0.3', '--dropout', '0.6', '--lr', '5e-5']},
    {'name': 'r16_ensemble_035', 'desc': 'ensemble a=0.35 v=0.35',
     'args': ['--ensemble_a', '0.35', '--ensemble_v', '0.35']},
]

def parse_final(log_path):
    with open(log_path) as f:
        m = re.search(r'FINAL best epoch (-?\d+): test_acc=([\d.]+), test_fscore=([\d.]+), composite=([\d.]+)', f.read())
    if not m:
        return None
    return {'epoch': int(m.group(1)), 'test_acc': float(m.group(2)),
            'test_fscore': float(m.group(3)), 'composite': float(m.group(4))}

def main():
    results_dir = './experiment_logs'
    os.makedirs(results_dir, exist_ok=True)
    all_results = []
    for exp in EXPERIMENTS:
        log_file = os.path.join(results_dir, f"{exp['name']}.log")
        cmd = ['python', 'multimodel_fusion.py', '--exp_name', exp['name'], '--log_file', log_file] + exp['args']
        print(f"\n>>> {exp['name']}: {exp['desc']}")
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        r = parse_final(log_file)
        if r:
            r.update(exp)
            r['delta_vs_best'] = round(r['composite'] - BASELINE['composite'], 2)
            r['beats_baseline'] = r['composite'] > BASELINE['composite']
            all_results.append(r)
            tag = '✓' if r['beats_baseline'] else '✗'
            print(f"{tag} acc={r['test_acc']} fscore={r['test_fscore']} comp={r['composite']} Δ={r['delta_vs_best']:+.2f}")

    all_results.sort(key=lambda x: x['composite'], reverse=True)
    path = os.path.join(results_dir, 'summary_round2.json')
    with open(path, 'w') as f:
        json.dump({'baseline': BASELINE, 'results': all_results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {path}")
    for r in all_results:
        mark = '★' if r['beats_baseline'] else ' '
        print(f"  {mark} {r['name']:28s} acc={r['test_acc']:5.2f} fscore={r['test_fscore']:5.2f} comp={r['composite']:6.2f}")

if __name__ == '__main__':
    main()

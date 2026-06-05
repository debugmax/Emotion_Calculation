#!/usr/bin/env python3
"""Aggregate MELD ablation experiment results into CSV and bilingual figures."""

import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, 'experiment_logs', 'fonts', 'NotoSansSC-Regular.otf')
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    _CJK_FONT = fm.FontProperties(fname=FONT_PATH).get_name()
else:
    _CJK_FONT = 'SimHei'
LOG_DIR = os.path.join(SCRIPT_DIR, 'experiment_logs')
CSV_PATH = os.path.join(LOG_DIR, 'ablation_results.csv')
FIG_DIR_ZH = os.path.join(LOG_DIR, 'figures', 'zh')
FIG_DIR_EN = os.path.join(LOG_DIR, 'figures', 'en')

ORIGINAL_BASELINE = {'test_acc': 66.02, 'test_fscore': 64.87, 'composite': 130.89}
TUNED_BASELINE = {'test_acc': 66.97, 'test_fscore': 65.47, 'composite': 132.44}
FINAL_BEST = {'test_acc': 67.05, 'test_fscore': 65.52, 'composite': 132.57}

FINAL_RE = re.compile(
    r'FINAL best epoch (-?\d+): test_acc=([\d.]+), test_fscore=([\d.]+), composite=([\d.]+)'
)

GROUP_RULES = [
    ('baseline', lambda n: n in {
        'original_baseline', 'r4_baseline_repro', 'd2_linear', 'f0_fixed', 'log_round2',
    }),
    ('training_strategy', lambda n: any(
        n.startswith(p) for p in ('r10_', 'r11_', 'r12_', 'r13_', 'r14_', 'r15_')
    ) or 'dropout' in n or 'low_lr' in n or 'combo' in n),
    ('loss_ensemble', lambda n: any(
        n.startswith(p) for p in ('r5_', 'r6_', 'r7_', 'r8_', 'r9_', 'r16_')
    ) or 'focal' in n or 'ensemble' in n or 'kd' in n),
    ('logit_fusion', lambda n: n.startswith('f') and len(n) > 1 and n[1].isdigit()),
    ('arch_fusion', lambda n: n.startswith('s') and len(n) > 1 and n[1].isdigit()
                         or 'rescross' in n or 'residual' in n or 'trimodal' in n
                         or 'cross_enhance' in n or 'speaker' in n),
    ('dual_metric', lambda n: n.startswith('d') and len(n) > 1 and (n[1].isdigit() or n.startswith('d11'))),
]

GROUP_LABELS = {
    'zh': {
        'baseline': '基线',
        'training_strategy': '训练策略',
        'loss_ensemble': '损失与集成',
        'logit_fusion': 'Logit融合',
        'arch_fusion': '结构融合',
        'dual_metric': '双指标搜索',
        'final_best': '最终最优',
    },
    'en': {
        'baseline': 'Baseline',
        'training_strategy': 'Training Strategy',
        'loss_ensemble': 'Loss & Ensemble',
        'logit_fusion': 'Logit Fusion',
        'arch_fusion': 'Architecture Fusion',
        'dual_metric': 'Dual-Metric Search',
        'final_best': 'Final Best',
    },
}


def classify_group(name):
    for group, rule in GROUP_RULES:
        if rule(name):
            return group
    if 'final' in name or name == 'd11_confirm':
        return 'dual_metric'
    return 'training_strategy'


def parse_final_line(text):
    matches = FINAL_RE.findall(text)
    if not matches:
        return None
    epoch, acc, fscore, composite = matches[-1]
    return {
        'epoch': int(epoch),
        'test_acc': float(acc),
        'test_fscore': float(fscore),
        'composite': float(composite),
    }


def load_summary_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    rows = []
    for item in data.get('results', []):
        rows.append({
            'group': classify_group(item['name']),
            'name': item['name'],
            'desc': item.get('desc', ''),
            'test_acc': item['test_acc'],
            'test_fscore': item['test_fscore'],
            'composite': item.get('composite', item['test_acc'] + item['test_fscore']),
            'args': json.dumps(item.get('args', []), ensure_ascii=False),
            'source_file': os.path.basename(path),
        })
    return rows


def scan_log_files():
    rows = []
    skip_prefixes = ('run_',)
    for path in sorted(glob.glob(os.path.join(LOG_DIR, '*.log'))):
        basename = os.path.basename(path)
        if basename.startswith(skip_prefixes):
            continue
        with open(path, 'r') as f:
            content = f.read()
        parsed = parse_final_line(content)
        if not parsed:
            continue
        name = os.path.splitext(basename)[0]
        desc = ''
        m = re.search(r'=== Round: (.+) ===', content)
        if m:
            desc = m.group(1)
        rows.append({
            'group': classify_group(name),
            'name': name,
            'desc': desc,
            'test_acc': parsed['test_acc'],
            'test_fscore': parsed['test_fscore'],
            'composite': parsed['composite'],
            'args': '',
            'source_file': basename,
        })
    return rows


def add_manual_entries():
    return [
        {
            'group': 'baseline',
            'name': 'original_baseline',
            'desc': '原始训练日志 epoch3 (log_baseline.txt)',
            'test_acc': ORIGINAL_BASELINE['test_acc'],
            'test_fscore': ORIGINAL_BASELINE['test_fscore'],
            'composite': ORIGINAL_BASELINE['composite'],
            'args': '[]',
            'source_file': 'log_baseline.txt',
        },
        {
            'group': 'dual_metric',
            'name': 'd11_confirm',
            'desc': '最终确认最优 seed3407 ens0.32/0.28',
            'test_acc': FINAL_BEST['test_acc'],
            'test_fscore': FINAL_BEST['test_fscore'],
            'composite': FINAL_BEST['composite'],
            'args': json.dumps({
                'seed': 3407, 'ensemble_a': 0.32, 'ensemble_v': 0.28,
                'fusion_arch': 'linear', 'fusion_mode': 'fixed',
            }, ensure_ascii=False),
            'source_file': 'log',
        },
    ]


def deduplicate(rows):
    by_name = {}
    priority = {'log': 3, 'summary': 1}
    for row in rows:
        name = row['name']
        src = row['source_file']
        score = priority.get('log' if src.endswith('.log') else 'summary', 2)
        if name not in by_name or score >= by_name[name][0]:
            by_name[name] = (score, row)
    return [v[1] for v in sorted(by_name.values(), key=lambda x: x[1]['name'])]


def enrich_deltas(rows):
    for row in rows:
        row['delta_acc'] = round(row['test_acc'] - ORIGINAL_BASELINE['test_acc'], 2)
        row['delta_fscore'] = round(row['test_fscore'] - ORIGINAL_BASELINE['test_fscore'], 2)
    rows.sort(key=lambda r: (-r['composite'], r['name']))
    return rows


def write_csv(rows):
    fields = [
        'group', 'name', 'desc', 'test_acc', 'test_fscore', 'composite',
        'delta_acc', 'delta_fscore', 'args', 'source_file',
    ]
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {CSV_PATH}')


def setup_font(lang):
    if lang == 'zh':
        plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False


def plot_ablation_by_group(rows, lang, out_dir):
    setup_font(lang)
    labels = GROUP_LABELS[lang]
    groups = ['baseline', 'training_strategy', 'loss_ensemble', 'logit_fusion', 'arch_fusion', 'dual_metric']
    best_by_group = {}
    for g in groups:
        items = [r for r in rows if r['group'] == g]
        if items:
            best = max(items, key=lambda r: r['composite'])
            best_by_group[g] = best

    names = [labels[g] for g in groups if g in best_by_group]
    accs = [best_by_group[g]['test_acc'] for g in groups if g in best_by_group]
    fscores = [best_by_group[g]['test_fscore'] for g in groups if g in best_by_group]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    acc_label = 'Test Acc (%)' if lang == 'en' else '测试准确率 (%)'
    f1_label = 'Test F1 (%)' if lang == 'en' else '测试 F1 (%)'
    title = 'Best Result per Ablation Group' if lang == 'en' else '各消融组最优结果'
    ax.bar(x - width / 2, accs, width, label=acc_label, color='#4C72B0')
    ax.bar(x + width / 2, fscores, width, label=f1_label, color='#DD8452')
    ax.axhline(ORIGINAL_BASELINE['test_acc'], color='#4C72B0', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(ORIGINAL_BASELINE['test_fscore'], color='#DD8452', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Score (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(63, 68)
    fig.tight_layout()
    path = os.path.join(out_dir, 'ablation_by_group.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_progress_timeline(lang, out_dir):
    setup_font(lang)
    milestones = [
        ('original', ORIGINAL_BASELINE, 'Original\n66.02/64.87' if lang == 'en' else '原始基线\n66.02/64.87'),
        ('tuned', TUNED_BASELINE, 'Tuned\n66.97/65.47' if lang == 'en' else '调参最优\n66.97/65.47'),
        ('arch', {'test_acc': 66.86, 'test_fscore': 65.65, 'composite': 132.51},
         'residual_cross\n66.86/65.65'),
        ('final', FINAL_BEST, 'Final Best\n67.05/65.52' if lang == 'en' else '最终最优\n67.05/65.52'),
    ]
    x = np.arange(len(milestones))
    accs = [m[1]['test_acc'] for m in milestones]
    fscores = [m[1]['test_fscore'] for m in milestones]
    labels = [m[2] for m in milestones]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, accs, 'o-', color='#4C72B0', linewidth=2, markersize=8, label='Acc' if lang == 'en' else '准确率')
    ax.plot(x, fscores, 's-', color='#DD8452', linewidth=2, markersize=8, label='F1')
    for i, (xi, acc, f1, lb) in enumerate(zip(x, accs, fscores, labels)):
        ax.annotate(f'{acc:.2f}', (xi, acc), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
        ax.annotate(f'{f1:.2f}', (xi, f1), textcoords='offset points', xytext=(0, -12), ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score (%)')
    ax.set_title('Optimization Progress Timeline' if lang == 'en' else '优化进展时间线')
    ax.legend()
    ax.set_ylim(64, 68)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'progress_timeline.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_effective_vs_ineffective(rows, lang, out_dir):
    setup_font(lang)
    effective = [
        ('d11_confirm', 67.05, 65.52, 'seed3407+ens0.32/0.28' if lang == 'en' else 'seed3407+集成0.32/0.28'),
        ('r6_ensemble_strong', 66.86, 65.32, 'Logit ensemble 0.3/0.3' if lang == 'en' else 'Logit集成0.3/0.3'),
        ('r15_combo_all', 66.97, 65.47, 'combo: ens+dropout+lr' if lang == 'en' else '组合:集成+dropout+lr'),
        ('s7_residual_cross', 66.86, 65.65, 'residual_cross arch' if lang == 'en' else 'residual_cross结构'),
    ]
    ineffective = [
        ('r8_focal_g15', 66.48, 64.95, 'Focal Loss g=1.5'),
        ('r7_text_fused_aux', 65.9, 65.09, 'fused aux loss' if lang == 'en' else '融合辅助损失'),
        ('f2_utterance', 64.87, 64.19, 'utterance fusion' if lang == 'en' else 'utterance融合'),
        ('r16_ensemble_035', 66.59, 64.89, 'ensemble 0.35/0.35' if lang == 'en' else '集成0.35/0.35'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, items, title in [
        (axes[0], effective, 'Effective Changes' if lang == 'en' else '有效改动'),
        (axes[1], ineffective, 'Ineffective Changes' if lang == 'en' else '无效改动'),
    ]:
        names = [it[3] for it in items]
        accs = [it[1] for it in items]
        fscores = [it[2] for it in items]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, accs, w, label='Acc' if lang == 'en' else '准确率', color='#55A868')
        ax.bar(x + w / 2, fscores, w, label='F1', color='#C44E52')
        ax.axhline(ORIGINAL_BASELINE['test_acc'], color='#55A868', linestyle='--', alpha=0.4)
        ax.axhline(ORIGINAL_BASELINE['test_fscore'], color='#C44E52', linestyle='--', alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_title(title)
        ax.set_ylim(63, 68)
    axes[0].legend()
    axes[0].set_ylabel('Score (%)')
    fig.suptitle('Effective vs Ineffective Ablations' if lang == 'en' else '有效 vs 无效消融对比', y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'effective_vs_ineffective.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def generate_figures(rows):
    for lang, out_dir in [('zh', FIG_DIR_ZH), ('en', FIG_DIR_EN)]:
        os.makedirs(out_dir, exist_ok=True)
        plot_ablation_by_group(rows, lang, out_dir)
        plot_progress_timeline(lang, out_dir)
        plot_effective_vs_ineffective(rows, lang, out_dir)


def collect_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, 'summary*.json'))):
        rows.extend(load_summary_json(path))
    rows.extend(scan_log_files())
    rows.extend(add_manual_entries())
    rows = deduplicate(rows)
    return enrich_deltas(rows)


def main():
    rows = collect_all()
    write_csv(rows)
    generate_figures(rows)
    print(f'Total experiments: {len(rows)}')


if __name__ == '__main__':
    main()

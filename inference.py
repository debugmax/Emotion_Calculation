#!/usr/bin/env python3
"""Standalone inference for MELD multimodal fusion model."""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

from dataset import MELD_MM_Dataset
from model import Transformer_Based_Model, ModalityFusionHead
from multimodel_fusion import seed_everything, train_or_eval_model


SPLIT_PATHS = {
    'train': './feature/first_stage_train_features.pkl',
    'dev': './feature/first_stage_dev_features.pkl',
    'test': './feature/first_stage_test_features.pkl',
}


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def build_args_from_config(config, cli_overrides=None):
    """Build a namespace-like object from checkpoint config + optional CLI overrides."""
    cfg_args = config.get('args', config)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--temp', type=float, default=2.0)
    parser.add_argument('--clsNum', type=int, default=7)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--kd_weight_a', type=float, default=0.01)
    parser.add_argument('--kd_weight_v', type=float, default=0.08)
    parser.add_argument('--ensemble_a', type=float, default=0.32)
    parser.add_argument('--ensemble_v', type=float, default=0.28)
    parser.add_argument('--fusion_arch', type=str, default='linear')
    parser.add_argument('--residual_alpha', type=float, default=0.1)
    parser.add_argument('--fusion_mode', type=str, default='fixed')
    parser.add_argument('--loss_target', type=str, default='text')
    parser.add_argument('--aux_fused_weight', type=float, default=0.3)
    parser.add_argument('--focal_gamma', type=float, default=0.0)
    parser.add_argument('--modality_drop', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--min_epochs', type=int, default=3)
    parser.add_argument('--exp_name', type=str, default='inference')
    parser.add_argument('--log_file', type=str, default='./log')

    defaults = vars(parser.parse_args([]))
    for key, val in cfg_args.items():
        if key in defaults:
            defaults[key] = val
    if cli_overrides:
        for key, val in cli_overrides.items():
            if val is not None and key in defaults:
                defaults[key] = val
    return argparse.Namespace(**defaults)


def run_inference(args, checkpoint, split, output=None, reference_json=None):
    seed_everything(args.seed)

    feature_path = SPLIT_PATHS[split]
    dataset = MELD_MM_Dataset(feature_path)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=16, collate_fn=dataset.collate_fn,
    )

    model = Transformer_Based_Model(args)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    fusion_head = ModalityFusionHead(
        args.hidden_dim, mode=args.fusion_mode,
        init_a=args.ensemble_a, init_v=args.ensemble_v,
    )
    fusion_head_path = os.path.join(os.path.dirname(checkpoint), 'fusion_head_best.bin')
    if args.fusion_mode != 'fixed' and os.path.exists(fusion_head_path):
        fusion_head.load_state_dict(torch.load(fusion_head_path, map_location='cpu'))

    _, acc, labels, preds, _, fscore, _, _ = train_or_eval_model(
        model, fusion_head, loader, epoch=0, train=False, args=args,
    )

    result = {
        'split': split,
        'accuracy': acc,
        'f1': fscore,
        'labels': labels.tolist() if hasattr(labels, 'tolist') else list(labels),
        'preds': preds.tolist() if hasattr(preds, 'tolist') else list(preds),
    }

    print(f'split={split}, accuracy={acc}, f1={fscore}')

    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'predictions saved to {output}')

    if reference_json and os.path.exists(reference_json):
        with open(reference_json, 'r') as f:
            ref = json.load(f)
        ref_preds = ref['preds']
        pred_match = (list(preds) == ref_preds)
        acc_match = abs(acc - 67.05) < 0.01 if split == 'test' else True
        fscore_match = abs(fscore - 65.52) < 0.01 if split == 'test' else True
        print(f'reference check: preds_match={pred_match}, acc_ok={acc_match}, fscore_ok={fscore_match}')
        if not pred_match:
            mismatches = sum(1 for a, b in zip(preds, ref_preds) if a != b)
            print(f'  mismatched predictions: {mismatches}/{len(ref_preds)}')
            sys.exit(1)
        if split == 'test' and (not acc_match or not fscore_match):
            print(f'  expected acc=67.05, fscore=65.52')
            sys.exit(1)
        print('validation passed')

    return result


def main():
    parser = argparse.ArgumentParser(description='MELD multimodal fusion inference')
    parser.add_argument('--checkpoint', type=str, default='./MELD/save_model/multimodal_fusion_best.bin')
    parser.add_argument('--config', type=str, default='./MELD/save_model/checkpoint_config.json')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--reference', type=str, default='./MELD/save_model/multimodal_fusion_best.json',
                        help='reference JSON for prediction consistency check')
    parser.add_argument('--no-validate', action='store_true', help='skip reference validation')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ensemble_a', type=float, default=None)
    parser.add_argument('--ensemble_v', type=float, default=None)
    parser.add_argument('--fusion_arch', type=str, default=None)
    parser.add_argument('--fusion_mode', type=str, default=None)
    cli = parser.parse_args()

    config = load_config(cli.config)
    overrides = {
        'batch_size': cli.batch_size,
        'seed': cli.seed,
        'ensemble_a': cli.ensemble_a,
        'ensemble_v': cli.ensemble_v,
        'fusion_arch': cli.fusion_arch,
        'fusion_mode': cli.fusion_mode,
    }
    args = build_args_from_config(config, overrides)

    ref = None if cli.no_validate else cli.reference
    run_inference(args, cli.checkpoint, cli.split, cli.output, reference_json=ref)


if __name__ == '__main__':
    main()

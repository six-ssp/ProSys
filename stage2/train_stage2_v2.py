"""CLI for training the ProSys Stage 2 V2 neural ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2.v2.dataset import (
    Stage2CandidateDataLoaderV2,
    Stage2CandidateDatasetV2,
)
from stage2.v2.model import Stage2ModelConfigV2
from stage2.v2.trainer import Stage2TrainConfigV2, run_epoch, train_stage2_model_v2


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the ProSys Stage 2 V2 neural ranker.')
    parser.add_argument('--family_dir', type=str, required=True)
    parser.add_argument('--train_table', type=str, required=True)
    parser.add_argument('--val_table', type=str, required=True)
    parser.add_argument('--test_table', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--route_fpsize', type=int, default=4096)
    parser.add_argument('--product_fpsize', type=int, default=2048)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--slates_per_batch', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--encoder_hidden_dim', type=int, default=256)
    parser.add_argument('--support_hidden_dim', type=int, default=256)
    parser.add_argument('--fusion_hidden_dim', type=int, default=512)
    parser.add_argument('--ranking_hidden_dim', type=int, default=256)
    parser.add_argument('--temperature_hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lambda_rank', type=float, default=1.0)
    parser.add_argument('--lambda_temp', type=float, default=0.2)
    parser.add_argument('--lambda_listmle', type=float, default=1.0)
    parser.add_argument('--lambda_bce', type=float, default=0.5)
    parser.add_argument('--lambda_margin', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--max_negatives', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--disable_product_branch', action='store_true')
    parser.add_argument('--disable_support_features', action='store_true')
    parser.add_argument('--disable_temperature_zscore', action='store_true')
    parser.add_argument('--disable_temp_positive_only', action='store_true')
    parser.add_argument('--max_train_slates', type=int, default=None)
    parser.add_argument('--max_val_slates', type=int, default=None)
    args = parser.parse_args()

    train_dataset = Stage2CandidateDatasetV2(
        candidate_table_file=args.train_table,
        family_dir=args.family_dir,
        route_fpsize=args.route_fpsize,
        product_fpsize=args.product_fpsize,
        radius=args.radius,
        max_slates=args.max_train_slates,
    )
    val_dataset = Stage2CandidateDatasetV2(
        candidate_table_file=args.val_table,
        family_dir=args.family_dir,
        route_fpsize=args.route_fpsize,
        product_fpsize=args.product_fpsize,
        radius=args.radius,
        max_slates=args.max_val_slates,
    )

    train_loader = Stage2CandidateDataLoaderV2(
        train_dataset,
        slates_per_batch=args.slates_per_batch,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = Stage2CandidateDataLoaderV2(
        val_dataset,
        slates_per_batch=args.slates_per_batch,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_config = Stage2ModelConfigV2(
        reagent_dim=train_dataset.reagent_dim,
        solvent_dim=train_dataset.solvent_dim,
        route_fp_dim=args.route_fpsize * 2,
        product_fp_dim=args.product_fpsize,
        encoder_hidden_dim=args.encoder_hidden_dim,
        support_hidden_dim=args.support_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        ranking_hidden_dim=args.ranking_hidden_dim,
        temperature_hidden_dim=args.temperature_hidden_dim,
        dropout=args.dropout,
        use_product_branch=not args.disable_product_branch,
        use_support_features=not args.disable_support_features,
        use_context_embedding=False,
    )
    train_config = Stage2TrainConfigV2(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_rank=args.lambda_rank,
        lambda_temp=args.lambda_temp,
        lambda_listmle=args.lambda_listmle,
        lambda_bce=args.lambda_bce,
        lambda_margin=args.lambda_margin,
        margin=args.margin,
        max_negatives=args.max_negatives,
        temperature_zscore_by_family=not args.disable_temperature_zscore,
        temp_positive_only=not args.disable_temp_positive_only,
        device=args.device,
    )

    result = train_stage2_model_v2(
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        model_config=model_config,
        train_config=train_config,
        train_frame=train_dataset.frame,
    )

    if args.test_table:
        import torch

        from stage2.v2.trainer import TemperatureStats
        from stage2.v2.model import Stage2NeuralRankerV2

        test_dataset = Stage2CandidateDatasetV2(
            candidate_table_file=args.test_table,
            family_dir=args.family_dir,
            route_fpsize=args.route_fpsize,
            product_fpsize=args.product_fpsize,
            radius=args.radius,
        )
        test_loader = Stage2CandidateDataLoaderV2(
            test_dataset,
            slates_per_batch=args.slates_per_batch,
            shuffle=False,
            num_workers=args.num_workers,
        )

        state = torch.load(result['best_checkpoint'], map_location=args.device)
        model = Stage2NeuralRankerV2(model_config).to(args.device)
        model.load_state_dict(state['state_dict'])
        test_metrics = run_epoch(
            model,
            test_loader,
            None,
            train_config,
            TemperatureStats(**state['temperature_stats']),
        )
        result['test'] = test_metrics
        summary_path = Path(args.output_dir) / 'train_summary.json'
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

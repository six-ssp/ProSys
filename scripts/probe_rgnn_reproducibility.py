#!/usr/bin/env python3
"""Repeated same-seed R-GNN fits: uncontrolled CUDA vs deterministic algorithms."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'Experiment/project_completion_20260913/temperature_probe'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('default', 'deterministic'))
    p.add_argument('--repeat', type=int)
    args = p.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.mode:
        import numpy as np
        import torch
        from stage3_XGBoost.reaction_gnn_features import ReactionGNNConfig, train_reaction_gnn_feature_model
        from prosys_shared.mainline import split_file_for_family
        from scripts.run_stage1_multiseed import write, sha
        output = STUDY / f'{args.mode}_{args.repeat}'
        if output.exists():
            raise FileExistsError(output)
        deterministic = args.mode == 'deterministic'
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        source = ROOT / 'Experiment/mainline_evidence_completion_20260913/compact/seed_0/Beckmann/bundle/rgnn/reaction_gnn_meta.json'
        config = ReactionGNNConfig(**json.loads(source.read_text())['config'])
        train, val = [split_file_for_family(ROOT, 'Beckmann', s) for s in ('train', 'val')]
        metadata = train_reaction_gnn_feature_model(train, val, output, config=config, force_retrain=True)
        state = torch.load(output / 'reaction_gnn.pt', map_location='cpu', weights_only=False)['model_state']
        tensor_hash = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            tensor_hash.update(name.encode())
            tensor_hash.update(tensor.numpy().tobytes())
        write(output / 'probe.json', {'mode': args.mode, 'repeat': args.repeat,
              'state_tensor_sha256': tensor_hash.hexdigest(), 'best_val_loss': metadata['best_val_loss'],
              'train_sha256': sha(train), 'val_sha256': sha(val), 'config': config.to_dict(),
              'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
              'threads': torch.get_num_threads(), 'cublas_workspace_config': os.getenv('CUBLAS_WORKSPACE_CONFIG'),
              'source_sha256': sha(ROOT / 'stage3_XGBoost/reaction_gnn_features.py')})
        return
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
    for mode in ('default', 'deterministic'):
        for repeat in (0, 1):
            if (STUDY / f'{mode}_{repeat}/probe.json').exists():
                continue
            with (STUDY / f'{mode}_{repeat}.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__)), '--mode', mode, '--repeat', str(repeat)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    from scripts.run_stage1_multiseed import write
    rows = [json.loads((STUDY / f'{m}_{r}/probe.json').read_text())
            for m in ('default', 'deterministic') for r in (0, 1)]
    write(STUDY / 'comparison.json', {'runs': rows,
          'default_same_seed_tensor_equal': rows[0]['state_tensor_sha256'] == rows[1]['state_tensor_sha256'],
          'deterministic_same_seed_tensor_equal': rows[2]['state_tensor_sha256'] == rows[3]['state_tensor_sha256'],
          'scope': 'Beckmann seed0 current-environment repeated encoder training only; not proof of historical drift cause'})
    print(json.dumps(json.loads((STUDY / 'comparison.json').read_text()), indent=2))


if __name__ == '__main__':
    main()

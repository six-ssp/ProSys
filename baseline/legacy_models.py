"""Legacy Stage 2 model wrappers with torch>=2.6 checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from Experiment.legacy_stage2.evaluate_model.eval_utils import MultiTask_Evaluator, Ranking_Evaluator, sort_string
from Experiment.legacy_stage2.train_multilabel.args_train import TrainArgs_rxn
from Experiment.legacy_stage2.train_multilabel.model_utils import Multitask_Multilabel, ReactionModel_LWTemp


class LegacyMultiTaskEvaluator(MultiTask_Evaluator):
    def load_model(
        self,
        model_path: str | Path,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    ):
        state = torch.load(str(model_path), map_location=device, weights_only=False)
        self.args_MT = TrainArgs_rxn()
        self.args_MT.from_dict(vars(state['args']), skip_unsettable=True)
        self.args_MT.device = device

        self.model = Multitask_Multilabel(self.args_MT)
        self.model.to(device)
        self.model.load_state_dict(state['state_dict'])
        self.model.eval()

    def make_input_rxn_condition(self, rxn_fp):
        """Legacy helper with faster tensor materialization for inference-time enumeration."""
        rxn_fp = rxn_fp.to(self.model.device)
        enumerated_features = self.enumerate_combinations(rxn_fp)
        enumerated_solvent, enumerated_reagent = list(zip(*enumerated_features))
        solvent_array = np.stack(enumerated_solvent).astype(np.float32, copy=False)
        reagent_array = np.stack(enumerated_reagent).astype(np.float32, copy=False)
        return torch.from_numpy(solvent_array), torch.from_numpy(reagent_array)


class LegacyRankingEvaluator(Ranking_Evaluator):
    def load_model(
        self,
        model_path: str | Path,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    ):
        self.device = device
        state = torch.load(str(model_path), map_location=device, weights_only=False)
        self.args_LW = TrainArgs_rxn()
        self.args_LW.from_dict(vars(state['args']), skip_unsettable=True)
        self.args_LW.device = device

        self.model = ReactionModel_LWTemp(
            self.args_LW,
            len_solvent=len(self.solvent_classes),
            len_reagent=len(self.reagent_classes),
        )
        self.model.to(device)
        self.model.load_state_dict(state['state_dict'])
        self.model.eval()

    def rank_top_contexts(self, rxn_fp, input_solvents, input_reagents, top_k: int | None = None):
        scores, temperatures = self.predict_scores(rxn_fp, input_solvents, input_reagents)
        if top_k is None or top_k >= int(scores.numel()):
            top_index = torch.argsort(scores, descending=True)
        else:
            _, top_index = torch.topk(scores, k=int(top_k), largest=True, sorted=True)

        top_index_cpu = top_index.detach().cpu()
        contexts = self.make_contexts(input_solvents[top_index_cpu], input_reagents[top_index_cpu])

        ranked = []
        for rank_pos, global_index in enumerate(top_index_cpu.tolist()):
            ranked.append(
                sort_string(contexts[rank_pos])
                + [float(temperatures[global_index])]
                + [float(scores[global_index].detach().cpu())]
            )
        return ranked

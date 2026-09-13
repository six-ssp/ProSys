"""Isolated no-KNN proposal/evidence intervention; mainline defaults unchanged."""

from stage2_ReaFNN.knn_condition_selector import KNNContextPoolBuilder


KNN_EVIDENCE_COLUMNS = (
    "from_baseline_knn", "knn_similarity_sum", "knn_similarity_max",
    "knn_neighbor_count", "knn_weighted_mean_yield", "stage2_knn_rank",
    "stage2_knn_prior", "stage2_knn_score",
)


class ReaFNNOnlyPoolBuilder(KNNContextPoolBuilder):
    """Retain neural historical proposals without any KNN lookup or evidence.

    Shared training reaction/context indices remain necessary for canonical
    leave-one-reaction-out context-frequency adjustment. They are not queried
    by molecular similarity in this intervention.
    """

    def _aggregate_knn_contexts(self, record, *, limit, leave_one_reaction_out=False):
        return []

    def _route_similarities(self, query_fp):
        raise AssertionError("ReaFNN-only must not perform KNN similarity lookup")

    def _independent_post_fusion_state(self, record, *, leave_one_reaction_out):
        rows = super()._independent_post_fusion_state(
            record, leave_one_reaction_out=leave_one_reaction_out)
        for row in rows:
            # The shared mainline populates a global-yield fallback in one
            # KNN-named field. Remove it here; neural/global priors remain.
            for field in KNN_EVIDENCE_COLUMNS:
                row[field] = 0.0
        return rows

    def _post_fusion_protocol(self):
        protocol = super()._post_fusion_protocol()
        if protocol["weight_grid"] != [0.0]:
            raise ValueError("ReaFNN-only requires fixed w=0, not validation tuning")
        protocol["intervention"] = "remove_knn_proposals_and_evidence"
        protocol["knn_retrieval_enabled"] = False
        return protocol

    def _select_independent_post_fusion_contexts(self, state, *, knn_weight):
        if float(knn_weight) != 0.0:
            raise ValueError("ReaFNN-only cannot use a nonzero KNN weight")
        return super()._select_independent_post_fusion_contexts(state, knn_weight=0.0)

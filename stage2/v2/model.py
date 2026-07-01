"""Neural ranker and temperature model for ProSys Stage 2 V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .constants import (
    CONTEXT_DENSE_COLUMNS_V2,
    PRODUCT_DESCRIPTOR_DIM_V2,
    ROUTE_DENSE_COLUMNS_V2,
    ROUTE_GRAPH_DIM_V2,
    SUPPORT_FEATURE_COLUMNS_V2,
)


@dataclass
class Stage2ModelConfigV2:
    reagent_dim: int
    solvent_dim: int
    route_fp_dim: int = 8192
    route_graph_dim: int = ROUTE_GRAPH_DIM_V2
    route_dense_dim: int = len(ROUTE_DENSE_COLUMNS_V2)
    context_dense_dim: int = len(CONTEXT_DENSE_COLUMNS_V2)
    product_fp_dim: int = 2048
    product_feat_dim: int = PRODUCT_DESCRIPTOR_DIM_V2
    support_feature_size: int = len(SUPPORT_FEATURE_COLUMNS_V2)
    encoder_hidden_dim: int = 256
    support_hidden_dim: int = 256
    fusion_hidden_dim: int = 512
    ranking_hidden_dim: int = 256
    temperature_hidden_dim: int = 256
    dropout: float = 0.1
    use_product_branch: bool = True
    use_support_features: bool = True
    use_context_embedding: bool = False
    use_route_graph_features: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouteEncoder(nn.Module):
    def __init__(self, config: Stage2ModelConfigV2):
        super().__init__()
        input_dim = config.route_fp_dim + config.route_dense_dim
        if config.use_route_graph_features:
            input_dim += config.route_graph_dim
        self.use_route_graph_features = config.use_route_graph_features
        self.encoder = MLPBlock(input_dim, config.encoder_hidden_dim, config.dropout)

    def forward(
        self,
        route_fp: torch.Tensor,
        route_dense_features: torch.Tensor,
        route_graph_features: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [route_fp, route_dense_features]
        if self.use_route_graph_features:
            pieces.append(route_graph_features)
        return self.encoder(torch.cat(pieces, dim=1))


class ProductEncoder(nn.Module):
    def __init__(self, config: Stage2ModelConfigV2):
        super().__init__()
        input_dim = config.product_fp_dim + config.product_feat_dim
        self.enabled = config.use_product_branch
        self.hidden_dim = config.encoder_hidden_dim
        self.encoder = MLPBlock(input_dim, config.encoder_hidden_dim, config.dropout)

    def forward(self, product_fp: torch.Tensor, product_features: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(
                product_fp.shape[0],
                self.hidden_dim,
                device=product_fp.device,
                dtype=product_fp.dtype,
            )
        return self.encoder(torch.cat((product_fp, product_features), dim=1))


class ContextEncoder(nn.Module):
    def __init__(self, config: Stage2ModelConfigV2):
        super().__init__()
        if config.use_context_embedding:
            raise ValueError('use_context_embedding=True is reserved for a later V2 upgrade path.')

        half_hidden = config.encoder_hidden_dim // 2
        self.reagent_encoder = MLPBlock(
            config.reagent_dim,
            half_hidden,
            config.dropout,
        )
        self.solvent_encoder = MLPBlock(
            config.solvent_dim,
            half_hidden,
            config.dropout,
        )
        self.dense_encoder = MLPBlock(
            config.context_dense_dim,
            config.encoder_hidden_dim,
            config.dropout,
        )
        self.output = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim * 2, config.encoder_hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        reagent_features: torch.Tensor,
        solvent_features: torch.Tensor,
        context_dense_features: torch.Tensor,
    ) -> torch.Tensor:
        reagent_hidden = self.reagent_encoder(reagent_features)
        solvent_hidden = self.solvent_encoder(solvent_features)
        dense_hidden = self.dense_encoder(context_dense_features)
        context_hidden = torch.cat((reagent_hidden, solvent_hidden), dim=1)
        return self.output(torch.cat((context_hidden, dense_hidden), dim=1))


class SupportEncoder(nn.Module):
    def __init__(self, config: Stage2ModelConfigV2):
        super().__init__()
        self.enabled = config.use_support_features
        self.hidden_dim = config.encoder_hidden_dim
        self.encoder = MLPBlock(
            config.support_feature_size,
            config.support_hidden_dim,
            config.dropout,
        )
        self.proj = nn.Linear(config.support_hidden_dim, config.encoder_hidden_dim)

    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(
                support_features.shape[0],
                self.hidden_dim,
                device=support_features.device,
                dtype=support_features.dtype,
            )
        return self.proj(self.encoder(support_features))


class FusionBlock(nn.Module):
    def __init__(self, config: Stage2ModelConfigV2):
        super().__init__()
        input_dim = config.encoder_hidden_dim * 8
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h_route: torch.Tensor,
        h_product: torch.Tensor,
        h_context: torch.Tensor,
        h_support: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [
            h_route,
            h_product,
            h_context,
            h_support,
            h_route * h_context,
            h_product * h_context,
            torch.abs(h_route - h_context),
            torch.abs(h_product - h_context),
        ]
        return self.net(torch.cat(pieces, dim=1))


class RankingTrunk(nn.Module):
    def __init__(self, config: Stage2ModelConfigV2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.ranking_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(config.ranking_hidden_dim, 1)

    def forward(self, h_fused: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(h_fused)).squeeze(-1)


class TemperatureTrunk(nn.Module):
    def __init__(self, config: Stage2ModelConfigV2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.temperature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(config.temperature_hidden_dim, 1)

    def forward(self, h_fused: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(h_fused)).squeeze(-1)


class Stage2NeuralRankerV2(nn.Module):
    def __init__(self, config: Stage2ModelConfigV2):
        super().__init__()
        self.config = config
        self.route_encoder = RouteEncoder(config)
        self.product_encoder = ProductEncoder(config)
        self.context_encoder = ContextEncoder(config)
        self.support_encoder = SupportEncoder(config)
        self.fusion = FusionBlock(config)
        self.ranking = RankingTrunk(config)
        self.temperature = TemperatureTrunk(config)

    def forward(
        self,
        *,
        route_fp: torch.Tensor,
        route_graph_features: torch.Tensor,
        route_dense_features: torch.Tensor,
        context_dense_features: torch.Tensor,
        product_fp: torch.Tensor,
        product_features: torch.Tensor,
        reagent_features: torch.Tensor,
        solvent_features: torch.Tensor,
        support_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_route = self.route_encoder(route_fp, route_dense_features, route_graph_features)
        h_product = self.product_encoder(product_fp, product_features)
        h_context = self.context_encoder(reagent_features, solvent_features, context_dense_features)
        h_support = self.support_encoder(support_features)
        h_fused = self.fusion(h_route, h_product, h_context, h_support)

        score_logit = self.ranking(h_fused)
        temperature_pred = self.temperature(h_fused)
        return {
            'score_logit': score_logit,
            'score_prob': torch.sigmoid(score_logit),
            'temperature_pred': temperature_pred,
        }

"""Feature extraction and score engine package.

Phase 2 of the pipeline: converts raw BuildingState metrics
into typed feature sets and deterministic 0-100 scores per zone.

Sub-modules
-----------
features : ZoneFeatureVector — labelled qualitative features
           derived from raw state metrics.
scores   : ZoneScoreVector + BuildingScoreReport — numeric
           0-100 scores as defined in architecture Section 7.
"""

from src.engine.features import (
    AirQualityFeature,
    ComfortFeature,
    EnergyFeature,
    ZoneFeatureVector,
    extract_zone_features,
)
from src.engine.scores import (
    BuildingScoreReport,
    ZoneScoreVector,
    score_building,
    score_zone,
)

__all__ = [
    # Feature types
    "ComfortFeature",
    "AirQualityFeature",
    "EnergyFeature",
    "ZoneFeatureVector",
    "extract_zone_features",
    # Score types
    "ZoneScoreVector",
    "BuildingScoreReport",
    "score_zone",
    "score_building",
]

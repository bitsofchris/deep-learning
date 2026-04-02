"""Feature registry — maps feature names to their generators.

Autoresearch and config.yaml reference features by name.
This registry resolves names to FeatureGenerator instances.

To register a new feature, import the generator and add it to REGISTRY.
"""

from .base import FeatureGenerator

# Feature name → FeatureGenerator instance
# Add new features here as they're implemented:
#
#   from .volatility import RollingVolatility
#   from .fourier import DominantFrequency
#
#   REGISTRY: dict[str, FeatureGenerator] = {
#       "volatility_5d": RollingVolatility(window=5),
#       "volatility_20d": RollingVolatility(window=20),
#       "volatility_60d": RollingVolatility(window=60),
#       "fourier_dominant_freq": DominantFrequency(),
#   }

REGISTRY: dict[str, FeatureGenerator] = {}


def get_feature(name: str) -> FeatureGenerator:
    """Look up a feature generator by name."""
    if name not in REGISTRY:
        available = list(REGISTRY.keys()) or ["(none registered yet)"]
        raise KeyError(f"Unknown feature '{name}'. Available: {available}")
    return REGISTRY[name]


def list_features() -> list[str]:
    """List all registered feature names."""
    return list(REGISTRY.keys())

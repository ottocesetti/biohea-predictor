# BioHEA-Predictor

**Physics-informed machine learning framework for biomedical high-entropy alloy discovery**

BioHEA-Predictor is a scientific machine learning framework designed to support the discovery and screening of **high-entropy alloys (HEAs)** for biomedical and structural applications.

The project focuses on reducing experimental trial-and-error by combining:

- compositional data analysis,
- physics-informed descriptors,
- CALPHAD-assisted thermodynamic features,
- machine learning models,
- applicability-domain estimation,
- feature diagnostics,
- and uncertainty-aware candidate screening.

The framework is intended to help prioritize alloy compositions with desirable combinations of phase stability, mechanical compatibility, and reduced experimental risk.

---

## Overview

Designing biomedical HEAs is challenging because the compositional space is extremely large and experimental validation is expensive.

BioHEA-Predictor aims to assist this process by predicting and analyzing properties such as:

- elastic modulus;
- phase formation tendency;
- brittle/intermetallic phase risk;
- thermodynamic stability indicators;
- applicability-domain risk;
- composition novelty;
- physically meaningful descriptor behavior.

The framework is predictive and decision-support oriented. It does not replace experimental validation, but helps identify which compositions are more promising to test first.

---

## Motivation

Metallic biomaterials require a careful balance between:

- low or moderate elastic modulus;
- mechanical integrity;
- phase stability;
- corrosion resistance potential;
- biocompatible element selection;
- avoidance of brittle phases;
- reduced cost and experimental effort.

High-entropy alloys offer a large design space, but this also creates a major optimization problem.

This project explores how **physics-informed machine learning** can be used to navigate that space more efficiently.

The central goal is:

> To reduce time, cost, and experimental trial-and-error in alloy discovery by prioritizing compositions with higher predicted potential and lower methodological risk.

---

## Current Scope

The current framework supports:

- composition-based prediction;
- physics-informed feature generation;
- thermodynamic descriptor integration;
- phase/properties modeling;
- applicability-domain estimation;
- latent-space diagnostics;
- feature redundancy analysis;
- leakage-aware validation design;
- future extension toward inverse design and active learning.

---

## Scientific Design Principles

BioHEA-Predictor is built around the idea that alloy compositions should not be treated as ordinary Euclidean vectors.

The framework explicitly considers:

### 1. Compositional Geometry

Alloy compositions live in a simplex:

x₁ + x₂ + ... + xₙ = 1

Therefore, the framework includes compositional preprocessing and ILR-based geometric representations to reduce artifacts caused by naïve composition handling.

2. Physics-Informed Features

Instead of relying only on raw elemental fractions, the model incorporates physically meaningful descriptors, including:

atomic size mismatch, δ;
valence electron concentration, VEC;
mixing enthalpy, ΔH_mix;
configurational entropy, ΔS_mix;
stability parameter, Ω;
melting-temperature-related descriptors;
physically motivated interaction terms.

These descriptors encode metallurgical assumptions into the learning process.

3. CALPHAD-Assisted Thermodynamics

The framework includes a Thermo-Calc/CALPHAD-oriented interface for thermodynamic descriptors such as:

Gibbs free energy;
enthalpy;
entropy;
predicted phase fractions;
thermodynamic failure flags;
solver-status metadata.

CALPHAD outputs are treated as simulated thermodynamic features, not experimental measurements.

4. Applicability Domain

The framework estimates whether a composition is inside or outside the region where the model has support.

Applicability-domain estimation is performed across multiple spaces:

ILR compositional geometry;
physical descriptor space;
latent manifold space.

The model reports raw scores, empirical percentile risk, and tail-risk indicators.

5. Leakage-Aware Validation

The framework is designed to avoid common machine learning leakage issues, including:

fitting latent transformations on the full dataset;
using test information during feature engineering;
misaligned train/test feature columns;
silently reusing cached thermodynamic features for different compositions;
treating failed thermodynamic calculations as valid physical values.
Methodology

The framework follows a modular architecture.

Composition
    ↓
Simplex validation and closure
    ↓
ILR compositional geometry
    ↓
Physics-informed descriptors
    ↓
CALPHAD-assisted thermodynamic descriptors
    ↓
Physically motivated interaction features
    ↓
Latent representation
    ↓
Predictive model
    ↓
Applicability-domain and diagnostic reports
Core Components
Feature Engineering
GeometricFeatureBuilder
PhysicalDescriptorBuilder
ThermodynamicDescriptorBuilder
PhysicsInteractionGenerator
LatentFeatureBuilder
Thermodynamic Layer
canonical thermodynamic descriptor handling;
cache key based on composition and thermodynamic regime;
explicit failure taxonomy;
metadata tracking for reproducibility.
Applicability Domain
Mahalanobis-based compositional and physical-space assessment;
latent-space novelty detection;
empirical percentile calibration;
extreme-tail risk estimation.
Diagnostics

The framework includes diagnostic tools for:

feature redundancy;
correlation and collinearity;
VIF analysis;
PCA collapse;
mutual information;
permutation importance;
SHAP-based importance;
latent dominance;
latent/physical shortcut detection;
thermodynamic missingness;
UMAP/latent-space stability.
Example Usage
from framework.HEAFrameworkV18_2 import HEAFrameworkV18_2
from thermodynamics.thermocalc_engine import ThermoCalcEngine

# Example objects expected by the framework
element_data = ...
binary_mixing_enthalpies = ...

engine = ThermoCalcEngine(
    cache_dir="cache/thermocalc",
    database="TCHEA5",
    database_version="unknown",
    thermocalc_version="unknown"
)

model = HEAFrameworkV18_2(
    element_data=element_data,
    thermocalc_engine=engine,
    binary_mixing_enthalpies=binary_mixing_enthalpies,
    latent_method="umap",
    latent_dimensions=8
)

X = {
    "composition": composition_dataframe,
    "processing": processing_dataframe
}

model.fit(X, y_phase)

result = model.predict(X_new)

print(result["phase_prediction"])
print(result["applicability"])
Legacy Usage Example

Earlier versions followed a simpler interface:

from src.model import HEAPredictor

predictor = HEAPredictor()

predictor.train(
    X_data,
    y_modulus,
    y_laves
)

sample = {
    "Ti": 50,
    "Nb": 20,
    "Ta": 10,
    "Zr": 10,
    "Mo": 5,
    "Fe": 5
}

result = predictor.predict_from_dict(sample)

print(result)

The current version expands this idea into a more complete scientific framework with stronger feature engineering, validation, and applicability-domain analysis.

Validation Strategy

The framework supports validation strategies such as:

train/test split;
k-fold cross-validation;
out-of-sample cascade predictions;
leakage-safe latent feature generation;
cluster-aware validation;
leave-system-out validation;
ablation studies;
applicability-domain diagnostics.

Recommended metrics include:

Regression
R²;
MAE;
RMSE;
calibration error;
prediction interval coverage.
Classification
accuracy;
balanced accuracy;
ROC-AUC;
precision;
recall;
F1-score;
calibration curves.
Discovery-Oriented Metrics
hit rate@k;
experiments-to-target;
cost-normalized gain;
Pareto hypervolume;
out-of-domain failure rate.
Project Structure
BioHEA-Predictor/
│
├── core/
│   ├── simplex.py
│   ├── geometric_features.py
│   ├── physical_descriptors.py
│   ├── thermodynamic_descriptors.py
│   ├── interaction_features.py
│   ├── latent_features.py
│   └── feature_pipeline.py
│
├── thermodynamics/
│   └── thermocalc_engine.py
│
├── models/
│   ├── compositional_phase_model.py
│   └── applicability_domain.py
│
├── validation/
│   ├── feature_diagnostics.py
│   ├── latent_diagnostics.py
│   ├── ad_diagnostics.py
│   ├── thermo_missingness.py
│   └── interaction_discovery.py
│
├── framework/
│   └── HEAFrameworkV18_2.py
│
├── data/
│   └── datasets and curated composition tables
│
├── notebooks/
│   └── exploratory analysis and experiments
│
├── results/
│   └── metrics, plots, and candidate reports
│
├── models_trained/
│   └── saved trained models
│
└── README.md
Current Status

This project is under active development.

Implemented:

physics-informed descriptors;
compositional geometry handling;
thermodynamic descriptor layer;
leakage-aware feature pipeline;
applicability-domain estimation;
latent-space diagnostics;
feature diagnostics;
thermodynamic missingness tracking.

In development:

inverse alloy design;
candidate composition generator;
active learning;
multi-objective optimization;
Pareto-based candidate selection;
cost-aware alloy screening;
experimental validation workflow.
Limitations

This framework is exploratory and research-oriented.

Current limitations include:

limited dataset size;
possible dataset bias;
no large-scale experimental validation yet;
CALPHAD dependence on database quality;
uncertainty estimates still require further calibration;
latent embeddings may be unstable in small datasets;
thermodynamic simulations should not be treated as experimental ground truth.

Predictions should be interpreted as scientific guidance, not final experimental confirmation.

Future Work

Planned extensions include:

active learning for alloy candidate selection;
inverse design based on user-defined target properties;
genetic algorithms and Bayesian optimization;
multi-objective alloy design;
cost-aware candidate ranking;
integration with Thermo-Calc workflows;
integration with experimental feedback loops;
validation using real biomedical HEA datasets;
closed-loop discovery of Ti-Zr-Nb-Ta-based biomedical alloys.
Scientific Positioning

Its main objective is to provide a transparent and physically informed decision-support framework for alloy discovery.

Author

Otto Amaral Cesetti

GitHub: @ottocesetti

Disclaimer

This project is currently a research prototype.

It is not intended for direct clinical, biomedical, industrial, or safety-critical use without independent experimental validation.

All predictions should be validated by appropriate thermodynamic simulations, mechanical testing, phase characterization, and biomedical compatibility studies.

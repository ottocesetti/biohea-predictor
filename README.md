# biohea-predictor

## Overview
BioHEA-Predictor is a machine learning-based tool designed to support the discovery of High Entropy Alloys (HEAs) for biomedical applications.

The model predicts:
- Elastic modulus (GPa)
- Probability of brittle phase formation (Laves phase)

The goal is to reduce experimental trial-and-error by prioritizing promising alloy compositions.

---

## Motivation

Designing metallic biomaterials is challenging due to:

- High experimental cost
- Complex multi-element compositional space
- Trade-off between mechanical compatibility and structural integrity

This project explores how data-driven methods can assist in navigating this space.

---

## Methodology

The model follows a **hybrid cascaded architecture**:

1. **Classification Stage**
   - Predicts the likelihood of Laves phase formation

2. **Regression Stage**
   - Predicts elastic modulus
   - Uses classification output as an additional feature

### Physics-Informed Features

Instead of using only raw composition, the model incorporates:

- Atomic size mismatch (δ)
- Valence electron concentration (VEC)
- Mixing entropy (ΔS_mix)
- Stability parameter (Ω)

These descriptors introduce physical meaning into the learning process.

---

## Validation Strategy

To ensure robustness:

- Train/Test split (80/20)
- Cross-validation (k-fold)
- Out-of-sample predictions (`cross_val_predict`) used in cascade

Metrics:
- Regression: R², MAE
- Classification: Accuracy, AUC

---

## Important Note

This model is **predictive, not causal**.

Due to limited dataset size, results should be interpreted as **exploratory**.  
Experimental validation is required for real-world applications.

## Development Environment

This project was initially developed and tested using Google Colab for rapid prototyping and experimentation.

---

## Example Usage

```python
from src.model import HEAPredictor

predictor = HEAPredictor()
predictor.train(X_data, y_modulus, y_laves)

sample = {
    'Ti': 50, 'Nb': 20, 'Ta': 10,
    'Zr': 10, 'Mo': 5, 'Fe': 5
}

result = predictor.predict_from_dict(sample)
print(result)



#Project Structure
#src/ → core implementation
#data/ → datasets
#models/ → trained models
#notebooks/ → exploratory analysis
#results/ → metrics and visualizations
#Limitations
#Small dataset size
#No experimental validation yet
#Potential model bias due to limited data diversity
#Future Work
#Integration with optimization algorithms (e.g., genetic algorithms)
#Expansion with real experimental datasets
#Multi-objective alloy design
#Integration with simulation tools (e.g., Thermo-Calc)
#Author

#Otto Amaral Cesetti

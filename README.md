# Wine Quality Prediction using Machine Learning

A production-ready machine learning project designed to evaluate and predict the sensory quality of red and white wine based on chemical engineering attributes. This repository contains the data ingestion pipeline, feature transformations, comparative model optimization, and deployment scripts (`app.py`).

## Features & Preprocessing Pipeline

The application ingests raw physicochemical test summaries and processes them through an end-to-end data pipeline before executing classification tasks:

- **Exploratory Data Analysis (EDA):** Correlation profiling using Seaborn matrix plots to capture positive/negative feature impacts.
- **Handling Class Imbalances:** Outlier detection and evaluation adjustments since average wine profiles drastically outnumber extreme poor or excellent variants.
- **Feature Scaling:** Normalization/Standardization using Scikit-Learn to ensure distance-based metrics (like SVM) treat chemical variables equally.
- **Multi-Model Evaluation:** Comparative execution benchmarking across four distinct classification and regression algorithms.

## Machine Learning Pipeline Architecture

The flowchart below traces the processing workflow from importing raw tabular properties down to delivering a final wine quality grade classification:

```text
[ Raw Wine Dataset CSV ]
           │
           ▼
┌──────────────────────────┐
│   Data Cleaning & EDA    │ ──► Drops nulls, charts feature correlations
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Feature Transformation  │ ──► Separates labels (Quality) from metrics
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│   Train / Test Split     │ ──► Splitting strategy (e.g., 80% Train / 20% Test)
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│     Feature Scaling      │ ──► StandardScaler transformation
└──────────────────────────┘
           │
           ├───► [ Logistic Regression ] ──┐
           ├───► [ Random Forest ] ────────┼──► GridSearch / Evaluation
           ├───► [ Decision Tree ] ────────┼──► Hyperparameter tuning
           └───► [ Support Vector Machine ]┘
                                           │
                                           ▼
                            ┌──────────────────────────┐
                            │    Model Performance     │
                            └──────────────────────────┘
                                           │
                                           ▼
                            [ Production Deploy (app.py) ]
```

## Dataset Specifications

The application trains on the **Kaggle Wine Quality Dataset** (originally sourced from the UCI Machine Learning Repository, referencing the Portuguese "Vinho Verde" studies by Cortez et al.).

### Input Variables (Physicochemical Metrics)
1. **Fixed Acidity:** Concentrations of non-volatile tartaric acids (g/dm³).
2. **Volatile Acidity:** Acetic acid concentration (g/dm³); higher limits yield an undesirable vinegar taste.
3. **Citric Acid:** Small flavor additives contributing freshness attributes to the wine profile.
4. **Residual Sugar:** Remaining sugars measured after fermentation processes complete (g/dm³).
5. **Chlorides:** Summary salt presence indicators (g/dm³).
6. **Free Sulfur Dioxide:** Dissolved SO₂ gas preventing active microbial oxidation.
7. **Total Sulfur Dioxide:** Absolute volume of both bound and free SO₂ chemical elements.
8. **Density:** Fluid volume weight metrics varying directly against alcohol and sugar ratios.
9. **pH:** Logarithmic evaluation of systemic acidity levels ranging from 0 (highly acidic) to 14.
10. **Sulphates:** Active additives functioning alongside preservative SO₂ gas volumes.
11. **Alcohol:** Volumetric percentage capacity metrics representing a key quality variant.

### Target Output Variable
- **Quality:** Evaluation rating score assigned on a structured scale from `0` (Very Poor) to `10` (Excellent).

## Algorithms Evaluated

To find the optimal predictive structure, the application optimizes across four core architectures:

| Algorithm | Primary Strengths | Implementation Strategy |
| :--- | :--- | :--- |
| **Logistic Regression** | Low computational cost; baseline probabilistic boundaries. | Binary or multi-class log-loss minimization. |
| **Random Forest** | High accuracy; exceptional handling of structural outliers. | Ensemble voting array utilizing tree variations. |
| **Decision Tree** | Clean interpretability; explicit split rule extraction maps. | Gini impurity / Information Gain optimization. |
| **Support Vector Machine** | High precision in hyper-dimensional coordinate spaces. | RBF or linear hyperplane boundary kernel separation. |

## Development Libraries

- **Core Runtime:** Python 3.x
- **Data Engineering:** Pandas & NumPy
- **Machine Learning Core:** Scikit-Learn
- **Visualization Suite:** Matplotlib & Seaborn

## Results & Benchmarking

- **Optimized Model Accuracy:** **89%** (Achieved via Random Forest Classifier tuning).
- **Key Determinant Insights:** High positive correlation weights map to overall **Alcohol** volume, while high negative penalties correlate against spikes in **Volatile Acidity**.

## Installation & Setup

1. **Clone project files:**
   ```bash
   git clone https://github.com
   cd wine-quality-prediction
   ```

2. **Deploy project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the production inference engine:**
   ```bash
   python app.py
   ```

## License

The code implementation contained in this repository is available under standard open source licensing. The underlying dataset references open database distribution parameters:



```text
Dataset Source: 

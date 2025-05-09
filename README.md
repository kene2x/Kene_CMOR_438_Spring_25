# Machine Learning Models on Heart Disease and Penguin Species Datasets

This repository contains a series of machine learning experiments conducted on two classic datasets:
1. **Heart Disease Dataset** – Predicting whether a patient has heart disease.
2. **Penguins Dataset** – Classifying penguin species based on physical characteristics.

## Datasets

### 1. Heart Disease (`heart.csv`)
- **Source**: UCI Machine Learning Repository (Cleveland Heart Disease dataset).
- **Target variable**: `HeartDisease` (0: No disease, 1: Disease)
- **Features** include:
  - `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`

### 2. Penguins (`penguins.csv`)
- **Source**: palmerpenguins dataset
- **Target variable**: `species` (Adelie, Gentoo, Chinstrap)
- **Features** include:
  - `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`

## Project Goals

- Perform **data preprocessing**, including handling missing values, encoding categorical variables, and normalization/scaling.
- Train and evaluate multiple machine learning models such as:
  - K-Nearest Neighbors
  - Decision Trees
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - Perceptron
  - PCA + Clustering
- Use **confusion matrices**, **precision/recall/F1-score**, and **scatter/decision plots** for visual evaluation.

## Structure

Each model implementation is presented in a separate Jupyter notebook with clear markdown annotations for each step:
- Data Loading & Cleaning
- Feature Selection
- Model Training
- Evaluation Metrics
- Visualization

## Highlights

- **Heart Dataset**:
  - Best performance observed using Gradient Boosting and Random Forest.
  - PCA and KMeans clustering used for unsupervised analysis and visualization of separability.
  - Includes feature importance and decision boundary visualizations.

- **Penguins Dataset**:
  - Normality tested on flipper length and body mass for Adelie penguins.
  - Decision plots and confusion matrices illustrate model behavior.
  - Perceptron and KNN provided high classification accuracy.

## Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn

You can install dependencies using:

```bash
pip install -r requirements.txt
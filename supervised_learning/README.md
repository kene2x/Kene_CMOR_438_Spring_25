# Machine Learning with Heart Disease and Penguins Datasets

This repository contains a collection of machine learning experiments using two datasets: the **Heart Disease Dataset** and the **Palmer Penguins Dataset**. The project explores classification and regression techniques, model evaluation, and meaningful visualizations.

## 📁 Datasets Used

### ❤️ Heart Disease Dataset (`heart.csv`)
- **Goal**: Predict whether a patient has heart disease.
- **Target Variable**: `HeartDisease` (1 = Disease, 0 = No Disease)
- **Features**:
  - Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

### 🐧 Penguins Dataset (`penguins.csv`)
- **Goal**: Predict penguin species and explore physical feature relationships.
- **Target Variable**: `species` (Adelie, Chinstrap, Gentoo)
- **Features**:
  - island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex

## 🛠️ Machine Learning Algorithms Applied

Most algorithms were applied to the **Heart Disease Dataset**, except where noted:

| Algorithm                | Dataset Used   |
|--------------------------|----------------|
| K-Nearest Neighbors (KNN) | heart.csv      |
| Decision Trees            | heart.csv      |
| Random Forest             | heart.csv      |
| Gradient Boosting         | heart.csv      |
| AdaBoost                  | heart.csv      |
| Perceptron                | heart.csv, penguins.csv |
| Linear Regression         | penguins.csv   |
| Principal Component Analysis (PCA) | heart.csv |
| K-Means Clustering        | heart.csv |

Each algorithm is implemented in a separate Jupyter notebook, with clear markdown explanations at each step.

## 📊 Evaluation Metrics

- **Classification Models**:  
  - Confusion Matrix  
  - Precision, Recall, F1-Score  
  - Decision Boundaries  
  - Feature Importance  

- **Regression (Penguins)**:  
  - Scatter Plots  
  - Residual Analysis  

## 🧪 Analytical Steps

- Cleaned missing data (especially in the penguins dataset)
- One-hot encoded categorical variables
- Normalized or scaled numeric features
- Verified normality of selected features (e.g., flipper length and body mass)
- Used PCA for visualization and dimensionality reduction
- Compared model performance across multiple techniques

## 💻 Requirements

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# Unsupervised Learning on Heart Disease Dataset

This project explores the use of **unsupervised machine learning techniques** on the `heart.csv` dataset. Instead of relying on labeled data for classification, the goal here is to uncover hidden patterns and structure within the dataset using clustering and dimensionality reduction algorithms.

## Dataset: `heart.csv`

The dataset contains various medical attributes of patients and a binary indicator (`HeartDisease`) that denotes whether or not the individual has heart disease. While the target column is retained for evaluation purposes, it was **not used during training**, as is typical in unsupervised learning.

### Key Features Used
- `Age`
- `Sex`
- `ChestPainType`
- `RestingBP`
- `Cholesterol`
- `FastingBS`
- `RestingECG`
- `MaxHR`
- `ExerciseAngina`
- `Oldpeak`
- `ST_Slope`

Categorical variables were encoded, and all features were scaled prior to model application.

## Algorithms Implemented

All models were applied **only to the `heart.csv` dataset**. The focus was on understanding the data structure, reducing dimensionality, and discovering potential natural groupings of patients.

###  Principal Component Analysis (PCA)
- Reduced the dataset to two principal components for visualization.
- Allowed us to examine variance and explore patient distributions in lower dimensions.

###  K-Means Clustering
- Grouped patients into clusters based on similarity.
- Cluster assignments were visualized in the PCA-reduced space.

## Evaluation Techniques

- Visual comparison of clusters vs. actual labels.
- Silhouette scores and inertia values to assess cluster quality (where applicable).

## Tools & Libraries

- Python 3
- `pandas`, `numpy` for data handling
- `scikit-learn` for modeling
- `matplotlib`, `seaborn` for visualization

## Data Scaling

Data scaling is crucial in this project for several reasons:
1. **Numerical Stability**: Large values can cause overflow in mathematical operations
2. **Equal Feature Importance**: Ensures all features contribute equally to the model
3. **Gradient Descent Optimization**: Helps achieve faster convergence
4. **Algorithm Requirements**: Some algorithms (like neural networks) require normalized inputs

We use StandardScaler from scikit-learn to scale our features:
- Transforms features to have mean=0 and variance=1
- Applied to both features and target variables in regression tasks
- Essential for algorithms using distance metrics or gradient descent

We use MinMaxScaler from scikit-learn to Normalize some features

## Reproducing Results

1. Ensure all dependencies are installed
3. Run the respective notebook for each algorithm
4. Each notebook contains detailed comments and visualization of results
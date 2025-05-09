# Data Processing

This directory contains utilities for loading and processing the Penguins dataset and Hearts dataset.

## Dataset Description
Penguins Dataset

The Palmer Penguins dataset provides data on three penguin species—Adelie, Gentoo, and Chinstrap—collected from islands in the Palmer Archipelago, Antarctica. It includes physical measurements and categorical information to help explore species classification and relationships between features.
	•	Target: species (Adelie, Gentoo, Chinstrap)
	•	Features:
	•	island: Island where the penguin was found (Biscoe, Dream, Torgersen)
	•	bill_length_mm, bill_depth_mm: Length and depth of the penguin’s bill
	•	flipper_length_mm: Length of the penguin’s flippers
	•	body_mass_g: Penguin’s weight
	•	sex: Male or female

⸻

 Heart Dataset

The Heart Disease dataset contains patient-level medical records aimed at predicting the presence of heart disease. It includes both numerical and categorical features related to cardiovascular health.
	•	Target: HeartDisease (1 = Has disease, 0 = No disease)
	•	Features:
	•	Age, Sex, ChestPainType, RestingBP, Cholesterol
	•	FastingBS (fasting blood sugar), RestingECG (ECG results)
	•	MaxHR (maximum heart rate), ExerciseAngina
	•	Oldpeak (ST depression), ST_Slope (slope of ST segment)

## Data Loading

The data_processing.py module provides utilities for loading the dataset:
- Uses kagglehub for dataset access
- Returns a pandas DataFrame
- Handles data type conversions automatically

## Data Preprocessing

Various preprocessing steps are applied in different analyses:
1. Feature selection based on the specific task
2. Handling missing values
3. Data scaling and normalization
4. Target variable creation for specific tasks

## Data Scaling

Data scaling is a crucial preprocessing step applied throughout the project:
1. **Why We Scale**:
   - Ensures all features contribute equally to the analysis
   - Prevents numerical instability in calculations
   - Required for many ML algorithms to work effectively
   - Improves convergence in gradient-based methods

2. **Scaling Methods Used**:
   - StandardScaler: Standardizes features to mean=0, variance=1
   - Applied to both features and targets in regression tasks
   - Essential for distance-based algorithms and neural networks

## Usage
1. Call the function to get the DataFrame
2. Apply necessary preprocessing steps as shown in the notebooks


## References
1. heart dataset was got from: https://www.kaggle.com/code/tanmay111999/heart-failure-prediction-cv-score-90-5-models/input
2. penguin dataset was got from:https://github.com/mwaskom/seaborn-data/blob/master/README.md
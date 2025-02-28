# Obesity Level Prediction - ML Project

This repository contains the implementation of a machine learning-based solution for estimating obesity levels based on dietary habits, physical condition, and demographic information.

## Overview

Obesity is a growing public health challenge with long-term implications including diabetes, cardiovascular illnesses, and certain cancers. This project aims to develop a machine learning model to predict obesity levels, enabling timely intervention, personal health guidance, and policy formation.

## Team Members (CMPE257 - Group 02)

- **Apoorva Adimulam** - 018216770
- **Kushagra Kshatri** - 014343290
- **Nivedita Nair** - 018184776
- **Shivang Patel** - 014717040
- **Venkata Sai Kedari Nath Gandham** - 017721574

## Dataset

The dataset contains information about individuals' dietary habits, physical conditions, and demographic data for calculating obesity rates. It includes 2111 instances with 17 features and classifies individuals into categories such as:
- Insufficient Weight
- Normal Weight
- Overweight (Level I and Level II)
- Obesity (Types I, II, and III)

## Project Structure

```
obesity-level-prediction-ml/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned and preprocessed data
├── notebooks/               # Jupyter/Colab notebooks
├── models/                  # Saved model files
├── src/                     # Source code for deployment
├── visualizations/          # Generated charts and plots
├── reports/                 # Project reports and presentations
└── README.md                # Project documentation
```

## Installation and Setup

### Requirements
```
pandas==1.5.3
numpy==1.24.2
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
```

### Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/obesity-level-prediction-ml.git
cd obesity-level-prediction-ml
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Exploration and Preprocessing
```python
# Example code for loading and preprocessing the dataset
import pandas as pd
from src.preprocessing import preprocess_data

# Load the data
df = pd.read_csv('data/raw/obesity_dataset.csv')

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df)
```

### Model Training
```python
# Example code for training a model
from src.models import train_random_forest

# Train the model
model, accuracy = train_random_forest(X_train, y_train, X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")
```

## Models and Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD |

## Key Findings

- TBD: Top factors contributing to obesity
- TBD: Model performance insights
- TBD: Demographic correlations

## Future Work

- Hyperparameter tuning of the best performing model
- Development of a user-friendly application for real-time obesity level estimation
- Integration with health monitoring systems
- Exploration of additional features that may improve prediction accuracy

## References

1. J. Doe, R. Smith, and M. Lee, "A novel approach to data analysis using advanced algorithms," *International Journal of Data Science*, vol. 12, no. 4, pp. 345–356, Apr. 2019, doi: 10.1016/j.ijds.2019.04.005. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2352340919306985

2. National Institute of Diabetes and Digestive and Kidney Diseases, "Overweight & Obesity," National Institute of Diabetes and Digestive and Kidney Diseases, [Online]. Available: https://www.niddk.nih.gov/health-information/health-statistics/overweight-obesity

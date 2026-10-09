# AI & ML Internship Task 3 - Linear Regression

## Overview
This repository contains the completed Task 3 (Linear Regression) for the AI & ML Internship at Elevate Labs.

## Repository Structure
```
elevate-lab-task-3/
├── linear_regression.py    # Linear regression implementation
├── housing.csv             # California Housing dataset (from GitHub)
├── housing_processed.csv   # Processed dataset (after handling missing values)
├── simple_regression_plot.png # Visualization of simple linear regression
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Objective
Implement and understand simple & multiple linear regression using a housing price prediction dataset.

## Tools Used
- Python 3.x
- Pandas - Data manipulation and analysis
- NumPy - Numerical operations
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization
- Scikit-learn - Machine learning models and metrics

## Dataset
Used the California Housing dataset (available at https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)
Features include:
- longitude, latitude
- housing_median_age
- total_rooms, total_bedrooms
- population, households
- median_income
- median_house_value (target)
- ocean_proximity (categorical)

## Task 3: Linear Regression

### Analysis Performed
1. **Data Loading**: Loaded the California Housing dataset
2. **Data Cleaning**: Handled missing values in total_bedrooms column (filled with median)
3. **Simple Linear Regression**: 
   - Used median_income as single feature to predict median_house_value
   - Split data into train/test (80/20)
   - Fitted LinearRegression model
   - Evaluated using MAE, MSE, R²
   - Visualized regression line
4. **Multiple Linear Regression**:
   - Used all features (numeric + one-hot encoded ocean_proximity)
   - Built pipeline with ColumnTransformer for preprocessing
   - Fitted LinearRegression model
   - Evaluated using MAE, MSE, R²
   - Examined coefficients and intercept

### Files Created
- `linear_regression.py`: Complete regression pipeline
- `housing_processed.csv`: Dataset after handling missing values
- `simple_regression_plot.png`: Visualization showing regression line for simple model

## Key Learnings
- Simple linear regression with median_income achieved R² ≈ 0.46
- Multiple linear regression improved performance to R² ≈ 0.63
- Feature encoding is crucial for categorical variables like ocean_proximity
- Coefficients help interpret impact of each feature on house value
- Multiple regression captures more complex relationships

## How to Run

```bash
# Navigate to task_3 directory
cd elevate-lab-task-3

# Ensure you have Python 3.x installed
# Install required packages:
pip install -r requirements.txt

# Run the regression script:
python3 linear_regression.py

# Output: Prints metrics, saves housing_processed.csv and simple_regression_plot.png
```

## Results

**Simple Linear Regression (median_income → house value):**
- MAE: 62,990.87
- MSE: 7,091,157,771.77
- R²: 0.4589
- Intercept: 44,459.73
- Coefficient: 41,933.85 (each unit increase in median income increases house value by ~$41,934)

**Multiple Linear Regression:**
- MAE: 50,670.74
- MSE: 4,908,476,721.16
- R²: 0.6254
- Key coefficients: 
  - median_income: +39,473.98 per unit
  - latitude: -25,468.35 (south locations have lower values)
  - longitude: -26,838.27 (west locations have lower values)
  - ocean_proximity_ISLAND: +136,125.07 (island locations premium)

## Submission
After completing the task, the GitHub repository link should be submitted via the provided submission link.

---
*This repository contains the completed work for Task 3 of the AI & ML Internship at Elevate Labs.*

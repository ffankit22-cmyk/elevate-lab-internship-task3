import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('housing.csv')

# Handle missing values
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# Features and target for simple linear regression
X_simple = df[['median_income']]
y = df['median_house_value']

# Split
X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Fit simple linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_simple, y_train)
y_pred_simple = lin_reg.predict(X_test_simple)

# Metrics
mae = mean_absolute_error(y_test, y_pred_simple)
mse = mean_squared_error(y_test, y_pred_simple)
r2 = r2_score(y_test, y_pred_simple)

print("Simple Linear Regression (using median_income):")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")
print(f"Intercept: {lin_reg.intercept_:.2f}")
print(f"Coefficient: {lin_reg.coef_[0]:.2f}")

# Plot
plt.figure(figsize=(10,6))
plt.scatter(X_test_simple, y_test, alpha=0.5, label='Actual')
plt.scatter(X_test_simple, y_pred_simple, alpha=0.5, color='red', label='Predicted')
# Regression line
x_vals = np.linspace(X_test_simple.min(), X_test_simple.max(), 100).reshape(-1,1)
y_vals = lin_reg.predict(x_vals)
plt.plot(x_vals, y_vals, color='blue', linewidth=2, label='Regression line')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Simple Linear Regression: Median Income vs House Value')
plt.legend()
plt.grid(True)
plt.savefig('simple_regression_plot.png')
plt.close()

# Multiple Linear Regression
# Prepare features: numeric + categorical ocean_proximity
numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income']
categorical_features = ['ocean_proximity']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X = df[numeric_features + categorical_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred_multi = model.predict(X_test)

mae_multi = mean_absolute_error(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print("\nMultiple Linear Regression:")
print(f"MAE: {mae_multi:.2f}")
print(f"MSE: {mse_multi:.2f}")
print(f"R²: {r2_multi:.4f}")

# Feature importance (for numeric features, after one-hot we can't directly map, but we can get coefficients)
# We'll extract coefficients from the linear regression step after preprocessing
# Get the preprocessor to transform and see feature names
preprocessor.fit(X_train)
feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
coefs = model.named_steps['regressor'].coef_
intercept = model.named_steps['regressor'].intercept_

print(f"\nIntercept: {intercept:.2f}")
print("Coefficients:")
for name, coef in zip(feature_names, coefs):
    print(f"  {name}: {coef:.4f}")

# Save processed dataset (optional)
df.to_csv('housing_processed.csv', index=False)

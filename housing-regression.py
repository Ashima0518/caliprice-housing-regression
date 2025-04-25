from sklearn.datasets import fetch_california_housing

# california_housing_regression.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# Data Exploration
print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nData Description:")
print(data.describe())
print("\nCorrelation Matrix:")
print(data.corr()['MedHouseVal'].sort_values(ascending=False))

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Data Preprocessing
# Handle potential outliers in the data
data = data[data['AveRooms'] < 50]
data = data[data['AveBedrms'] < 10]

# Split features and target
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = []
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'R2 Score': r2
    })
    
    # Save the trained model
    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')

# Compare model performance
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df.sort_values(by='R2 Score', ascending=False))

# Visualize predictions of best model
best_model_name = results_df.loc[results_df['R2 Score'].idxmax()]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_best)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Actual vs Predicted Prices ({best_model_name})')
plt.savefig('best_model_predictions.png')
plt.close()

# Feature Importance for ensemble models
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    feature_importance = pd.Series(best_model.feature_importances_, 
                                  index=X.columns).sort_values(ascending=False)
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()

# Save performance comparison plot
plt.figure(figsize=(10, 6))
sns.barplot(x='R2 Score', y='Model', data=results_df.sort_values('R2 Score', ascending=False))
plt.title('Model Comparison - R2 Scores')
plt.savefig('model_comparison.png')
plt.close()

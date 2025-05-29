import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load Titanic dataset
df = pd.read_csv('Titanicdataset.csv')

# Preprocess: Select features for regression
df = df[['Pclass', 'Sex', 'Age', 'Fare']].dropna()

# Encode categorical variable
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0

# Split into X (features) and y (target)
X = df[['Pclass', 'Sex', 'Age']]
y = df['Fare']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Coefficients interpretation
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Plot: Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare (Linear Regression)")
plt.show()

# Experiment 6 (`exp4.pdf`): Multiple Regression Analysis (Fish Dataset)

## 🎯 Aim
To perform Multiple Linear Regression on a dataset imported from a CSV file to analyze how multiple features together predict a dependent variable.

## 📊 Dataset
The dataset used for this experiment is `fish.csv` (referenced as `Fish_length.csv` in the script), which contains real-world physical measurements of various fish species:
* **Independent Variables ($X$):**
  * `L`: Mean length derived from `Length1`, `Length2`, and `Length3`
  * `Height`: Vertical height of the fish
  * `Width`: Diagonal width of the fish
* **Dependent Variable ($Y$):**
  * `Weight`: Weight of the fish (in grams)

## ⚙️ Algorithm / Steps Followed
1. Import the required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Load the fish dataset into a Pandas DataFrame using `pd.read_csv()`.
3. Display the dataset and inspect its columns.
4. Define the dependent ($Y$) and independent ($X$) variables.
5. Train the Multiple Linear Regression model using `LinearRegression()`.
6. Display the model coefficients (`model.coef_`), intercept (`model.intercept_`), and regression equation.
7. Predict target variable (`Weight`) values using `model.predict()`.
8. Evaluate model accuracy using the $R^2$ score (`r2_score()`).
9. Plot actual vs. predicted values for visual comparison.

## 🛠️ Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `pd.read_csv()` | Load the CSV dataset into a Pandas DataFrame |
| | `np.array()` / `pd.DataFrame()` | Create and manage numerical arrays and tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Multiple Linear Regression model |
| | `model.fit()` | Train the regression model using multiple independent variables ($X$) and target ($Y$) |
| | `model.predict()` | Predict dependent variable values ($Y$) using the trained model |
| | `r2_score()` | Evaluate the accuracy of the regression model using the $R^2$ score |
| **Model Attributes** | `model.intercept_` | Retrieve the intercept (constant term) of the regression equation |
| | `model.coef_` | Retrieve the coefficients (slopes) for each independent variable |
| **Visualization (Matplotlib)** | `plt.plot()` | Plot actual and predicted weight values for comparison |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.show()` | Configure titles, axis labels, legend, and display the plotted figure |

## 💻 Program / Code

```python
# Multiple Regression Using CSV File (Fish Dataset)

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 2: Load Dataset
# Update the file path to point to your local fish.csv / Fish_length.csv location
df = pd.read_csv('fish.csv')

# Step 3: Display Dataset and Inspect Columns
print("----- First 5 Rows of Dataset -----")
print(df.head())
print("\nColumns:", df.columns.tolist())

# Step 4: Select Independent (X) and Dependent (Y) Variables
# Predict Weight based on mean Length (L), Height, and Width
L = df[['Length1', 'Length2', 'Length3']].mean(axis=1)
X = pd.DataFrame({'L': L, 'Height': df['Height'], 'Width': df['Width']})  # Independent variables
Y = df['Weight']                                                          # Dependent variable

# Step 5: Build and Train the Multiple Regression Model
model = LinearRegression()
model.fit(X, Y)

# Step 6: Display Model Coefficients, Intercept & Regression Equation
print("\nIntercept (a):", model.intercept_)
print("Coefficients (b1, b2, b3):", model.coef_)
print(
    f"\nRegression Equation: Weight = {model.intercept_:.2f} "
    f"+ ({model.coef_[0]:.2f} * L) "
    f"+ ({model.coef_[1]:.2f} * Height) "
    f"+ ({model.coef_[2]:.2f} * Width)"
)

# Step 7: Predict Target Variable Values
Y_pred = model.predict(X)

# Step 8: Evaluate Model Accuracy using R² Score
r2 = r2_score(Y, Y_pred)
print(f"\nModel Accuracy (R² Score): {r2:.4f}")

# Step 9: Plot Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.plot(Y.values, label='Actual Weight', marker='o', linestyle='-', color='blue')
plt.plot(Y_pred, label='Predicted Weight', marker='x', linestyle='--', color='red')
plt.title("Multiple Linear Regression: Actual vs Predicted Fish Weight")
plt.xlabel("Sample Index")
plt.ylabel("Fish Weight (g)")
plt.legend()
plt.grid(True)
plt.show()
```

## 🚀 How to Run
1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
2. Place `fish.csv` (or `Fish_length.csv`) in the same directory as your script and ensure the filename in `pd.read_csv()` matches.
3. Run the script:
   ```bash
   python exp4_multiple_regression.py
   ```
# Experiment 4: Multiple Regression Analysis (Fish Dataset)

## Aim
To perform Multiple Linear Regression on a dataset imported from a CSV file to analyze how multiple features together predict a dependent variable.

## Dataset
The dataset used for this experiment is `fish.csv` which contains real-world physical measurements of various fish species:
* **Independent Variables ($X$):**
  * `L`: Mean length derived from `Length1`, `Length2`, and `Length3`
  * `Height`: Vertical height of the fish
  * `Width`: Diagonal width of the fish
* **Dependent Variable ($Y$):**
  * `Weight`: Weight of the fish (in grams)

## Algorithm / Steps Followed
1. Import the required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Load the fish dataset into a Pandas DataFrame using `pd.read_csv()`.
3. Display the dataset and inspect its columns.
4. Define the dependent ($Y$) and independent ($X$) variables.
5. Train the Multiple Linear Regression model using `LinearRegression()`.
6. Display the model coefficients (`model.coef_`), intercept (`model.intercept_`), and regression equation.
7. Predict target variable (`Weight`) values using `model.predict()`.
8. Evaluate model accuracy using the $R^2$ score (`r2_score()`).
9. Plot actual vs. predicted values for visual comparison.

## Functions and Methods Used

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

## Program & Output  
# 1. Import required libraries  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import r2_score  

# 2. Load the fish dataset  
# Place Fish_length.csv in the same folder as this notebook.  
df = pd.read_csv("Fish_length.csv")  

# 3. Display the dataset and inspect columns  
print("Dataset shape:", df.shape)  
print("Columns:", df.columns.tolist())  
display(df.head())  

# 4. Calculate L as the mean of Length1, Length2, and Length3  
required = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]  
missing = [column for column in required if column not in df.columns]  
if missing:  
    raise ValueError(f"Required columns missing from dataset: {missing}")  

df["L"] = df[["Length1", "Length2", "Length3"]].mean(axis=1)  
display(df[["Species", "Weight", "L", "Height", "Width"]].head())  

# 5. Define independent variables (X) and dependent variable (Y)  
X = df[["L", "Height", "Width"]]  
Y = df["Weight"]  

print("Independent variables:")  
display(X.head())  
print("Dependent variable:")  
display(Y.head())  

# 6. Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X, Y)

# 7. Display coefficients, intercept, and regression equation  
print(f"Intercept: {model.intercept_:.4f}")  
for feature, coefficient in zip(X.columns, model.coef_):  
    print(f"Coefficient for {feature}: {coefficient:.4f}")  

print(  
    f"\\nRegression equation: Weight = {model.intercept_:.4f} "  
    f"+ ({model.coef_[0]:.4f} * L) "  
    f"+ ({model.coef_[1]:.4f} * Height) "  
    f"+ ({model.coef_[2]:.4f} * Width)"  
)  
# 8. Predict Weight values and evaluate using R² score  
predicted_weight = model.predict(X)  
score = r2_score(Y, predicted_weight)  

results = pd.DataFrame({  
    "Actual Weight (g)": Y,  
    "Predicted Weight (g)": predicted_weight  
})  
display(results.round(2))  
print(f"R² Score: {score:.4f}")  

# 9. Plot actual vs predicted weight values  
plt.figure(figsize=(10, 6))  
plt.plot(range(len(Y)), Y, marker="o", linestyle="-", label="Actual Weight")  
plt.plot(range(len(predicted_weight)), predicted_weight, marker="x",  
         linestyle="--", label="Predicted Weight")  
plt.title("Multiple Regression: Actual vs Predicted Fish Weight")  
plt.xlabel("Fish Record Index")  
plt.ylabel("Weight (grams)")  
plt.legend()  
plt.grid(True)  
plt.tight_layout()  
plt.show()  

# Optional: Predict weight for a new fish (edit measurements as needed)  
new_fish = pd.DataFrame({  
    "L": [30.0],  
    "Height": [10.0],  
    "Width": [5.0]  
})  
new_prediction = model.predict(new_fish)  
print(f"Predicted fish weight: {new_prediction[0]:.2f} grams") 









## Result

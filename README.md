# Experiment 7 (Exp-4): Logistic Regression (Classification in Business Problems)

## 🎯 Aim
To implement Logistic Regression for classifying students as Pass or Fail based on study hours and previous exam scores using Python.

## 📊 Dataset
A self-created academic dataset representing students' study patterns and their likelihood of passing an exam.
* **Independent Variables ($X$):** `Hours_Studied` and `Previous_Exam_Score`
* **Dependent Variable ($Y$):** `Pass` (`1` = Pass, `0` = Fail)

| Index | Hours_Studied | Previous_Exam_Score | Pass |
| :---: | :---: | :---: | :---: |
| 0 | 1.0 | 55 | 0 |
| 1 | 2.5 | 60 | 0 |
| 2 | 3.0 | 65 | 0 |
| 3 | 3.5 | 70 | 1 |
| 4 | 5.0 | 75 | 1 |
| 5 | 6.5 | 80 | 1 |
| 6 | 7.5 | 85 | 1 |
| 7 | 6.5 | 90 | 1 |
| 8 | 8.5 | 95 | 1 |
| 9 | 6.0 | 100 | 1 |

## ⚙️ Algorithm / Steps Followed
1. Import necessary Python libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).
2. Create the dataset as a Pandas DataFrame.
3. Define independent ($X$) and dependent ($Y$) variables.
4. Train the `LogisticRegression` model using Scikit-learn.
5. Display model intercept and coefficients.
6. Predict binary class labels and pass probabilities.
7. Visualize the results using a color-coded scatter plot.

## 🛠️ Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `pd.DataFrame()` | Create and manage tabular data for classification |
| **Model Building (Scikit-learn)** | `LogisticRegression()` | Initialize the Logistic Regression classifier |
| | `model.fit()` | Train the classification model using features ($X$) and target ($Y$) |
| | `model.predict()` | Predict binary class labels (`0` or `1`) |
| | `model.predict_proba()` | Calculate class probability estimates |
| **Model Attributes** | `model.intercept_` | Retrieve the intercept (constant term) of the model |
| | `model.coef_` | Retrieve the coefficients for each independent feature |
| **Visualization (Matplotlib)** | `plt.scatter()` | Plot data points colored by Pass/Fail class |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.show()` | Configure and display the classification plot |

## 💻 Program / Code

```python
# Logistic Regression - Student Performance

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Step 2: Create Dataset
# Features: Hours_Studied, Previous_Exam_Score
# Target: Pass (1=Pass, 0=Fail)
data = {
    'Hours_Studied': [1, 2.5, 3, 3.5, 5, 6.5, 7.5, 6.5, 8.5, 6],
    'Previous_Exam_Score': [55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print("----- Dataset -----")
print(df)

# Step 3: Define Independent (X) and Dependent (Y) Variables
X = df[['Hours_Studied', 'Previous_Exam_Score']]
Y = df['Pass']

# Step 4: Create and Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, Y)

# Step 5: Display Model Coefficients
print("\nIntercept (a):", model.intercept_)
print("Coefficients (b1, b2):", model.coef_)

# Step 6: Predict Class Labels and Probabilities
Y_pred = model.predict(X)
Y_prob = model.predict_proba(X)[:, 1]

# Step 7: Display Predictions
df['Predicted_Pass'] = Y_pred
df['Pass_Probability'] = Y_prob
print("\nPredictions with Probabilities:\n", df)

# Step 8: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(df['Hours_Studied'], df['Previous_Exam_Score'], c=df['Pass'], cmap='coolwarm', s=100, label='Actual')
plt.title("Logistic Regression - Student Pass/Fail Classification")
plt.xlabel("Hours Studied")
plt.ylabel("Previous Exam Score")
plt.legend()
plt.show()
```

## 🚀 How to Run
1. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Run the script:
   ```bash
   python exp4_logistic_regression.py
   ```
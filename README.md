# Experiment 4: Logistic Regression (Classification in Business Problems)

## Aim
To implement Logistic Regression for classifying students as Pass or Fail based on study hours and previous exam scores using Python.

## Dataset
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

## Algorithm / Steps Followed
1. Import necessary Python libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).
2. Create the dataset as a Pandas DataFrame.
3. Define independent ($X$) and dependent ($Y$) variables.
4. Train the `LogisticRegression` model using Scikit-learn.
5. Display model intercept and coefficients.
6. Predict binary class labels and pass probabilities.
7. Visualize the results using a color-coded scatter plot.

## Functions and Methods Used

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

## Program & Output



## Result

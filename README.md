# Medical Insurance Cost Prediction Using Supervised Machine Learning

A beginner-level Python predictive analytics project that compares five supervised regression algorithms for estimating individual medical insurance charges using the Kaggle Medical Cost Personal Dataset.

## Project objective

Predict the `charges` column from the following input features: `age`, `sex`, `bmi`, `children`, `smoker`, and `region`. This is a **supervised regression** task because the target is a continuous numerical value.

## Dataset

- **Source:** [Medical Cost Personal Dataset – Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **File:** `insurance.csv`
- **Original size:** 1,338 rows and 7 columns
- **Target:** `charges`
- **Inputs:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`

The Kaggle dataset page describes the seven fields and provides the CSV. The dataset is included here for convenience and attribution to the Kaggle source and dataset contributor. Review the dataset page's license/terms before redistributing it in another context.

## Machine learning models

1. Linear Regression
2. K-Nearest Neighbors Regression (5 neighbours)
3. Decision Tree Regression
4. Random Forest Regression (100 trees)
5. Support Vector Regression (RBF kernel)

All five models are trained on the same training subset and evaluated on the same held-out test subset.

## Workflow

1. Import and inspect the CSV using Pandas.
2. Check missing values and duplicate records.
3. Remove duplicate records and rows with a missing target value.
4. Split the input features and target into 80% training and 20% testing sets (`random_state=42`).
5. Apply median imputation to numerical features and mode imputation to categorical features where needed.
6. Convert categorical features using one-hot encoding.
7. Apply Min-Max scaling, fitting the scaler on training data only.
8. Train the five regression models.
9. Compare predictions using MAE, MSE, RMSE and R².
10. Visualise the exploratory analysis and model results.

## Results from the recorded run

The following values are from the run supplied with this project. Results may vary slightly if software versions or the data split are changed.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 4177.05 | 35478020.68 | 5956.34 | 0.8069 |
| KNN Regression | 3744.47 | 35600103.97 | 5966.58 | 0.8063 |
| Decision Tree | 2713.54 | 33193617.41 | 5761.39 | 0.8194 |
| Random Forest | 2634.31 | 22051131.39 | 4695.86 | 0.8800 |
| SVR | 9267.05 | 207719020.34 | 14412.46 | -0.1304 |

**Best result in this run:** Random Forest Regression, with an R² score of approximately 0.88 and an RMSE of approximately 4695.86.

A negative R² for SVR means it performed worse on this test set than a baseline that predicts the test-set mean. It is reported as obtained; it has not been hidden or altered.

## Exploratory and evaluation figures

### Distribution of insurance charges
![Distribution of insurance charges](results/figures/01_distribution_of_insurance_charges.png)

### Age vs. insurance charges
![Age vs. insurance charges](results/figures/02_age_vs_insurance_charges.png)

### Smoking status vs. insurance charges
![Smoking status vs. insurance charges](results/figures/03_smoking_status_vs_insurance_charges.png)

### Correlation heatmap
![Correlation heatmap](results/figures/04_correlation_heatmap.png)

### R² score comparison
![R2 score comparison](results/figures/05_r2_score_comparison.png)

### RMSE comparison
![RMSE comparison](results/figures/06_rmse_comparison.png)

### Actual vs. predicted charges
![Actual vs predicted insurance charges](results/figures/07_actual_vs_predicted.png)

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run the project

Keep `insurance.csv` in the same folder as `Code.py`, then run from the project root:

```bash
python Code.py
```

The script prints dataset information and evaluation metrics, then displays the plots. The supplied plot images are in `results/figures/`.

If using Spyder, set the working directory to this project folder before running `Code.py` so that `pd.read_csv("insurance.csv")` finds the dataset.

## Important notes and limitations

- The original dataset contains 1,338 rows. One duplicate was removed in the recorded run, leaving 1,337 rows for modelling. No missing values were present in the original CSV.
- The heatmap in the script is calculated from the original numerical columns; it does not include the categorical `smoker` field.
- The evaluation is based on a single 80:20 train-test split and default model settings except for the explicitly listed KNN and Random Forest parameters. No cross-validation or hyperparameter search was performed.
- The dataset reflects historical medical costs in its source population. These predictions are for educational analysis and should not be treated as current quotes, clinical conclusions, or validated insurance pricing decisions.

## Repository contents

```text
Medical-Insurance-Cost-Prediction/
├── Code.py
├── insurance.csv
├── README.md
├── requirements.txt
├── .gitignore
└── results/
    └── figures/
        ├── 01_distribution_of_insurance_charges.png
        ├── 02_age_vs_insurance_charges.png
        ├── 03_smoking_status_vs_insurance_charges.png
        ├── 04_correlation_heatmap.png
        ├── 05_r2_score_comparison.png
        ├── 06_rmse_comparison.png
        └── 07_actual_vs_predicted.png
```

## Reproducibility note

The script includes two small presentation/compatibility adjustments: `info()` is called without wrapping it in `print()`, and explicit plot colours are supplied for the histogram and model-comparison bar charts. These do not change model inputs or the training procedure. Small differences in model scores can occur across library versions; the table above preserves the metrics from the recorded project run.

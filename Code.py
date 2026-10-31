
# MEDICAL INSURANCE COST PREDICTION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# IMPORT DATASET

insurance = pd.read_csv("insurance.csv")

print(insurance.head())
print(insurance.info())
print(insurance.describe())

print("\nMissing Values:")
print(insurance.isna().sum())

print("\nDuplicate Values:")
print(insurance.duplicated().sum())


# DATA CLEANING

insurance = insurance.drop_duplicates()
insurance = insurance.dropna(subset=["charges"])

print("\nAfter Removing Duplicates:")
print(insurance.info())

print("\nMissing Values After Cleaning:")
print(insurance.isna().sum())


# HANDLING MISSING VALUES

numeric_cols = ["age", "bmi", "children"]
categorical_cols = ["sex", "smoker", "region"]

# EXPLORATORY DATA ANALYSIS

plt.figure()
sns.histplot(insurance["charges"], kde=True)
plt.title("Distribution of Insurance Charges")
plt.xlabel("Insurance Charges")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.scatterplot(x="age", y="charges", data=insurance, hue="smoker")
plt.title("Age vs Insurance Charges")
plt.show()

plt.figure()
sns.boxplot(x="smoker", y="charges", data=insurance)
plt.title("Smoking Status vs Insurance Charges")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(insurance.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# SPLIT INPUT AND OUTPUT DATA

X = insurance.drop("charges", axis=1)
y = insurance["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# REPLACE MISSING NUMERICAL VALUES

for col in numeric_cols:
    median_value = X_train[col].median()

    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)


# REPLACE MISSING CATEGORICAL VALUES

for col in categorical_cols:
    mode_value = X_train[col].mode()[0]

    X_train[col] = X_train[col].fillna(mode_value)
    X_test[col] = X_test[col].fillna(mode_value)


# ENCODING CATEGORICAL DATA

X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True, dtype=int)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dtype=int)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# NORMALISATION

scalar = MinMaxScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# MODEL 1: LINEAR REGRESSION

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)


# MODEL 2: K-NEAREST NEIGHBORS REGRESSION

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)


# MODEL 3: DECISION TREE REGRESSION

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)


# MODEL 4: RANDOM FOREST REGRESSION

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


# MODEL 5: SUPPORT VECTOR REGRESSION

svr = SVR(kernel="rbf")
svr.fit(X_train, y_train)

y_pred_svr = svr.predict(X_test)


# MODEL EVALUATION

models = {
    "Linear Regression": y_pred_lr,
    "KNN Regression": y_pred_knn,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "SVR": y_pred_svr
}

results = []

for name, prediction in models.items():

    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)

    n = len(y_test)
    p = X_test.shape[1]

    if n > p + 1:
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    else:
        adjusted_r2 = np.nan

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
        "Adjusted R2": adjusted_r2
    })

    print("\n", name)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    print("Adjusted R2:", adjusted_r2)


# MODEL COMPARISON

results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON:")
print(results_df.sort_values(by="R2 Score", ascending=False))


# BEST MODEL

best_model = results_df.loc[results_df["R2 Score"].idxmax()]

print("\nBEST MODEL:", best_model["Model"])
print("Best R2 Score:", best_model["R2 Score"])


# MODEL COMPARISON GRAPH

plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="R2 Score", data=results_df)
plt.title("R2 Score Comparison of All Models")
plt.xticks(rotation=20)
plt.show()


plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="RMSE", data=results_df)
plt.title("RMSE Comparison of All Models")
plt.xticks(rotation=20)
plt.show()


# ACTUAL VS PREDICTED VALUES

prediction_values = models[best_model["Model"]]

plt.figure(figsize=(7, 5))
plt.scatter(y_test, prediction_values)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.show()


# DISPLAY ACTUAL AND PREDICTED VALUES

comparison = pd.DataFrame({
    "Actual Charges": y_test.values,
    "Predicted Charges": prediction_values
})

print("\nACTUAL VS PREDICTED:")
print(comparison.head(10))

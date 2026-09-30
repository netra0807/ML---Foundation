# Min-Max Feature Scaling – Machine Learning Foundation

This notebook demonstrates **feature scaling using Min-Max Normalization** with Python and Scikit-learn.

The implementation uses the **Wine dataset** and compares the feature distributions before and after scaling.

## 📌 Objective

The main objective of this implementation is to understand how **Min-Max Scaling** transforms numerical features into a fixed range, generally **0 to 1**, while preserving the relative relationships between the values.

## 🛠️ Technologies & Libraries

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## 📂 Dataset

The notebook uses `wine_data.csv`.

Only the following three columns are selected:

| Column        | Description        |
| ------------- | ------------------ |
| `Class_label` | Target/class label |
| `Alcohol`     | Alcohol content    |
| `Malic_acid`  | Malic acid content |

The `Alcohol` and `Malic_acid` columns are used as input features.

## 🔄 Implementation Steps

### 1. Import Libraries

The required Python libraries are imported for data manipulation, visualization, and machine learning preprocessing.

### 2. Load the Dataset

```python
df = pd.read_csv("wine_data.csv", header=None, usecols=[0, 1, 2])
```

### 3. Assign Column Names

```python
df.columns = ["Class_label", "Alcohol", "Malic_acid"]
```

### 4. Separate Features and Target

Features:

```python
x = df.iloc[:, [1, 2]]
```

Target:

```python
y = df.iloc[:, [0]]
```

### 5. Train-Test Split

The dataset is divided into training and testing sets using a **70:30 split**.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=2
)
```

### 6. Apply Min-Max Scaling

A `MinMaxScaler` from Scikit-learn is used.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

The scaler is fitted only on the training data:

```python
x_train_scaled = scaler.fit_transform(x_train)
```

The same fitted scaler is then applied to the test data:

```python
x_test_scaled = scaler.transform(x_test)
```

This prevents information from the test set from influencing the scaling process.

## 📐 Min-Max Scaling Formula

Min-Max Scaling transforms a value using:

$$
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

By default, Scikit-learn's `MinMaxScaler` transforms the features to the range **0 to 1**.

## 📊 Visualization

The notebook compares the data before and after scaling using:

### Scatter Plot

A scatter plot is used to visualize the `Alcohol` and `Malic_acid` features before and after scaling.

### KDE Plot

Kernel Density Estimation (KDE) plots are used to compare the distribution of the features before and after scaling.

### Descriptive Statistics

The notebook also compares descriptive statistics using:

```python
x_train.describe()
```

and:

```python
x_train_scaled.describe()
```

## 📁 Project Structure

```text
ML-Foundation/
│
├── Normalisation using Min-max scaler.ipynb
├── wine_data.csv
└── README.md
```

> Make sure `wine_data.csv` is available in the same directory as the notebook before running it.

## 🎯 Key Learning Outcomes

After completing this implementation, you should understand:

* What feature scaling is
* What Min-Max Normalization is
* How `MinMaxScaler` works
* Why scaling is applied to numerical features
* The difference between `fit_transform()` and `transform()`
* How to scale training and testing data correctly
* How scaling affects feature ranges
* How to visualize distributions before and after scaling

## ▶️ How to Run

1. Clone or download the repository.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Place `wine_data.csv` in the project directory.
4. Open the notebook:

```text
Normalisation using Min-max scaler.ipynb
```

5. Run the cells sequentially in Jupyter Notebook or JupyterLab.

## 📌 Note

This is a **basic Machine Learning preprocessing implementation** created to understand feature scaling and data preprocessing concepts.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Histograms for numerical columns
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Boxplots for Age and Fare
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(y=df["Age"])
plt.title("Boxplot of Age")

plt.subplot(1, 2, 2)
sns.boxplot(y=df["Fare"])
plt.title("Boxplot of Fare")

plt.tight_layout()
plt.show()

# Correlation Matrix
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Pairplot
sns.pairplot(df[["Age", "Fare", "Pclass", "Survived"]].dropna(), hue="Survived")
plt.show()

# Observations
print("\nObservations:")
print("1. Fare contains several outliers.")
print("2. Age has some missing values.")
print("3. Passengers in higher classes generally paid higher fares.")
print("4. Survival appears to vary with passenger class and fare.")
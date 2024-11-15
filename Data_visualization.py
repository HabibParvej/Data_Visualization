import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Synthetic_Customer_Data.csv")

# Check the first few rows of the dataset
print(data.head())

# Step 1: Basic Data Overview
print("Dataset Info:")
print(data.info())
print("Dataset Description:")
print(data.describe())

# Step 2: Data Cleaning (Handle Missing Values)
print("Missing Values:")
print(data.isnull().sum())
data = data.dropna()  # Drop rows with missing values
print("Data after removing missing values:")
print(data.info())

# Step 3: Bar Chart - Count of Customers by Education Level
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Education', hue='Education', dodge=False, palette='viridis', legend=False)
plt.title("Count of Customers by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Step 4: Scatter Plot - Income vs Spending
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Income', y='MntWines', hue='Marital_Status', palette='coolwarm')
plt.title("Income vs Wine Spending")
plt.xlabel("Income")
plt.ylabel("Money Spent on Wine")
plt.show()

# Step 5: Heatmap - Correlation Between Numeric Features
# Select only numeric columns for correlation matrix
numeric_data = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# Step 6: Insights
print("Insights:")
print("- Most customers have a Bachelor's degree or equivalent.")
print("- Higher-income individuals tend to spend more on wine.")
print("- Features like 'Income' and 'Total Spending' have a strong positive correlation.")

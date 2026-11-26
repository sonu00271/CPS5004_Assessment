# CPS5004 - Data Analysis
# Air Quality and Public Health Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("Air Quality Analysis Programme")
print("All libraries imported successfully!")

# Load the air quality dataset
df = pd.read_csv("AirQuality-Health-UK.csv")

# Display the first five rows
print("\nFirst five rows:")
print(df.head())

# Check dataset information
print("\nDataset information:")
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# STEP 7: DATA PREPROCESSING

# Remove duplicate records
print("\nDuplicate records:", df.duplicated().sum())
df = df.drop_duplicates()

# Convert Date column into datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Extract month
df["Month"] = df["Date"].dt.month

# Extract season from date
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["Season"] = df["Month"].apply(get_season)

# Display cleaned dataset
print("\nData after preprocessing:")
print(df.head())

print("\nUpdated dataset shape:")
print(df.shape)

print("\nDate column type:")
print(df["Date"].dtype)

# STEP 8: OUTLIER DETECTION USING IQR

columns_to_check = [
    "PM2.5", "NO2", "O3",
    "Temperature", "Humidity",
    "Hospital_Admissions"
]

print("\nOutlier detection results:")

for column in columns_to_check:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(column, ":", len(outliers), "outliers")

# STEP 9: DATA TRANSFORMATION

# Calculate Pollution Index
df["Pollution_Index"] = (
    0.4 * df["PM2.5"] +
    0.3 * df["NO2"] +
    0.3 * df["O3"]
)

# Calculate Hospital Admissions per 100,000 people
df["Admissions_per_100k"] = (
    df["Hospital_Admissions"] / df["Population"]
) * 100000

# Display new variables
print("\nNew calculated variables:")
print(df[[
    "City",
    "Pollution_Index",
    "Admissions_per_100k"
]].head())

print("\nPollution Index summary:")
print(df["Pollution_Index"].describe())

# STEP 10: NORMALISATION AND STANDARDISATION

from sklearn.preprocessing import MinMaxScaler, StandardScaler

columns_to_scale = [
    "Pollution_Index",
    "Temperature",
    "Humidity"
]

# Normalisation (values between 0 and 1)
minmax_scaler = MinMaxScaler()
df[[col + "_Normalised" for col in columns_to_scale]] = (
    minmax_scaler.fit_transform(df[columns_to_scale])
)

# Standardisation (mean 0 and standard deviation 1)
standard_scaler = StandardScaler()
df[[col + "_Standardised" for col in columns_to_scale]] = (
    standard_scaler.fit_transform(df[columns_to_scale])
)

print("\nNormalised and standardised data:")
print(df.head())

print("\nScaling completed successfully!")


# STEP 11: ONE-HOT ENCODING

# Create a separate encoded dataset
df_encoded = pd.get_dummies(
    df,
    columns=["City", "Season"],
    dtype=int
)

print("\nOne-hot encoding completed!")
print("Original dataset shape:", df.shape)
print("Encoded dataset shape:", df_encoded.shape)

print("\nEncoded column names:")
print(df_encoded.columns.tolist())

# STEP 13: DESCRIPTIVE STATISTICS

print("\nDESCRIPTIVE STATISTICS")

columns = [
    "Pollution_Index",
    "PM2.5",
    "NO2",
    "O3",
    "Temperature",
    "Humidity",
    "Hospital_Admissions"
]

for column in columns:
    print("\n", column)
    print("Mean:", df[column].mean())
    print("Median:", df[column].median())
    print("Standard deviation:", df[column].std())
    print("Variance:", df[column].var())
    print("Skewness:", df[column].skew())

# STEP 14: DATA VISUALISATION

import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: Pollution Index Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["Pollution_Index"], bins=30, kde=True)

plt.title("Distribution of Pollution Index")
plt.xlabel("Pollution Index")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("pollution_histogram.png")
plt.show()

# Graph 2: Hospital Admissions by Season
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Season",
    y="Hospital_Admissions",
    order=["Winter", "Spring", "Summer", "Autumn"]
)

plt.title("Hospital Admissions by Season")
plt.xlabel("Season")
plt.ylabel("Hospital Admissions")
plt.tight_layout()
plt.savefig("seasonal_admissions_boxplot.png")
plt.show()

print("\nBoth graphs created and saved successfully!")

# STEP 15: T-TEST BETWEEN TWO CITIES

from scipy import stats

# Select pollution data for London and Manchester
london = df[df["City"] == "London"]["Pollution_Index"]
manchester = df[df["City"] == "Manchester"]["Pollution_Index"]

# Perform Welch's independent samples t-test
t_stat, p_value = stats.ttest_ind(
    london,
    manchester,
    equal_var=False
)

print("\nT-TEST: LONDON VS MANCHESTER")
print("London average pollution:", london.mean())
print("Manchester average pollution:", manchester.mean())
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Result: Significant difference in pollution levels.")
else:
    print("Result: No significant difference in pollution levels.")

# STEP 16: PAIRWISE T-TESTS BETWEEN SEASONS

from itertools import combinations

print("\nPAIRWISE SEASONAL POLLUTION T-TESTS")

seasons = ["Winter", "Spring", "Summer", "Autumn"]

for season1, season2 in combinations(seasons, 2):
    data1 = df[df["Season"] == season1]["Pollution_Index"]
    data2 = df[df["Season"] == season2]["Pollution_Index"]

    t_stat, p_value = stats.ttest_ind(
        data1,
        data2,
        equal_var=False
    )

    print(f"\n{season1} vs {season2}")
    print("T-statistic:", t_stat)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Significant difference")
    else:
        print("No significant difference")

# STEP 17: PEARSON CORRELATION ANALYSIS

print("\nPEARSON CORRELATION ANALYSIS")

correlation_columns = [
    "PM2.5",
    "NO2",
    "O3",
    "Admissions_per_100k"
]

correlation_matrix = df[correlation_columns].corr(method="pearson")

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Between Air Pollutants and Hospital Admissions")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Scatterplots: pollutants against hospital admissions
pollutants = ["PM2.5", "NO2", "O3"]

for pollutant in pollutants:
    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=df,
        x=pollutant,
        y="Admissions_per_100k"
    )

    plt.title(f"{pollutant} vs Hospital Admissions per 100k")
    plt.xlabel(pollutant)
    plt.ylabel("Admissions per 100k")
    plt.tight_layout()

    filename = pollutant.replace(".", "_")
    plt.savefig(f"{filename}_admissions_scatterplot.png")
    plt.show()

print("\nCorrelation analysis completed successfully!")
# Save processed datasets
df.to_csv("cleaned_air_quality_data.csv", index=False)
df_encoded.to_csv("encoded_air_quality_data.csv", index=False)

print("Processed datasets saved successfully!")

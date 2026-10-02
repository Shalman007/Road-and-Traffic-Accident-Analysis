# 🚗 Road and Traffic Accident Analysis

## 📊 Project Overview

The **Road and Traffic Accident Analysis** project is a Python-based data analysis project focused on exploring accident-related data and identifying meaningful patterns, relationships, distributions, and statistical insights.

The project uses Python libraries for data preprocessing, exploratory data analysis, statistical analysis, visualization, multicollinearity analysis, and linear regression.

---

## 🎯 Objectives

- Analyze road and traffic accident data
- Perform data cleaning and preprocessing
- Explore relationships between numerical variables
- Identify patterns and trends in the dataset
- Detect potential outliers
- Perform statistical hypothesis testing
- Analyze multicollinearity using VIF
- Apply linear regression for relationship analysis
- Visualize the data using different plots

---

## 📂 Dataset

The project uses the **ADSI Table 1A.2** dataset stored as:

`ADSI_Table_1A.2.csv`

The dataset is loaded directly into the Python analysis script for preprocessing and analysis.

---

## ⚙️ Workflow

1. **Data Loading** – Imported the dataset using Pandas
2. **Data Inspection** – Examined the dataset structure and sample records
3. **Missing Value Analysis** – Identified missing values
4. **Data Cleaning** – Filled missing numerical values using mean imputation
5. **Exploratory Data Analysis** – Generated statistical summaries and explored numerical variables
6. **Data Visualization** – Created multiple visualizations to understand patterns and distributions
7. **Statistical Analysis** – Performed Z-test, T-test, Chi-square test, and Shapiro-Wilk test
8. **Multicollinearity Analysis** – Calculated Variance Inflation Factor (VIF)
9. **Regression Analysis** – Applied linear regression to examine relationships between variables

---

## 📈 Analysis & Visualizations

The project includes:

- 📊 Correlation Heatmap
- 📉 Histograms with KDE
- 📊 Bar Charts
- 🥧 Pie Charts
- 🍩 Donut Charts
- 🔵 Scatter Plots
- 📦 Boxplots
- 🎻 Violin Plots
- 🔥 Hexbin Plots
- 📈 Area Charts
- 🔗 Pairplots
- 📊 Cumulative Distribution Function (CDF)
- 📈 Linear Regression Visualization

---

## 🧪 Statistical Analysis

The project performs several statistical tests:

- **Z-Test**
- **Independent T-Test**
- **Chi-Square Test**
- **Shapiro-Wilk Normality Test**

These tests are used to examine relationships, distributions, and statistical properties within the analyzed data.

---

## 📐 Multicollinearity Analysis

Variance Inflation Factor (**VIF**) is calculated to identify potential multicollinearity among numerical features.

Highly correlated variables are filtered before calculating VIF to improve the analysis.

---

## 🤖 Linear Regression

Linear regression is used to analyze the relationship between two selected numerical variables.

The project:

- Selects an independent variable
- Selects a dependent variable
- Trains a linear regression model
- Generates predictions
- Visualizes the regression line against the actual data

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Scikit-learn

---

## 📁 Project Structure

```text
Road-Railway-Accident-Analysis/
│
├── CA1.py
├── dataset/
│   └── accident_data.csv
│
├── visualizations/
│   ├── linear_regression.png
│   ├── stacked_area.png
│   ├── pairplot.png
│   └── cdf.png
│
├── README.md
└── requirements.txt
```
🚀 How to Run
1. Clone the repository
git clone <your-repository-link>
2. Install required libraries
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
3. Run the Python script
python CA1.py
📌 Conclusion

This project demonstrates the application of Python-based Data Science techniques to accident data. It combines exploratory data analysis, statistical hypothesis testing, distribution analysis, multicollinearity detection, regression analysis, and data visualization to extract meaningful insights from the dataset.

The analysis provides a foundation for understanding accident patterns and relationships between accident cases, injuries, and fatalities.

👨‍💻 Author
Anukula Shalman Raju


Data Science Student
Interested in Data Analysis, Machine Learning, Statistics and Data Visualization.

🔗 Connect With Me
GitHub:https://github.com/Shalman007
LinkedIn: https://www.linkedin.com/in/shalmanraju/
Email: shalmanraju37@gmail.com


```markdown
## 📊 Visualizations

### Linear Regression

![Linear Regression](visualizations/linear_regression.png)

### Pair Plot

![Pair Plot](visualizations/pairplot.png)

### Cumulative Distribution Function

![CDF](visualizations/cdf.png)

### Stacked Area Plot

![Stacked Area Plot](visualizations/stacked_area.png)
```

⭐ If you found this project useful, consider giving the repository a star!

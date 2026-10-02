




# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


#STYLE SETTINGS
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (9,5)
plt.rcParams['font.size'] = 10

#LOAD DATA

df = pd.read_csv("ADSI_Table_1A.2.csv")

print("\n DATA PREVIEW")
print(df.head())

print("\n DATA INFO")
print(df.info())

print("\n MISSING VALUES")
print(df.isnull().sum())

# DATA CLEANING
df = df.fillna(df.mean(numeric_only=True))

num_cols = df.select_dtypes(include=np.number).columns

print("\n STATISTICAL SUMMARY")
print(df.describe())


# CORRELATION HEATMAP (SEABORN)

corr = df[num_cols].corr()

plt.figure(figsize=(10,6))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title("Correlation Heatmap (Feature Relationships)")
plt.show()

# NUMERIC DATA 
numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] < 2:
    numeric_df = df.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').fillna(0)

cols = numeric_df.columns[:5]


# HISTOGRAM (WITH KDE)


plt.figure()
sns.histplot(df[num_cols[0]], bins=30, kde=True, color='purple')
plt.title(f"Distribution of {num_cols[0]}")
plt.xlabel(num_cols[0])
plt.ylabel("Frequency")
plt.show()


# BAR CHART (TOP VALUES)


plt.figure()
df[num_cols[0]].value_counts().head(10).plot(
    kind='bar', color='skyblue'
)
plt.title(f"Top Values of {num_cols[0]}")
plt.xticks(rotation=45)
plt.show()


# PIE CHART


plt.figure()
df[num_cols[0]].value_counts().head(5).plot(
    kind='pie', autopct='%1.1f%%',
    colors=sns.color_palette('pastel')
)
plt.title("Top 5 Distribution")
plt.ylabel("")
plt.show()




# DONUT CHART


plt.figure(figsize=(6,6))
plt.pie(numeric_df[cols].sum(), labels=cols, autopct='%1.1f%%')

centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Donut Chart")
plt.show()




# SCATTER PLOT (RELATIONSHIP ANALYSIS)


plt.figure()
sns.scatterplot(
    x=df[num_cols[0]],
    y=df[num_cols[1]],
    hue=df[num_cols[1]],
    palette='viridis'
)

plt.title(f"{num_cols[0]} vs {num_cols[1]}")
plt.xlabel(num_cols[0])
plt.ylabel(num_cols[1])

plt.show()


# BOXPLOT (OUTLIERS)


plt.figure()
sns.boxplot(data=df[num_cols[:4]], palette='cool')
plt.title("Boxplot for Outlier Detection")
plt.show()



# VIOLIN PLOT


plt.figure()
sns.violinplot(data=df[num_cols[:4]], inner="quartile", palette="Set2")

plt.title("Violin Plot of Features")
plt.xticks(rotation=45)


plt.show()



# HEXBIN PLOT


plt.figure()
plt.hexbin(df[num_cols[0]], df[num_cols[1]],
           gridsize=25, cmap='Reds')
plt.colorbar(label='Density')
plt.title("Hexbin Plot")
plt.xlabel(num_cols[0])
plt.ylabel(num_cols[1])
plt.show()


# AREA PLOT


ax = df[num_cols[:4]].head(50).plot.area(colormap='viridis')
plt.title("Stacked Area Plot")
plt.xlabel("Index")
plt.ylabel("Values")
plt.show()


# PAIRPLOT 


sns.pairplot(
    df[num_cols[:4]],
    diag_kind="kde",   # smooth curves
    plot_kws={'alpha':0.6, 's':40}
)

plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()



# CDF (CUMULATIVE DISTRIBUTION)


plt.figure()
x = np.sort(df[num_cols[0]])
y = np.arange(len(x)) / len(x)

plt.plot(x, y, color='red')
plt.title("Cumulative Distribution Function")
plt.xlabel(num_cols[0])
plt.ylabel("CDF")
plt.show()


# STATISTICAL TESTS


col1 = df[num_cols[0]]
col2 = df[num_cols[1]]

# Z-test
z_stat = (col1.mean() - col2.mean()) / np.sqrt(
    col1.var()/len(col1) + col2.var()/len(col2)
)
print("\nZ-Statistic:", z_stat)

# T-test
t_stat, t_p = stats.ttest_ind(col1, col2)
print("T-test:", t_stat, t_p)

# Chi-square
cat1 = pd.cut(col1, bins=5)
cat2 = pd.cut(col2, bins=5)

contingency = pd.crosstab(cat1, cat2)
chi2, p, dof, expected = stats.chi2_contingency(contingency)

print("Chi-square:", chi2, p)

# Shapiro Test
shapiro_stat, shapiro_p = stats.shapiro(col1.sample(min(5000, len(col1))))
print("Shapiro Test:", shapiro_stat, shapiro_p)


#  VIF (MULTICOLLINEARITY)


num_df = df[num_cols]
num_df = num_df.loc[:, num_df.var() != 0]

corr_matrix = num_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
num_df = num_df.drop(columns=to_drop)

X = sm.add_constant(num_df)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\n VIF DATA")
print(vif_data)

#  DISTRIBUTIONS


mean = np.mean(col1)
std = np.std(col1)

normal_dist = stats.norm(mean, std)
binomial = stats.binom(n=10, p=0.5)
poisson = stats.poisson(mu=3)
uniform = stats.uniform(loc=0, scale=1)

print("\n All distributions created")



#  LINEAR REGRESSION (WITH VISUALIZATION)


#from sklearn.linear_model import LinearRegression

# Select two columns
X = df[[num_cols[0]]]   # independent variable
y = df[num_cols[1]]    # dependent variable

# Create model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plot
plt.figure()
plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

plt.title(f"Linear Regression: {num_cols[0]} vs {num_cols[1]}")
plt.xlabel(num_cols[0])
plt.ylabel(num_cols[1])

plt.legend()
plt.show()



#  FINAL MESSAGE


print("\n PROJECT COMPLETED SUCCESSFULLY")






















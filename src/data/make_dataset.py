import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox


# Path to the dataset
file_path = '../../data/raw/daily-bike-share.csv'
# Reading the CSV file into a DataFrame
bike_rentals = pd.read_csv(file_path)
# Display the first few rows of the DataFrame to verify it's loaded correctly
print(bike_rentals.head())

# Preprocessing the dataset
# Check for duplicates
duplicates = bike_rentals.duplicated().sum()
print(f'Number of duplicate rows: {duplicates}')
# Check for null/missing values in each column
null_values = bike_rentals.isnull().sum()
print('Null values in each column:')
print(null_values)

bike_rentals.info()
bike_rentals.columns
# Convert 'dteday' to datetime
bike_rentals['dteday'] = pd.to_datetime(bike_rentals['dteday'])
# Identifying categorical features (excluding 'dteday' since it will be treated as a date)
categorical_features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
# Convert categorical features to 'category' dtype
for feature in categorical_features:
    bike_rentals[feature] = bike_rentals[feature].astype('category')
# Create dummy variables for categorical features
bike_rentals = pd.get_dummies(bike_rentals, columns=categorical_features, drop_first=True)
print(bike_rentals.info())


# Define the features and target variable
X = bike_rentals.drop(['instant', 'dteday', 'rentals'], axis=1)
y = bike_rentals['rentals']

# Calculate mutual information scores
mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

# Plot mutual information scores
plt.figure(figsize=(10, 8))
mi_scores.plot(kind='bar')
plt.title('Mutual Information Scores')
plt.ylabel('Mutual Information Score')
plt.show()

# Identifying and plotting relationships for continuous features
continuous_features = ['temp', 'atemp', 'hum', 'windspeed']
for feature in continuous_features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=bike_rentals, x=feature, y='rentals')
    plt.title(f'Rentals vs. {feature}')
    plt.xlabel(feature)
    plt.ylabel('Rentals')
    plt.show()

# Correlation analysis for continuous features
correlations = bike_rentals[continuous_features + ['rentals']].corr()
print("Correlation coefficients between continuous features and 'rentals':\n", correlations['rentals'])

# Identifying outliers with IQR
Q1 = bike_rentals['rentals'].quantile(0.25)
Q3 = bike_rentals['rentals'].quantile(0.75)
IQR = Q3 - Q1
outlier_thresholds = {
    'Lower': Q1 - 1.5 * IQR,
    'Upper': Q3 + 1.5 * IQR
}
print("Outlier thresholds:", outlier_thresholds)

# Visualizing outliers with a box plot
plt.figure(figsize=(6, 4))
sns.boxplot(data=bike_rentals, x='rentals')
plt.title('Box Plot of Rentals')
plt.xlabel('Rentals')
plt.show()



# Applying the Box-Cox transformation to the rentals column
# Check if there are any non-positive values in the 'rentals' column
non_positive_rentals = (bike_rentals['rentals'] <= 0).sum()

# If there are non-positive values, we'll adjust by adding a constant before the transformation
# Otherwise, we'll proceed directly with the Box-Cox transformation
if non_positive_rentals > 0:
    # Add a constant to make all values positive
    adjustment_constant = abs(bike_rentals['rentals'].min()) + 1
    bike_rentals['rentals_adjusted'] = bike_rentals['rentals'] + adjustment_constant
else:
    bike_rentals['rentals_adjusted'] = bike_rentals['rentals']

# Apply the Box-Cox transformation
rentals_adjusted_boxcox, fitted_lambda = stats.boxcox(bike_rentals['rentals_adjusted'])

# Add the transformed rentals back to the dataframe
bike_rentals['rentals_boxcox'] = rentals_adjusted_boxcox

# Output the lambda used and the first few rows to verify the transformation
fitted_lambda, bike_rentals[['rentals', 'rentals_adjusted', 'rentals_boxcox']].head()

# Plot the histogram of the transformed data
plt.figure(figsize=(14, 7))

# Original Data Histogram
plt.subplot(1, 2, 1)
sns.histplot(bike_rentals['rentals'], kde=True)
plt.title('Original Data Distribution')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')

# Transformed Data Histogram
plt.subplot(1, 2, 2)
sns.histplot(bike_rentals['rentals_boxcox'], kde=True, color='orange')
plt.title('Box-Cox Transformed Data Distribution')
plt.xlabel('Transformed Rentals')
plt.tight_layout()
plt.show()

# Q-Q plot for the transformed data
plt.figure(figsize=(10, 5))
stats.probplot(bike_rentals['rentals_boxcox'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Box-Cox Transformed Rentals')
plt.show()

# Perform the Shapiro-Wilk test on the transformed data
shapiro_test_transformed = stats.shapiro(bike_rentals['rentals_boxcox'])
shapiro_test_transformed

bike_rentals






# Refining the Dataset
# Columns to keep based on mutual information score
columns_to_keep = [
    'atemp', 'temp', 'workingday_1', 'season_3', 'yr_1', 'hum', 'season_2',
    'mnth_2', 'weekday_6', 'mnth_6', 'mnth_12', 'rentals'
]
# Refine the bike_rentals dataframe and make a copy to avoid SettingWithCopyWarning
bike_rentals_refined = bike_rentals[columns_to_keep].copy()

# Handling Outliers
# Calculate IQR and determine upper threshold
Q1 = bike_rentals_refined['rentals'].quantile(0.25)
Q3 = bike_rentals_refined['rentals'].quantile(0.75)
IQR = Q3 - Q1
upper_threshold = Q3 + 1.5 * IQR
# Cap the outliers using .loc to ensure the changes are made in place
bike_rentals_refined.loc[bike_rentals_refined['rentals'] > upper_threshold, 'rentals'] = upper_threshold






# Function to compare regression models
def compare_regression_models(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize the models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
    }
    
    scores = {}
    # Train, evaluate, and compare models
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Make predictions
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Calculate RMSE
        r2 = r2_score(y_test, y_pred)  # Calculate R-squared
        scores[name] = {'RMSE': rmse, 'R2': r2}
        print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")
    return scores

# Define features and target variable for the model
X_refined = bike_rentals_refined.drop(['rentals'], axis=1) # Features
y_refined = bike_rentals_refined['rentals'] # Target

# Define features and target variable for the model
# X = bike_rentals.drop(['instant', 'dteday', 'rentals'], axis=1)
# y = bike_rentals['rentals']

# Call the function to compare the models
model_scores = compare_regression_models(X_refined, y_refined)

# Find and print the best model based on RMSE
best_model = min(model_scores, key=lambda k: model_scores[k]['RMSE'])
print(f"Best model: {best_model} with RMSE: {model_scores[best_model]['RMSE']:.2f} and R2: {model_scores[best_model]['R2']:.2f}")


def tune_random_forest(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the Random Forest regressor
    model = RandomForestRegressor()

    # Initialize KFold with the training set size
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Initialize the Grid Search model with 5-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Print best parameters found by grid search
    print("Best Parameters:", grid_search.best_params_)

    # Use the best estimator to make predictions on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate and print the RMSE and R2 score for the test set
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("Test Set RMSE:", rmse)
    print("Test Set R2:", r2)

    return best_model

# Run the tuning function with your features X and target y
best_rf_model = tune_random_forest(X, y)




# Assuming `best_rf_model` is your trained RandomForestRegressor model from the tuning function
feature_importances = best_rf_model.feature_importances_

# Creating a DataFrame for feature importances
features = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sorting the features based on importance
features = features.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=features)
plt.title('Feature Importance in Predicting Bike Rentals')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Check if the data distribution is close to normal
plt.figure(figsize=(8, 6))
sns.histplot(y_refined, kde=True)
plt.title('Distribution of Bike Rentals (Refined Dataset)')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.show()

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.hist(y_refined, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Simulated Rentals')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Q-Q plot
plt.figure(figsize=(10, 5))
stats.probplot(y_refined, dist="norm", plot=plt)
plt.title('Q-Q Plot of Simulated Rentals')
plt.show()

# Perform the Shapiro-Wilk test
shapiro_test = stats.shapiro(y_refined[:1000])  # Shapiro-Wilk test has a limitation on the number of samples
shapiro_test


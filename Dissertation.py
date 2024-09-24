#### PREMILINARY DATA ANALYSIS

# loading the packages for analysis
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd
import seaborn as sns

# Load the dataset
df = pd.read_csv("F:/Msc BA/Dissertation/Fintech_user.csv/Fintech_user.csv")

# View the headers
print("Headers:")
print(df.columns)

# View first five rows of dataset
df.head(5)

# Summarize the dataset
print("\nSummary of the dataset:")
print(df.describe())


####  SUMMARY STATISTICS


# Display basic summary statistics
summary = df.describe(include='all')  # 'all' includes categorical and numerical columns
print(summary)

# for specific data types, you can use:
numeric_summary = df.describe()  # By default, this only includes numeric data
print("Numeric Summary Statistics:")
print(numeric_summary)

# For categorical data
categorical_summary = df.describe(include=['object'])  # This includes categorical columns
print("Categorical Summary Statistics:")
print(categorical_summary)

# Display additional information
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


####  DATA CLEANING  

# Remove user id variable from the database as it is irrelevant for analysis
df.drop(columns=['user'], inplace=True)
# Remove zodiac sign variable as it is not suitable for analysis
df.drop(columns=['zodiac_sign'], inplace=True)


# Find and display columns with NA values
print("\nColumns with NA values:")
na_columns = df.columns[df.isna().any()].tolist()
print(na_columns)

# Impute NA values with the mean for Numerical variable
for column in na_columns:
    if df[column].dtype in ['float64', 'int64']:  # Imputing numeric columns are imputed with mean
        df[column].fillna(df[column].mean(), inplace=True)
    elif df[column].dtype == 'object':  #Imputing categorical columns are imputed with mode
        df[column].fillna(df[column].mode()[0], inplace=True)    
    
# Get the NA values
na_counts = df.isna().sum()

# Print only the columns that have NA values
na_columns = na_counts[na_counts > 0]
print("\nColumns with NA values and their counts:")
print(na_columns)   

# Factorizing categorical variables
df['housing'], uniques = pd.factorize(df['housing'])
df['payment_type'], uniques = pd.factorize(df['payment_type'])

print(df)
print("Unique categories:", uniques)

# Identifying outliers with boxplots
def plot_boxplots(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns].plot(kind='box', subplots=True, layout=(len(numeric_columns) // 3 + 1, 3), figsize=(15, 10), sharex=False, sharey=False)
    plt.tight_layout()
    plt.show()
# Plotting the box plots for each numerical variable in the dataset
plot_boxplots(df)

# Replacing outliers with mean 
df['age'] = np.where(df['age'] >= 75, df['age'].mean(), df['age'])
# removing oultiers in variables
df = df[df['credit_score'] > 100]
df = df[df['purchases_partners'] < 500]
df = df[df['cc_recommended'] < 400]
df = df[df['cc_disliked'] < 20]
df = df[df['cc_liked'] < 5]
df = df[df['cc_taken'] < 15]
df = df[df['cc_application_begin'] < 200]
df = df[df['reward_rate'] < 2.5]
df = df[df['rewards_earned'] < 90]
df = df[df['withdrawal'] < 20]


#### DATA VISUALIZATION 


# Distribution of churn 
# Plot the distribution of the 'churn' variable with count labels on top of the bars
plt.figure(figsize=(10, 6))

ax = sns.countplot(x='churn', data=df, palette='Set2')
plt.title('Distribution of Churn')
plt.xlabel('Churn')
plt.ylabel('Count')

# Add count labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.3, p.get_height() + 200), ha='center', fontsize=12)
plt.show()


# Correlation matrix
# Function to plot correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    # Check if dataframe has a name attribute and set a default if not
    filename = getattr(df, 'dataframeName', 'dataset')    
    # Drop columns with NaN values
    df = df.dropna(axis='columns')  
    # Select only integer and float columns
    df = df.select_dtypes(include=[np.number])    
    # Keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]] 
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return 
    # Calculate the correlation matrix
    corr = df.corr()   
    # Set up the matplotlib figure
    plt.figure(figsize=(graphWidth, graphWidth))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={'shrink': .5}, linewidths=0.5, annot_kws={"size": 6})    
    # Customize the ticks and labels
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=10)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
    
# Plot the correlation matrix
plotCorrelationMatrix(df, graphWidth=10)

# Bar plots

import matplotlib.pyplot as plt
import seaborn as sns
# Set up the plotting environment
sns.set(style="whitegrid")
# Define the categorical variables for creating bar plots
categorical_variables = ['housing', 'cc_liked', 'payment_type',  
                         'cc_taken', 'rejected_loan', 'is_referred']
# Set up the figure for multiple subplots
plt.figure(figsize=(15, 20))
# Loop through variables to create bar plots
for i, var in enumerate(categorical_variables):
    plt.subplot(len(categorical_variables) // 2 + 1, 2, i + 1)
    sns.countplot(x=var, hue='churn', data=df)
    plt.title(f'{var} by Churn Status')
    plt.xlabel(var)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Churn vs. Age
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='churn', multiple='stack', kde=True)
plt.title('Churn vs. Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


## scatterplot 
# Define a list of variable pairs to create scatter plots
scatter_variables = [
    ('rewards_earned', 'purchases_partners'),
]

# Set up the figure for multiple subplots
plt.figure(figsize=(15, 8))

# Loop through variable pairs to create scatter plots
for i, (var1, var2) in enumerate(scatter_variables):
    plt.subplot(len(scatter_variables) // 2 + 1, 2, i + 1)
    sns.scatterplot(data=df, x=var1, y=var2, hue='churn', palette={0: 'blue', 1: 'red'}, edgecolor='w', s=50)
    plt.title(f'{var1} vs. {var2}')
    plt.xlabel(var1)
    plt.ylabel(var2)

plt.tight_layout()
plt.show()


#### SUPERVISED MACHINE LEARNING

# Splitting the dataset
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

# assigning the dataset into a new dataset
dfn = df

# seperating target variables
X = dfn.drop(['churn'], axis=1)
y = dfn['churn']
# splitting the dataset in 8:2 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# MODEL - 1 - RANDOM FOREST

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import lime
import lime.lime_tabular
import shap  # Import SHAP
from imblearn.over_sampling import SMOTE  # Import SMOTE

# Define the number of folds for cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_aucs = []
conf_matrices = []

# Define the hyperparameters grid to search
param_grid = {
    'n_estimators': [300],
    'max_depth': [20],
    'min_samples_split': [5],
    'min_samples_leaf': [2],
    'bootstrap': [False]
}

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X, y):
    # Split the data into training and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    # Fit GridSearchCV on the resampled training data
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    # Get the best parameters
    best_rf_classifier = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_rf_classifier.predict(X_test)
    y_prob = best_rf_classifier.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Store metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_aucs.append(auc(fpr, tpr))

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the cross-validation results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")
print(f"Mean AUC: {np.mean(roc_aucs):.4f}")
print(f"Standard Deviation of AUC: {np.std(roc_aucs):.4f}")

# Plot the last confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices[-1], annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix RF')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC Curve of the last fold
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_aucs[-1])
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.legend(loc='lower right')
plt.show()

# Plot Feature Importances from the last fold
importances = best_rf_classifier.feature_importances_
features = X_train.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importances')
plt.show()

# LIME Explainer for the last fold
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_resampled.values,  # Pass the resampled training data as a NumPy array
    feature_names=X_train.columns.tolist(),  # List of feature names
    class_names=['0', '1'],  # Name your classes
    mode='classification'  # This is a classification problem
)

# Explain a single instance from the last fold
i = 0  # Index of the instance to explain
lime_exp = lime_explainer.explain_instance(X_test.values[i], best_rf_classifier.predict_proba, num_features=5)

# Display the LIME explanation
# Visualize the LIME explanation
lime_exp.as_pyplot_figure()
plt.show()

# SHAP Analysis
# Create SHAP explainer for Random Forest
explainer = shap.TreeExplainer(best_rf_classifier)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Summary plot of SHAP values for the test set (global explanation)
shap.summary_plot(shap_values[1], X_test)

# SHAP force plot for a single instance (local explanation)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0, :], feature_names=X_test.columns)



# MODEL - 2 - ARTIFICIAL NEURAL NETWORKS (ANN)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular

# Define the number of folds for cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_scores = []
roc_aucs = []
conf_matrices = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build the ANN model
    model = Sequential()
    # Add input layer and first hidden layer
    model.add(Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)))
    # Add second hidden layer
    model.add(Dense(units=16, activation='relu'))
    # Add output layer 
    model.add(Dense(units=1, activation='sigmoid'))
    # Compile the ANN model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    # Train the ANN model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Evaluate the ANN model
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)  # Convert probabilities to binary output
    
    # Store accuracy
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    # ROC Curve and AUC
    y_prob = model.predict(X_test)  # Probability estimates for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_aucs.append(auc(fpr, tpr))

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the cross-validation results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")
print(f"Mean AUC: {np.mean(roc_aucs):.4f}")
print(f"Standard Deviation of AUC: {np.std(roc_aucs):.4f}")

# Plot the last confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices[-1], annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix ANN')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC Curve of the last fold
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_aucs[-1])
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - ANN')
plt.legend(loc='lower right')
plt.show()

# Define a function to return both class probabilities
def predict_proba(X):
    prob = model.predict(X)
    return np.hstack([(1 - prob), prob])

# LIME Explainer for the last fold
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X.columns, class_names=['0', '1'], mode='classification')

# Explain a single instance from the last fold
i = 0  # Index of the instance to explain
lime_exp = lime_explainer.explain_instance(X_test[i], predict_proba, num_features=5)
lime_exp.show_in_notebook(show_all=False)

# Visualize the LIME explanation
lime_exp.as_pyplot_figure()
plt.show()


# MODEL - 3 - LOGISTIC REGRESSION

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular


# Define the number of folds for cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_aucs = []
conf_matrices = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X, y):
    # Split the data into training and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize the Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)

    # Fit the model
    log_reg.fit(X_train, y_train)

    # Make predictions
    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Store metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_aucs.append(auc(fpr, tpr))


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the cross-validation results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")
print(f"Mean AUC: {np.mean(roc_aucs):.4f}")
print(f"Standard Deviation of AUC: {np.std(roc_aucs):.4f}")


# Plot the last confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices[-1], annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix LR')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC Curve of the last fold
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_aucs[-1])
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Plot Feature Importance from the last fold
feature_importance = abs(log_reg.coef_[0])
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance - Logistic Regression')
plt.show()

# LIME Explainer for the last fold
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, 
    feature_names=X_train.columns, 
    class_names=['0', '1'], 
    mode='classification'
)

# Explain a single instance from the last fold
i = 0  # Index of the instance to explain
lime_exp = lime_explainer.explain_instance(X_test.values[i], log_reg.predict_proba, num_features=5)

# Display the LIME explanation
lime_exp.show_in_notebook(show_all=False)

# Visualize the LIME explanation as a plot
lime_exp.as_pyplot_figure()
plt.show()


# MODEL - 4 -  Support Vector Machines (SVM)
    
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular


# Define the number of folds for cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_aucs = []
conf_matrices = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X, y):
    # Split the data into training and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize the Support Vector Classifier (SVC)
    svm_model = SVC(probability=True, kernel='rbf', random_state=42)

    # Fit the model
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Store metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_aucs.append(auc(fpr, tpr))

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the cross-validation results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")
print(f"Mean AUC: {np.mean(roc_aucs):.4f}")
print(f"Standard Deviation of AUC: {np.std(roc_aucs):.4f}")

# Plot the last confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices[-1], annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix SVM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC Curve of the last fold
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_aucs[-1])
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - SVM')
plt.legend(loc='lower right')
plt.show()

# LIME Explainer for the last fold
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,  # Pass the training data as a NumPy array
    feature_names=X_train.columns.tolist(),  # List of feature names
    class_names=['Not Churn', 'Churn'],  # Name your classes
    mode='classification'  # This is a classification problem
)

# Explain a single instance from the last fold
i = 0  # Index of the instance to explain
lime_exp = lime_explainer.explain_instance(X_test.values[i], svm_model.predict_proba, num_features=5)

# Display the LIME explanation
lime_exp.show_in_notebook(show_all=False)

# Visualize the LIME explanation as a plot
lime_exp.as_pyplot_figure()
plt.show()

# MODEL - 5 -  GBM

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular


# Define the number of folds for cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_aucs = []
conf_matrices = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X, y):
    # Split the data into training and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize the Gradient Boosting Classifier
    gbm_model = GradientBoostingClassifier(random_state=42)

    # Fit the model
    gbm_model.fit(X_train, y_train)

    # Make predictions
    y_pred = gbm_model.predict(X_test)
    y_prob = gbm_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Store metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_aucs.append(auc(fpr, tpr))

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the cross-validation results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")
print(f"Mean AUC: {np.mean(roc_aucs):.4f}")
print(f"Standard Deviation of AUC: {np.std(roc_aucs):.4f}")


# Plot the last confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices[-1], annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix GBM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC Curve of the last fold
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_aucs[-1])
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - GBM')
plt.legend(loc='lower right')
plt.show()

# Plot Feature Importances from the last fold
importances = gbm_model.feature_importances_
features = X.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importances GBM')
plt.show()

# LIME Explainer for the last fold
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,  # Pass the training data as a NumPy array
    feature_names=X_train.columns.tolist(),  # List of feature names
    class_names=['Not Churn', 'Churn'],  # Name your classes
    mode='classification'  # This is a classification problem
)

# Explain a single instance from the last fold
i = 0  # Index of the instance to explain
lime_exp = lime_explainer.explain_instance(X_test.values[i], gbm_model.predict_proba, num_features=5)

# Display the LIME explanation
lime_exp.show_in_notebook(show_all=False)

# Visualize the LIME explanation as a plot
lime_exp.as_pyplot_figure()
plt.show()


# MODEL - 6 -  XG BOOST

# Import necessary libraries
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular 
from sklearn.model_selection import StratifiedKFold 


# Define the number of folds for cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_aucs = []
conf_matrices = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X, y):
    # Split the data into training and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize the XGBoost Classifier
    xgb_classifier = xgb.XGBClassifier(random_state=42)

    # Fit the model
    xgb_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_classifier.predict(X_test)
    y_prob = xgb_classifier.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Store metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_aucs.append(auc(fpr, tpr))

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the cross-validation results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")
print(f"Mean AUC: {np.mean(roc_aucs):.4f}")
print(f"Standard Deviation of AUC: {np.std(roc_aucs):.4f}")

# Plot the last confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices[-1], annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC Curve of the last fold
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_aucs[-1])
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - XGBoost')
plt.legend(loc='lower right')
plt.show()

# Extract feature importances from the last fold
importances = xgb_classifier.feature_importances_
features = X_train.columns

# Create a DataFrame for plotting
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance using Seaborn
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importance - XGBoost')
plt.show()

# LIME Explainer for the last fold
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,  # Pass the training data as a NumPy array
    feature_names=X_train.columns.tolist(),  # List of feature names
    class_names=['Class 0', 'Class 1'],  # Name your classes
    mode='classification'  # This is a classification problem
)

# Explain a single instance from the last fold
i = 0  # Index of the instance to explain
lime_exp = lime_explainer.explain_instance(X_test.values[i], xgb_classifier.predict_proba, num_features=5)

# Display the LIME explanation
lime_exp.show_in_notebook(show_all=False)

# Visualize the LIME explanation as a plot
lime_exp.as_pyplot_figure()
plt.show()





## MODEL COMPARISIONS - VISUALIZATONS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data for the models
data = {
    'Model': [
        'Random Forest', 
        'XG Boost', 
        'Gradient Boosting Machine (GBM)', 
        'Artificial Neural Network (ANN)', 
        'Support Vector Machine (SVM)', 
        'Logistic regression'
    ],
    'Accuracy': [73.5, 72.19, 70.82, 68.5, 64.91, 64.59],
    'Precision': [73, 72, 71, 68, 64, 64],
    'Recall': [74, 72, 70, 68, 65, 65],
    'F1 score': [73, 72, 71, 68, 63, 64],
    'Support': [8017, 5344, 8907, 8907, 8907, 8907],
    'Mean accuracy': [73.16, 72.3, 70.87, 69.61, 65.7, 65.18],
    'St Dev Accuracy': [0.13, 0.15, 0.17, 0.96, 0.69, 0.42],
    'Mean AUC': [80, 79.06, 76.74, 75.66, 70.14, 69.92],
    'St Dev AUC': [0.07, 0.39, 0.08, 0.3, 0.81, 0.5]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df)


# Bar plot for Accuracy, Precision, Recall, F1 Score

# Set the figure size
plt.figure(figsize=(12, 8))

# Melt the DataFrame for easier plotting
df_melted = pd.melt(df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1 score'])

# Plot accuracy, precision, recall, and F1 score
ax = sns.barplot(x='Model', y='value', hue='variable', data=df_melted)

# Add numerical values on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', 
                fontsize=8, color='black', 
                xytext=(0, 5), 
                textcoords='offset points')

plt.title('Model Comparison on Accuracy, Precision, Recall, and F1 Score')
plt.xticks(rotation=45)
plt.ylabel('Percentage')
plt.show()

# Line Plot for Mean Accuracy and Mean AUC

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot Mean Accuracy and Mean AUC
sns.lineplot(x='Model', y='value', hue='variable', 
             data=pd.melt(df, id_vars=['Model'], value_vars=['Mean accuracy', 'Mean AUC']))

plt.title('Model Comparison on Mean Accuracy and Mean AUC')
plt.xticks(rotation=45)
plt.show()

# Box Plot for Standard Deviation of Accuracy and AUC

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot Standard Deviation of Accuracy and AUC
sns.boxplot(x='variable', y='value', data=pd.melt(df, id_vars=['Model'], value_vars=['St Dev Accuracy', 'St Dev AUC']))

plt.title('Box Plot of Standard Deviation for Accuracy and AUC')
plt.xticks(rotation=45)
plt.show()


## Calibration curves

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier  # Simple ANN using sklearn

    
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XG Boost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Artificial Neural Network (ANN)': MLPClassifier(random_state=42, max_iter=1000),
    'Support Vector Machine (SVM)': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

plt.figure(figsize=(10, 8))

# Plot calibration curve for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    plt.plot(prob_pred, prob_true, marker='o', label=name)

# Plot perfectly calibrated line
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

plt.title('Calibration Curves (Reliability Diagrams)')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.show()



# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import shap  # Import SHAP

# Define the number of folds for cross-validation
n_splits = 3
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_aucs = []
conf_matrices = []

# K-Fold Cross-Validation
for train_index, test_index in kf.split(X, y):
    # Split the data into training and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize the Gradient Boosting Classifier
    gbm_model = GradientBoostingClassifier(random_state=42)

    # Fit the model
    gbm_model.fit(X_train, y_train)

    # Make predictions
    y_pred = gbm_model.predict(X_test)
    y_prob = gbm_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Store metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_aucs.append(auc(fpr, tpr))

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the cross-validation results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")
print(f"Mean AUC: {np.mean(roc_aucs):.4f}")
print(f"Standard Deviation of AUC: {np.std(roc_aucs):.4f}")


# Plot the  confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices[-1], annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix GBM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot ROC Curve of the last fold
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_aucs[-1])
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - GBM')
plt.legend(loc='lower right')
plt.show()

# Plot Feature Importances from the last fold
importances = gbm_model.feature_importances_
features = X.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importances GBM')
plt.show()

# LIME Explainer for the last fold
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,  # Pass the training data as a NumPy array
    feature_names=X_train.columns.tolist(),  # List of feature names
    class_names=['Not Churn', 'Churn'],  # Name your classes
    mode='classification'  # This is a classification problem
)

# Explain a single instance from the last fold
i = 0  # Index of the instance to explain
lime_exp = lime_explainer.explain_instance(X_test.values[i], gbm_model.predict_proba, num_features=5)

# Display the LIME explanation
lime_exp.show_in_notebook(show_all=False)

# Visualize the LIME explanation as a plot
lime_exp.as_pyplot_figure()
plt.show()

# SHAP Explainer for the last fold
shap_explainer = shap.TreeExplainer(gbm_model)

# Calculate SHAP values for the test set
shap_values = shap_explainer.shap_values(X_test)

# Handle the case where SHAP values might only return one output (e.g., for a single class)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Use the positive class for binary classification

# Visualize SHAP values for a single instance (e.g., the first instance in X_test)
shap.initjs()
shap.force_plot(shap_explainer.expected_value, shap_values[0], X_test.iloc[0], feature_names=X_test.columns)

# SHAP summary plot for the test set
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.show()

# SHAP dependence plot for a specific feature (e.g., the most important feature)
plt.figure(figsize=(8, 6))
shap.dependence_plot(importances_df['Feature'].iloc[0], shap_values, X_test)
plt.show()



## collage of confusion matrix

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load all the confusion matrix images
image_paths = [
    'F:/Msc BA/Dissertation/CM - RF.png',  
    'F:/Msc BA/Dissertation/CM ANN.png',
    'F:/Msc BA/Dissertation/CM - XG Boost.png',
    'F:/Msc BA/Dissertation/CM - GBM.png',
    'F:/Msc BA/Dissertation/CM - LR.png',
    'F:/Msc BA/Dissertation/CM - SVM.png'
]

images = [mpimg.imread(img_path) for img_path in image_paths]

# Set up the 3x2 grid for the collage
fig, axs = plt.subplots(2, 3, figsize=(20, 13))

# Plot each image in the grid
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.axis('off')  # Hide axes

plt.tight_layout()
plt.show()


## PR curves

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XG Boost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Artificial Neural Network (ANN)': MLPClassifier(random_state=42, max_iter=1000),
    'Support Vector Machine (SVM)': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

plt.figure(figsize=(10, 8))


# Plot PR curve for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    average_precision = average_precision_score(y_test, y_prob)

    plt.plot(recall, precision, lw=2, label=f'{name} (AP = {average_precision:.2f})')

# Plot the baseline
plt.plot([0, 1], [y_test.mean(), y_test.mean()], linestyle='--', label='No Skill')

plt.title('Precision-Recall Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.show()



## Sensitivity analysis

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Sensitivity analysis function

def sensitivity_analysis(model, X, feature_names, percent_change=0.01):
    # Store sensitivity results in a DataFrame
    sensitivity_results = pd.DataFrame(columns=["Feature", "Original Prediction", "Perturbed Prediction", "Change in Prediction"]) 
    # Generate original predictions
    original_predictions = model.predict_proba(X)[:, 1]
    for i, feature_name in enumerate(feature_names):
        # Create a copy of the data
        X_perturbed = X.copy()
        # Perturb the feature by increasing its value by the given percent change
        X_perturbed[:, i] += X_perturbed[:, i] * percent_change
        # Generate perturbed predictions
        perturbed_predictions = model.predict_proba(X_perturbed)[:, 1]
        # Calculate change in predictions
        change_in_prediction = perturbed_predictions - original_predictions
        # Store the results in a DataFrame
        feature_results = pd.DataFrame({
            "Feature": [feature_name] * len(X),
            "Original Prediction": original_predictions,
            "Perturbed Prediction": perturbed_predictions,
            "Change in Prediction": change_in_prediction
        })
        
        sensitivity_results = pd.concat([sensitivity_results, feature_results], axis=0)   
    return sensitivity_results
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Artificial Neural Network': MLPClassifier(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}
# Fit the models
for name, model in models.items():
    model.fit(X_train, y_train)
# Select a subset of the test set for sensitivity analysis
X_test_subset = X_test[:10]  # Taking first 10 samples for quick analysis
X_test_values = X_test_subset.values  # Convert to numpy array
feature_names = X.columns.tolist()  # List of feature names

# Run sensitivity analysis for all models
for name, model in models.items():
    print(f"Running sensitivity analysis for {name}...")
    sensitivity_results = sensitivity_analysis(model, X_test_values, feature_names)
    print(sensitivity_results.head())  # Display first few results
    print("\n")




import numpy as np
import pandas as pd

def sensitivity_analysis(model, X, feature_names, percent_change=0.1):
    """
    Perform sensitivity analysis by perturbing each feature and observing changes in the model's prediction.

    Args:
    - model: Trained model (should have a .predict() method)
    - X: Input features (numpy array or dataframe)
    - feature_names: List of feature names
    - percent_change: Percentage change to apply for perturbation

    Returns:
    - A dataframe showing the change in predictions for each feature perturbation.
    """
    sensitivity_results = []

    # Loop through each feature
    for i, feature in enumerate(feature_names):
        # Copy the original dataset
        X_perturbed = X.copy()

        # Perturb the feature by increasing it by 'percent_change' percent
        if isinstance(X_perturbed, pd.DataFrame):
            X_perturbed[feature] *= (1 + percent_change)
        else:
            X_perturbed[:, i] *= (1 + percent_change)

        # Get the original predictions
        original_pred = model.predict(X)
        
        # Get the new predictions after perturbing
        new_pred = model.predict(X_perturbed)
        
        # Calculate the change in prediction
        change_in_prediction = new_pred - original_pred
        
        # Store the results
        for sample_idx in range(len(X)):
            sensitivity_results.append({
                "Sample": sample_idx,
                "Feature": feature,
                "Original Prediction": original_pred[sample_idx],
                "New Prediction": new_pred[sample_idx],
                "Change in Prediction": change_in_prediction[sample_idx]
            })
    
    # Return the sensitivity results as a DataFrame
    return pd.DataFrame(sensitivity_results)


# You can perform this for any model: Random Forest, SVM, XGBoost, etc.
sensitivity_results_rf = sensitivity_analysis(rf_classifier, X_test, X_test.columns)
sensitivity_results_xgb = sensitivity_analysis(xgb_classifier, X_test, X_test.columns)

# Display the sensitivity results
sensitivity_results_rf.head()




























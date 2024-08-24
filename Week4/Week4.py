import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the dataset
df = pd.read_csv('vegemite.csv')

# Shuffle the dataset
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

# Get 333 samples from each class to ensure near-equal class distribution
df_class_0 = df[df['Class'] == 0].sample(333, random_state=1)
df_class_1 = df[df['Class'] == 1].sample(333, random_state=1)
df_class_2 = df[df['Class'] == 2].sample(334, random_state=1)  # taking the remainder

# Combine these samples into one DataFrame
df_test = pd.concat([df_class_0, df_class_1, df_class_2]).sample(frac=1, random_state=1).reset_index(drop=True)

# The remaining data will be used for training
df_train = df.drop(df_test.index)

# Verify the distribution in the test dataset
print(df_test['Class'].value_counts())

# Identify constant value columns
constant_columns = [col for col in df_train.columns if df_train[col].nunique() == 1]

# Drop these columns from both training and testing datasets
df_train = df_train.drop(columns=constant_columns)
df_test = df_test.drop(columns=constant_columns)

# Display the constant columns
print('Constant value columns:', constant_columns)

# Identify columns with few unique integer values (e.g., fewer than 10 unique values)
categorical_columns = [col for col in df_train.columns if
                       df_train[col].nunique() < 10 and df_train[col].dtype == 'int64']

# Convert these columns to categorical features in both datasets
for col in categorical_columns:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')

# Display the categorical columns
print("Categorical Columns:", categorical_columns)

# Check the class distribution in the training dataset
class_distribution = df_train['Class'].value_counts()
print("Class Distribution in Training Data:\n", class_distribution)

# Separate features and target variable
X_train = df_train.drop(columns='Class')
y_train = df_train['Class']

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=1)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Combine the resampled data into a new DataFrame
df_train_balanced = pd.concat([X_resampled, y_resampled], axis=1)

# Check the new class distribution
print("Class Distribution after SMOTE:\n", df_train_balanced['Class'].value_counts())

df_train_balanced.to_csv('vegemite_balanced.csv', index=False)

# Compute the absolute correlation matrix
correlation_matrix = abs(df_train_balanced.corr())

# Extract only the lower triangle of the correlation matrix
lower_triangle = np.tril(correlation_matrix, k=-1)

# Mask the upper triangle values in the heatmap
mask = lower_triangle == 0

# Plot a heatmap of the lower triangle of the correlation matrix
plt.figure(figsize=(18, 16))
sns.heatmap(lower_triangle, center=0.5, cmap='coolwarm', annot=True, xticklabels=correlation_matrix.index,
            yticklabels=correlation_matrix.columns, cbar=True, linewidths=1, mask=mask, annot_kws={"size": 6})

# Rotate the x and y-axis labels to prevent them from going out of bounds
plt.xticks(rotation=90, ha='center', fontsize=8)  # Rotate x-axis labels by 90 degrees
plt.yticks(rotation=0, fontsize=8)  # Keep y-axis labels horizontal

plt.title('Correlation Heatmap', fontsize=16, pad=20)
plt.tight_layout(pad=2)  # Adjust the layout to reduce space around the plot
# Use subplots_adjust to move the plot upwards and utilize empty space
plt.subplots_adjust(top=0.95, bottom=0.2)  # top moves the plot up, bottom shifts the bottom edge up
plt.show()

# Select the features with correlation greater than 0.9
df_train_balanced['FFTE Production solids PV * FFTE Discharge solids'] = df_train_balanced['FFTE Production solids PV'] * df_train_balanced['FFTE Discharge solids']
df_train_balanced['FFTE Pump 2 * FFTE Pump 1'] = df_train_balanced['FFTE Pump 2'] * df_train_balanced['FFTE Pump 1']
df_train_balanced['FFTE Temperature 2 - 1 * FFTE Temperature 1 - 1'] = df_train_balanced['FFTE Temperature 2 - 1'] * df_train_balanced['FFTE Temperature 1 - 1']
df_train_balanced['FFTE Temperature 3 - 2 + FFTE Temperature 1 - 1'] = df_train_balanced['FFTE Temperature 3 - 2'] * df_train_balanced['FFTE Temperature 1 - 1']
df_train_balanced['FFTE Temperature 3 - 2 + FFTE Temperature 2 - 1'] = df_train_balanced['FFTE Temperature 3 - 2'] * df_train_balanced['FFTE Temperature 2 - 1']

# Define the composite features to be added to the final DataFrame
composite_features = ['FFTE Production solids PV * FFTE Discharge solids',
                      'FFTE Pump 2 * FFTE Pump 1',
                      'FFTE Temperature 2 - 1 * FFTE Temperature 1 - 1',
                      'FFTE Temperature 3 - 2 + FFTE Temperature 1 - 1',
                      'FFTE Temperature 3 - 2 + FFTE Temperature 2 - 1']

# Select the original feature columns
original_features = [col for col in df_train_balanced.columns if col not in composite_features + ['Class']]

# Combine the original features, composite features, and the target column (Class)
final_columns = original_features + composite_features + ['Class']

# Create the final DataFrame with the specified column order
df_composite_features = df_train_balanced[final_columns]

# Save the final DataFrame to a CSV file for future use
df_composite_features.to_csv('vegemite_composite_features.csv', index=False)

# Calculate the number of features (excluding the 'Class' column)
num_features = df_composite_features.shape[1] - 1  # Subtracting 1 to exclude the target column 'Class'

print(f"Number of features in the final dataset: {num_features}")

# List of original features that were used to create composite features
features_to_remove = [
    'FFTE Production solids PV',
    'FFTE Discharge solids',
    'FFTE Pump 2',
    'FFTE Pump 1',
    'FFTE Temperature 2 - 1',
    'FFTE Temperature 1 - 1',
    'FFTE Temperature 3 - 2'
]

# Remove these features from the dataset
df_final = df_composite_features.drop(columns=features_to_remove)
df_final.to_csv('vegemite_final.csv', index=False)

# Verify the removal by printing the remaining columns
print("Remaining columns after removal of highly correlated features:")
print(df_final.columns)

# Separate features and target variable
X = df_train_balanced.drop(columns='Class')
y = df_train_balanced['Class']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize the models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=1),
    'Random Forest': RandomForestClassifier(random_state=1),
    'SVM': SVC(random_state=1),
    'MLP Classifier': MLPClassifier(random_state=1, max_iter=500),
    'SGD': SGDClassifier(random_state=1)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Model: {name}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("-" * 60)

# Select the best model based on the validation results
best_model = models['Random Forest']

# Define the filename for the model
filename = 'random_forest_model.pkl'

# Save the model to a file using pickle
pickle.dump(best_model, open(filename, 'wb'))

# Select the features with correlation greater than 0.9
df_test['FFTE Production solids PV * FFTE Discharge solids'] = df_test['FFTE Production solids PV'] * df_test['FFTE Discharge solids']
df_test['FFTE Pump 2 * FFTE Pump 1'] = df_test['FFTE Pump 2'] * df_test['FFTE Pump 1']
df_test['FFTE Temperature 2 - 1 * FFTE Temperature 1 - 1'] = df_test['FFTE Temperature 2 - 1'] * df_test['FFTE Temperature 1 - 1']
df_test['FFTE Temperature 3 - 2 + FFTE Temperature 1 - 1'] = df_test['FFTE Temperature 3 - 2'] * df_test['FFTE Temperature 1 - 1']
df_test['FFTE Temperature 3 - 2 + FFTE Temperature 2 - 1'] = df_test['FFTE Temperature 3 - 2'] * df_test['FFTE Temperature 2 - 1']

# Select the original feature columns
original_features = [col for col in df_test.columns if col not in composite_features + ['Class']]

# Combine the original features, composite features, and the target column (Class)
final_columns = original_features + composite_features + ['Class']

# Create the final DataFrame with the specified column order
df_test = df_test[final_columns]

# Verify the removal by printing the remaining columns
print("Remaining columns in test data after removal of highly correlated features:")
print(df_test.columns)

# Load the saved model
loaded_model = pickle.load(open(filename, 'rb'))

# Separate features and target variable from the test dataset
X_test = df_test.drop(columns='Class')
y_test = df_test['Class']

# Make predictions using the loaded model
y_pred_test = loaded_model.predict(X_test)

print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred_test))

print("Confusion Matrix on Test Data:")
print(confusion_matrix(y_test, y_pred_test))

# Test and evaluate other models on the test data
for name, model in models.items():
    if name != 'Random Forest':
        print(f"Evaluating {name} on test data...")
        y_pred_test = model.predict(X_test)
        print(f"Classification Report for {name} on Test Data:")
        print(classification_report(y_test, y_pred_test))
        print(f"Confusion Matrix for {name} on Test Data:")
        print(confusion_matrix(y_test, y_pred_test))
        print("-" * 60)

# Select only the columns that end with 'SP'
sp_columns = [col for col in df_final.columns if col.endswith('SP')]

# Create a new DataFrame with only SP features and the target 'Class'
df_sp = df_final[sp_columns + ['Class']]

# Display the selected SP features
print("Selected SP Features:")
print(df_sp.columns)

# Separate features and target variable
X_sp = df_sp.drop(columns='Class')
y_sp = df_sp['Class']

# Train a Decision Tree model
decision_tree_sp = DecisionTreeClassifier(random_state=1)
decision_tree_sp.fit(X_sp, y_sp)

# Print the tree in a text format
tree_rules = export_text(decision_tree_sp, feature_names=list(X_sp.columns))
print("Decision Tree Rules based on SP features:")
print(tree_rules)

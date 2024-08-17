import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load the datasets
boning_df = pd.read_csv('Boning.csv')
slicing_df = pd.read_csv('Slicing.csv')

# Columns to extract based on student number
columns_to_extract = [
    'Frame',
    'L5 x', 'L5 y', 'L5 z',
    'T12 x', 'T12 y', 'T12 z'
]

# Extract the columns for both activities
boning_extracted = boning_df.loc[:, columns_to_extract].copy()
slicing_extracted = slicing_df.loc[:, columns_to_extract].copy()

# Add the class labels (0 for boning, 1 for slicing)
boning_extracted['Class'] = 0
slicing_extracted['Class'] = 1

# Combine the datasets
combined_df = pd.concat([boning_extracted, slicing_extracted], ignore_index=True)

# Save the combined data
combined_df.to_csv('combined_data.csv', index=False)


# Function to calculate RMS
def rms(*args):
    return np.sqrt(np.mean([x ** 2 for x in args]))


# Calculate composite columns for both column sets
for axis in ['L5', 'T12']:
    combined_df[f'{axis} rms_xy'] = combined_df.apply(lambda row: rms(row[f'{axis} x'], row[f'{axis} y']), axis=1)
    combined_df[f'{axis} rms_yz'] = combined_df.apply(lambda row: rms(row[f'{axis} y'], row[f'{axis} z']), axis=1)
    combined_df[f'{axis} rms_zx'] = combined_df.apply(lambda row: rms(row[f'{axis} z'], row[f'{axis} x']), axis=1)
    combined_df[f'{axis} rms_xyz'] = combined_df.apply(
        lambda row: rms(row[f'{axis} x'], row[f'{axis} y'], row[f'{axis} z']), axis=1)

    combined_df[f'{axis} roll'] = combined_df.apply(
        lambda row: 180 * np.arctan2(row[f'{axis} y'], rms(row[f'{axis} x'], row[f'{axis} z'])) / np.pi, axis=1)
    combined_df[f'{axis} pitch'] = combined_df.apply(
        lambda row: 180 * np.arctan2(row[f'{axis} x'], rms(row[f'{axis} y'], row[f'{axis} z'])) / np.pi, axis=1)

# Reorder columns to match the specified format
final_columns = [
    'Frame',
    'L5 x', 'L5 y', 'L5 z',
    'T12 x', 'T12 y', 'T12 z',
    'L5 rms_xy', 'L5 rms_yz', 'L5 rms_zx', 'L5 rms_xyz', 'L5 roll', 'L5 pitch',
    'T12 rms_xy', 'T12 rms_yz', 'T12 rms_zx', 'T12 rms_xyz', 'T12 roll', 'T12 pitch',
    'Class'
]

combined_df = combined_df[final_columns]

# Save the updated DataFrame
combined_df.to_csv('combined_data_with_composites.csv', index=False)


# Function to compute the statistical features
def compute_features(data):
    features = {}
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['min'] = np.min(data)
    features['max'] = np.max(data)

    # Calculate Area Under the Curve (AUC) using a safer approach
    if len(data) > 1:  # Ensure there are enough points to calculate differences
        auc = np.cumsum(data) - np.concatenate(([0], np.cumsum(np.diff(data) / 2)), axis=0)
        features['auc'] = auc.iloc[-1] if isinstance(auc, pd.Series) else auc[-1]
    else:
        features['auc'] = np.sum(data)  # Fallback for very short or empty data

    # Calculate Peaks
    peaks = 0
    for idx in range(1, len(data) - 1):
        if data.iloc[idx - 1] < data.iloc[idx] > data.iloc[idx + 1] or data.iloc[idx - 1] > data.iloc[idx] < data.iloc[idx + 1]:
            peaks += 1
    features['peaks'] = peaks

    return features


# List of dictionaries to hold the computed features for each segment
features_list = []

# List of columns to compute features for (exclude 'Frame' and 'Class')
columns_to_compute = combined_df.columns[1:-1]  # Columns 2-19

# Process each segment of 60 frames (1 minute)
for i in range(0, len(combined_df), 60):
    segment = combined_df.iloc[i:i + 60]
    segment_features = {}

    # Add frame label to the segment features
    segment_features['Frame'] = combined_df['Frame'].iloc[i]

    for col in columns_to_compute:
        features = compute_features(segment[col])
        for key, value in features.items():
            segment_features[f'{col}_{key}'] = value

    # Add class label to the segment features
    segment_features['Class'] = combined_df['Class'].iloc[i]

    # Append the features dictionary to the list
    features_list.append(segment_features)

# Convert the list of dictionaries to a DataFrame
feature_df = pd.DataFrame(features_list)

# Save the features
feature_df.to_csv('feature_data.csv', index=False)

# Drop the 'Frame' column and separate features (X) and target variable (y)
X = feature_df.drop(columns=['Frame', 'Class'])
y = feature_df['Class']

# Split the data into 70% train and 30% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize the SVM model
clf = svm.SVC(kernel='rbf')

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Measure the accuracy
accuracy_70_30 = accuracy_score(y_test, y_pred)
print(f"SVM Model accuracy with 70/30 train-test split: {accuracy_70_30 * 100:.2f}%")

# Perform 10-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=10)

# Output the cross-validation accuracy scores
cv_mean_accuracy = cv_scores.mean() * 100
print(f"SVM Mean cross-validation accuracy: {cv_mean_accuracy:.2f}%")

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Initialize GridSearchCV
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=10)

# Fit GridSearchCV to the training data
grid.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
best_params = grid.best_params_
print(f"Best parameters found: {best_params}")

# Use the best estimator to make predictions on the test set
best_clf = grid.best_estimator_
y_pred_best = best_clf.predict(X_test)

# Measure the accuracy with the best model on the test split
best_accuracy_70_30 = accuracy_score(y_test, y_pred_best)
print(f"SVM Model accuracy with optimal hyperparameters (70/30 split): {best_accuracy_70_30 * 100:.2f}%")

# Perform 10-fold cross-validation with the best model
best_cv_scores = cross_val_score(best_clf, X, y, cv=10)
best_cv_mean_accuracy = best_cv_scores.mean() * 100
print(f"SVM Mean cross-validation accuracy with optimal hyperparameters: {best_cv_mean_accuracy:.2f}%")

# Select the 10 best features
selector = SelectKBest(f_classif, k=10)
X_best = selector.fit_transform(X, y)

# Split the data again with the selected features
X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X_best, y, test_size=0.3, random_state=1)

# Train the model with the selected features and optimal hyperparameters
best_clf.fit(X_train_best, y_train_best)
y_pred_best = best_clf.predict(X_test_best)

# Measure the accuracy with the selected features on the test split
best_feature_accuracy_70_30 = accuracy_score(y_test_best, y_pred_best)
print(f"SVM Model accuracy with selected features (70/30 split): {best_feature_accuracy_70_30 * 100:.2f}%")

# Perform 10-fold cross-validation with the selected features
best_feature_cv_scores = cross_val_score(best_clf, X_best, y, cv=10)
best_feature_cv_mean_accuracy = best_feature_cv_scores.mean() * 100
print(f"SVM Mean cross-validation accuracy with selected features: {best_feature_cv_mean_accuracy:.2f}%")

# Apply PCA to reduce to 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Split the PCA-reduced data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=1)

# Train the model with PCA-reduced features and optimal hyperparameters
best_clf.fit(X_train_pca, y_train_pca)
y_pred_pca = best_clf.predict(X_test_pca)

# Measure the accuracy with PCA on the test split
best_pca_accuracy_70_30 = accuracy_score(y_test_pca, y_pred_pca)
print(f"SVM Model accuracy with PCA (70/30 split): {best_pca_accuracy_70_30 * 100:.2f}%")

# Perform 10-fold cross-validation with PCA
best_pca_cv_scores = cross_val_score(best_clf, X_pca, y, cv=10)
best_pca_cv_mean_accuracy = best_pca_cv_scores.mean() * 100
print(f"SVM Mean cross-validation accuracy with PCA: {best_pca_cv_mean_accuracy:.2f}%")

# Load the original dataset
combined_df = pd.read_csv('combined_data.csv')

# Drop the 'Frame' column and separate features (X) and target variable (y)
X = combined_df.drop(columns=['Frame', 'Class'])
y = combined_df['Class']

# Standardize the features (SGD and MLP benefit from this)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into 70% train and 30% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize the SGDClassifier
sgd_clf = SGDClassifier(random_state=1)

# Train-test split accuracy
sgd_clf.fit(X_train, y_train)
y_pred_sgd = sgd_clf.predict(X_test)
sgd_train_test_acc = accuracy_score(y_test, y_pred_sgd)
print(f"SGDClassifier accuracy with train-test split: {sgd_train_test_acc * 100:.2f}%")

# 10-fold cross-validation accuracy
sgd_cv_scores = cross_val_score(sgd_clf, X, y, cv=10)
sgd_cv_acc = sgd_cv_scores.mean() * 100
print(f"SGDClassifier cross-validation accuracy: {sgd_cv_acc:.2f}%")

# Initialize the RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=1)

# Train-test split accuracy
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_train_test_acc = accuracy_score(y_test, y_pred_rf)
print(f"RandomForestClassifier accuracy with train-test split: {rf_train_test_acc * 100:.2f}%")

# 10-fold cross-validation accuracy
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=10)
rf_cv_acc = rf_cv_scores.mean() * 100
print(f"RandomForestClassifier cross-validation accuracy: {rf_cv_acc:.2f}%")

# Initialize the MLPClassifier
mlp_clf = MLPClassifier(random_state=1, max_iter=500)

# Train-test split accuracy
mlp_clf.fit(X_train, y_train)
y_pred_mlp = mlp_clf.predict(X_test)
mlp_train_test_acc = accuracy_score(y_test, y_pred_mlp)
print(f"MLPClassifier accuracy with train-test split: {mlp_train_test_acc * 100:.2f}%")

# 10-fold cross-validation accuracy
mlp_cv_scores = cross_val_score(mlp_clf, X, y, cv=10)
mlp_cv_acc = mlp_cv_scores.mean() * 100
print(f"MLPClassifier cross-validation accuracy: {mlp_cv_acc:.2f}%")

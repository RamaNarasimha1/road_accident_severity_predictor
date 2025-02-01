import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("C:/RAS/cleaned (2).csv")

# Select only the relevant columns
columns_to_keep = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation', 
                   'Driving_experience', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                   'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                   'Pedestrian_movement', 'Cause_of_accident', 'Accident_severity']
df = df[columns_to_keep].dropna()

# Split features and target
X = df.drop(['Accident_severity'], axis=1)
y = df['Accident_severity']

# Identify categorical columns
categorical_cols = X.columns.tolist()

# Apply One-Hot Encoding to categorical columns and return dense output
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_cols)
    ],
    remainder='passthrough' 
)

# Apply encoding and scaling
X_encoded = preprocessor.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Handle imbalanced data using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1-score: {f1:.4f}\n")
    
    return accuracy, precision, recall, f1, confusion_matrix(y_test, predictions)

# Hyperparameter tuning and training using RandomizedSearchCV

def tune_and_train(model, param_dist, X_train, y_train, X_test, y_test, model_name, n_iter=20):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=3, n_jobs=-1, verbose=1, random_state=42)
    random_search.fit(X_train, y_train)
    return evaluate_model(random_search.best_estimator_, X_test, y_test, model_name)

# 1. Histogram-based Gradient Boosting
hgb_params = {
    'learning_rate': np.linspace(0.01, 0.2, 5),
    'max_depth': [3, 5, 7, 10],
    'max_iter': [100, 200, 300],
    'l2_regularization': [0, 1, 10]
}
hgb_metrics = tune_and_train(HistGradientBoostingClassifier(), hgb_params, X_train, y_train, X_test, y_test, "HGB")

# 2. Random Forest
rf_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_metrics = tune_and_train(RandomForestClassifier(), rf_params, X_train, y_train, X_test, y_test, "RF")

# 3. SVM
svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['linear', 'rbf']
}
svm_metrics = tune_and_train(SVC(), svm_params, X_train, y_train, X_test, y_test, "SVM")

# 4. KNN
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_metrics = tune_and_train(KNeighborsClassifier(), knn_params, X_train, y_train, X_test, y_test, "KNN")

# Plot metrics for comparison
def plot_bar_graph(metric, values, algorithms):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, values, color='lightblue', edgecolor='black')
    max_index = values.index(max(values))
    bars[max_index].set_edgecolor('red')
    plt.xlabel('Algorithms')
    plt.ylabel(metric)
    plt.title(f'{metric} of Different Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Collect and plot metrics
algorithms = ['HGB', 'RF', 'SVM', 'KNN']
metrics_data = [hgb_metrics, rf_metrics, svm_metrics, knn_metrics]

for i, metric_name in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-score']):
    values = [m[i] for m in metrics_data]
    plot_bar_graph(metric_name, values, algorithms)

# Confusion matrices
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

for algo, metrics in zip(algorithms, metrics_data):
    plot_confusion_matrix(metrics[4], f'{algo} Confusion Matrix')

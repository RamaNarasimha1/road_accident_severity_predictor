import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("C:/RAS/cleaned (2).csv")

# Select only the relevant columns
columns_to_keep = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation', 
                   'Driving_experience', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                   'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                   'Pedestrian_movement', 'Cause_of_accident', 'Accident_severity']
df = df[columns_to_keep]

# Drop rows with missing values
df = df.dropna()

# Split features and target
X = df.drop(['Accident_severity'], axis=1)
y = df['Accident_severity']

# Identify categorical columns
categorical_cols = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation', 
                    'Driving_experience', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                    'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                    'Pedestrian_movement', 'Cause_of_accident']

# Apply One-Hot Encoding to categorical columns and return dense output
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_cols)
    ],
    remainder='passthrough'  # Leave any numerical columns as they are (if any exist)
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

# Function to evaluate and print metrics
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    print(f"{model_name} Accuracy: {accuracy}")
    print(f"{model_name} Precision: {precision}")
    print(f"{model_name} Recall: {recall}")
    print(f"{model_name} F1-score: {f1}")

    return accuracy, precision, recall, f1, confusion_matrix(y_test, predictions)

# Hyperparameter tuning and training

# 1. Histogram-based Gradient Boosting
hgb_params = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'max_iter': [100, 200, 300, 500],
    'l2_regularization': [0, 1, 10]
}
hgb_model = GridSearchCV(HistGradientBoostingClassifier(), hgb_params, cv=5, n_jobs=-1, verbose=2)
hgb_model.fit(X_train, y_train)
hgb_metrics = evaluate_model(hgb_model, X_test, y_test, "Histogram-based Gradient Boosting")

# 2. Random Forest
rf_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_model = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, n_jobs=-1, verbose=2)
rf_model.fit(X_train, y_train)
rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# 3. SVM
svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['linear', 'rbf', 'poly']
}
svm_model = GridSearchCV(SVC(), svm_params, cv=5, n_jobs=-1, verbose=2)
svm_model.fit(X_train, y_train)
svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM")

# 4. KNN
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, n_jobs=-1, verbose=2)
knn_model.fit(X_train, y_train)
knn_metrics = evaluate_model(knn_model, X_test, y_test, "KNN")

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

# Collect metrics
metrics = ['Precision', 'Accuracy', 'F1-score', 'Recall']
precision_values = [hgb_metrics[1], rf_metrics[1], svm_metrics[1], knn_metrics[1]]
accuracy_values = [hgb_metrics[0], rf_metrics[0], svm_metrics[0], knn_metrics[0]]
f1_score_values = [hgb_metrics[3], rf_metrics[3], svm_metrics[3], knn_metrics[3]]
recall_values = [hgb_metrics[2], rf_metrics[2], svm_metrics[2], knn_metrics[2]]
algorithms = ['HGB', 'RF', 'SVM', 'KNN']

# Plot graphs
plot_bar_graph('Precision', precision_values, algorithms)
plot_bar_graph('Accuracy', accuracy_values, algorithms)
plot_bar_graph('F1-score', f1_score_values, algorithms)
plot_bar_graph('Recall', recall_values, algorithms)

# Confusion matrices
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_confusion_matrix(hgb_metrics[4], 'HGB Confusion Matrix')
plot_confusion_matrix(rf_metrics[4], 'RF Confusion Matrix')
plot_confusion_matrix(svm_metrics[4], 'SVM Confusion Matrix')
plot_confusion_matrix(knn_metrics[4], 'KNN Confusion Matrix')

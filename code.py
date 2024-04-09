import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

# Specify the path to your CSV file
data_path = "C:/Users/krama/Downloads/archive (10)/RTADatasetE.csv"
data = pd.read_csv(data_path)

# Define irrelevant columns to ignore
irrelevant_columns = ['Time', 'Day_of_week', 'Educational_level', 'Vehicle_driver_relation', 
                      'Service_year_of_vehicle', 'Defect_of_vehicle', 'Area_accident_occured', 
                      'Number_of_vehicles_involved', 'Number_of_casualties', 'Pedestrian_movement']

# Remove irrelevant columns from the dataset
data_filtered = data.drop(irrelevant_columns, axis=1)

# Preprocessing: Encoding categorical variables
label_encoder = LabelEncoder()
categorical_cols = data_filtered.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data_filtered[col] = label_encoder.fit_transform(data_filtered[col])

# Split data into features (X) and target (y)
X = data_filtered.drop('Accident_severity', axis=1)
y = data_filtered['Accident_severity']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_weights = class_weight.compute_class_weight('balanced', classes=y.unique(), y=y)

# Initialize classifiers with class weights
classifiers = {
    "SVM": SVC(kernel='linear', class_weight=dict(zip(range(len(class_weights)), class_weights))),
    "Decision Tree": DecisionTreeClassifier(class_weight=dict(zip(range(len(class_weights)), class_weights))),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=dict(zip(range(len(class_weights)), class_weights))),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight=dict(zip(range(len(class_weights)), class_weights))),
    "AdaBoost": AdaBoostClassifier(n_estimators=100)
}

# Train and evaluate classifiers
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy:", accuracy)

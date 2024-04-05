# importing the libraries

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load train data
train_data = pd.read_excel("01 Train Data.xlsx")

# Drop rows with missing 'Placement Status'
train_data.dropna(subset=['Placement Status'], inplace=True)

# 'Placed' with 0 and 'Not Placed' with 1
train_data['Placement Status'].replace({'Placed': 0, 'Not placed': 1}, inplace=True)

# Cleaning the data using emails
train_data = train_data.drop_duplicates(subset=["Email ID"])

# Define a function to select useful columns based on relevance to prediction task
def select_useful_columns(data):
    
    useful_columns = ["First Name", "Email ID", "Ticket Type", "Attendee #", "Attendee Status", "College Name",
                      "Designation", "CGPA", "Speaking Skills", "ML Knowledge", "Placement Status"]
    
    # Check if each column exists in the dataset before adding it to the list
    useful_columns = [col for col in useful_columns if col in data.columns]
    
    return useful_columns

# Selecting only useful columns
useful_columns = select_useful_columns(train_data)

# Filter the dataset to keep only useful columns
train_data = train_data[useful_columns]

# Define numerical and categorical columns
numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).drop("Placement Status", axis=1).columns
categorical_columns = train_data.select_dtypes(include=['object']).columns

# Separate features (x_train) and target (y_train)
x = train_data.drop("Placement Status", axis=1)
y = train_data["Placement Status"]

# Split the data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=10)

# preprocessing steps for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'))])

# Combining the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])

# Define the ML model
model = RandomForestClassifier(n_estimators=250, random_state=10)

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model
pipeline.fit(x_train, y_train)

# Perform k-fold cross-validation
scores = cross_val_score(pipeline, x, y, cv=5)

print("Cross-validation scores: ", scores)
print("Average cross-validation score: ", scores.mean())

# Model Evaluation
y_pred = pipeline.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Calculate precision and recall
precision = precision_score(y_val, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_val, y_pred, average='weighted')
print("Validation Precision:", precision)
print("Validation Recall:", recall)

# Load test data
test_data = pd.read_excel("02 Test Data.xlsx")

# Cleaning the data using emails
test_data = test_data.drop_duplicates(subset=["Email ID"])

# Selecting only useful columns (excluding 'Placement Status')
test_data = test_data[useful_columns[:-1]]

# Make predictions on test data
predictions = pipeline.predict(test_data)

# Save output to a separate excel file
output_df = pd.DataFrame({"First Name": test_data["First Name"],
                          "Email ID": test_data["Email ID"],
                          "Placement Status": predictions})
output_df.to_excel("Prediction of Placement.xlsx", index=False)
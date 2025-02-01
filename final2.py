#!/usr/bin/env python
# coding: utf-8

#  # SPORT INJURY ANALYSIS

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams


# **Data Understanding and Exploration:**

# In[4]:


desu = pd.read_csv('final2.csv')


# In[5]:


desu.info()


# In[6]:


desu.head()


# In[7]:


print(f"Dataset shape: {desu.shape}")


# **Check for missing values**

# In[8]:


# Check for missing values
print(desu.isnull().sum())


# In[9]:


#to uderstand the destribution of numerical features
desu.describe()


# In[10]:


desu = desu.rename(columns={'athlete_id':'footballer_id'}) 


# In[11]:


desu.sort_values("postion", ascending = False,inplace= True)


# In[12]:


desu


# In[13]:


"""Maximum workload taken by a player ina game"""
desu["game_workload"].max()


# In[14]:


"""player with maximum worload in a game"""
desu.loc[desu['game_workload'].idxmax()]


# In[15]:


rcParams['figure.figsize'] = 10, 4
for i in range(1,31):
    plt.plot(desu[desu.footballer_id == i]['date'],desu[desu.footballer_id == i]['value'],
             color='blue', linewidth=0.7)
    plt.plot(desu[(desu.footballer_id == i) & (desu.injury == "Yes")]['date'],
             desu[(desu.footballer_id== i) & (desu.injury == "Yes")]['value'],
             color='red', linewidth=1, marker='o')
    plt.grid()
    plt.legend(['performance value', 'injury'])
    plt.xlabel('Timescale')
    plt.ylabel('Amount of Workload')
    plt.title('value and Injury over time for Footballer id:' + str(i))
    plt.show()


# In[16]:


workload_counts = desu.groupby("footballer_id")['date'].count().sort_values().reset_index()


# In[17]:


workload_counts.head()


# In[18]:


injury_counts = desu.groupby("footballer_id").count().reset_index()
injury_counts = injury_counts.drop('date', axis =1)
injury_counts.head()


# In[19]:


player_stats = pd.merge(workload_counts,injury_counts, how='left', left_on=['footballer_id',], right_on = ['value'])


# In[20]:


player_stats.fillna(0, inplace= True)
player_stats.head()


# In[21]:


# Ensure that 'date' is in datetime format
desu['date'] = pd.to_datetime(desu['date'], errors='coerce')

# Extract year and month from 'date'
desu['year'] = desu['date'].dt.strftime('%Y')
desu['month'] = desu['date'].dt.strftime('%m')


# In[22]:


desu.head()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np

# Set the figure size globally
plt.rcParams['figure.figsize'] = (10, 4)

# Get unique footballer IDs
footballer_ids = desu['footballer_id'].unique()

# Loop through each footballer
for footballer in footballer_ids:
    plt.figure()  # Create a new figure
    desu[desu['footballer_id'] == footballer].groupby(['year', 'month'])['footballer_id'].count().plot(kind="bar") 
    plt.title(f'Number of games played per month by Footballer Id {footballer}')
    plt.yticks(np.arange(0, 11, 1))
    plt.xlabel('Months and Years')
    plt.ylabel('Number of matches')
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.close()  # Close the figure to free memory

# Display all the plots (if needed, you can choose to display selectively)
plt.show()


# In[24]:


from pylab import rcParams


# For understanding the distribution of numerical features like performance value and game_workload.

# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of 'value'
plt.figure(figsize=(10, 6))
sns.histplot(desu['value'], bins=30, kde=True)
plt.title('Distribution of Performance Value')
plt.xlabel('Performance Value')
plt.ylabel('Frequency')
plt.show()

# Histogram of 'game_workload'
plt.figure(figsize=(10, 6))
sns.histplot(desu['game_workload'], bins=30, kde=True)
plt.title('Distribution of Game Workload')
plt.xlabel('Game Workload')
plt.ylabel('Frequency')
plt.show()


# Plotting the number of injures occured by each athlete

# In[26]:


rcParams['figure.figsize'] = 10, 4
plt.bar(desu["footballer_id"], desu["injury"])
plt.xticks(desu["footballer_id"])
plt.legend(['injury'])
plt.title("Total Number of Injures occured by players")
plt.xlabel('footballer ids')
plt.ylabel('Number of Injuries')
plt.show()


# In[27]:


"""Observed facts
Non injured player 3,4,6,7,8,9,10,12,14,15,16,17,18,19,31 (0)
least injured player 11,5,13,23,25,28 (1)
 other players are most injured players"""


# In[ ]:





# **by using count plot to see the distribution of injury status**

# In[28]:


# Count plot for injury status
plt.figure(figsize=(8, 5))
sns.countplot(x='injury', data=desu)
plt.title('Injury Status Distribution')
plt.xlabel('Injury Status')
plt.ylabel('Count')
plt.show()


# In[29]:


#  Dropping rows with missing values (if any)
desu.dropna()


# to summarize the data distribution use distribution of catagoricaal features

# In[30]:


# Distribution of categorical features
position_distribution = desu['postion'].value_counts()
injury_distribution = desu['injury'].value_counts()

print("Position Distribution:\n", position_distribution)
print("\nInjury Distribution:\n", injury_distribution)


# # visualizing data distribution

# In[77]:


# Count plot for position
plt.figure(figsize=(10, 6))
sns.countplot(y='postion', data=desu, order=desu['postion'].value_counts().index)
plt.title('Position Distribution')
plt.xlabel('Count')
plt.ylabel('Position')
plt.show()

# Count plot for injury status
plt.figure(figsize=(8, 5))
sns.countplot(x='injury', data=desu)
plt.title('Injury Status Distribution')
plt.xlabel('Injury Status')
plt.ylabel('Count')
plt.show()


# 
# ***Identifying Outliers
# Outliers can be detected using various methods, such as the Interquartile Range (IQR) method or visualizations like box plots. Here’s how to do it using the IQR method:***

# In[79]:


# Function to identify outliers using IQR
def identify_outliers_iqr(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

# Identify outliers for 'value' and 'game_workload'
outliers_value = identify_outliers_iqr(desu, 'value')
outliers_workload = identify_outliers_iqr(desu, 'game_workload')

print(f"Number of outliers in 'performance value': {outliers_value.shape[0]}")
print(f"Number of outliers in 'game_workload': {outliers_workload.shape[0]}")


# In[ ]:





# # Dtat Quality Issues

# In[88]:


# Check for negative values in numerical columns
unrealistic_values = desu[(desu['value'] < 0) | (desu['game_workload'] < 0)]
print("Unrealistic Values:\n", unrealistic_values)


# In[ ]:





# # Handle the outlier in the data preprocessing

# In[32]:


def remove_outliers_iqr(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

# Remove outliers for 'value' and 'game_workload'
desu = remove_outliers_iqr(desu, 'value')
desu = remove_outliers_iqr(desu, 'game_workload')


# In[35]:


from sklearn.preprocessing import LabelEncoder

# Initialize Label Encoder
label_encoder = LabelEncoder()

# Label Encoding for ordinal categorical features (if any)
# Assuming 'injury' is ordinal (e.g., 'no' < 'yes')
desu['injury'] = label_encoder.fit_transform(desu['injury'])

# One-Hot Encoding for nominal categorical features
# Assuming 'postion' is nominal
desu = pd.get_dummies(desu, columns=['value'], drop_first=True)


# In[36]:


desu.head()


# In[38]:


desu


# In[40]:


desu.head()


# In[41]:


print(desu['injury'].isna().sum())


# In[42]:


desu = desu[desu['injury'].notna()]
desu


# In[43]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the data
data = pd.read_csv('final2.csv')

# Preprocess data
if 'date_column' in data.columns:  # Replace 'date_column' with the actual name
    data['date_column'] = pd.to_datetime(data['date_column']).astype(int) / 10**9  # Convert to seconds since epoch

data = pd.get_dummies(data, drop_first=True)

# X = features, y = target variable (e.g., 'injury')
X = data.drop('value', axis=1)
y = data['value']

# Check class distribution
print(y.value_counts())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model with zero_division parameter
print(classification_report(y_test, y_pred, zero_division=0))


# # Splite Dataset

# In[44]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('final2.csv')  # Replace with your actual file path

# Define features (X) and target variable (y)
X = data.drop('injury', axis=1)  # Replace 'injury' with your target variable
y = data['injury']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# # Train the Model

# In[46]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('final2.csv')  # Replace with your actual file path

# Convert date columns to datetime if applicable
# data['date_column'] = pd.to_datetime(data['date_column'], errors='coerce')

# Identify categorical columns (replace with your actual column names)
categorical_cols = ['postion', 'value','footballer_id','date','game_workload']  # Example column names

# One-Hot Encoding for categorical variables
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define features (X) and target variable (y)
X = data.drop('injury', axis=1)  # Replace 'injury' with your target variable
y = data['injury']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[ ]:





# In[48]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Load the dataset
data = pd.read_csv('final2.csv')  # Replace with your actual file path

# Check for missing values
if data.isnull().sum().any():
    print("Missing values detected. Please handle them before proceeding.")
    # You can choose to drop or fill missing values
    # data = data.dropna()  # Example: drop rows with missing values
    # or use data.fillna(method='ffill', inplace=True)  # Example: fill missing values

# Identify categorical columns and encode them
categorical_cols = ['postion', 'value', 'footballer_id', 'date', 'game_workload']  # Example column names
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype(str)  # Ensure categorical columns are of type string

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define features (X) and target variable (y)
if 'injury' not in data.columns:
    raise ValueError("Target variable 'injury' not found in the dataset.")

X = data.drop('injury', axis=1)  # Replace 'injury' with your target variable
y = data['injury']  # Target variable

# Check for class imbalance
print("Class distribution:\n", y.value_counts())

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None]  # Updated valid values for max_features
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

# Fit the model on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# Make predictions on the test set using the best found model
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, zero_division=0))


#   # Explanation of Evaluation Metrics
# 

# In[235]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score, 
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(y_test, y_pred):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(set(y_test)), yticklabels=list(set(y_test)))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_classification_report(y_test, y_pred):
    """Plot the classification report heatmap."""
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='YlGnBu', linewidths=0.5)
    plt.title('Classification Report Heatmap')
    plt.show()

def plot_roc_curve_multiclass(best_rf_model, X_test, y_test):
    """Plot ROC curve for binary or multiclass classification."""
    n_classes = len(set(y_test))
    
    # Binarize output for multiclass ROC curve plotting
    y_test_bin = label_binarize(y_test, classes=list(set(y_test)))
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i], best_rf_model.predict_proba(X_test)[:, i], name=f"Class {i} ROC"
        )
    plt.title("ROC Curve (Multiclass)")
    plt.show()

def plot_precision_recall_curve_multiclass(best_rf_model, X_test, y_test):
    """Plot Precision-Recall curve for binary or multiclass classification."""
    n_classes = len(set(y_test))
    
    # Binarize output for multiclass precision-recall curve plotting
    y_test_bin = label_binarize(y_test, classes=list(set(y_test)))
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        PrecisionRecallDisplay.from_predictions(
            y_test_bin[:, i], best_rf_model.predict_proba(X_test)[:, i], name=f"Class {i} PR"
        )
    plt.title("Precision-Recall Curve (Multiclass)")
    plt.show()

# Call visualization functions
plot_confusion_matrix(y_test, y_pred)
plot_classification_report(y_test, y_pred)
plot_roc_curve_multiclass(best_rf_model, X_test, y_test)
plot_precision_recall_curve_multiclass(best_rf_model, X_test, y_test)


# 

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('final2.csv')

# Check if 'date' column exists
if 'date' not in df.columns:
    raise KeyError("The 'date' column is missing in the dataset. Please check the column names.")

# Ensure there are no NaN values in the 'date' column
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Drop rows where 'date' conversion failed (if any)
df = df.dropna(subset=['date'])

# Encode the categorical 'injury' column
label_encoder = LabelEncoder()
df['injury'] = label_encoder.fit_transform(df['injury'])

# Encode the 'postion' column
df['postion'] = label_encoder.fit_transform(df['postion'])

# Extract useful features from 'date'
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Drop the original 'date' column
df.drop('date', axis=1, inplace=True)

# Define features and target variable
X = df.drop('injury', axis=1)
y = df['injury']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preparation completed successfully.")


# # Logistic Regression

# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Initialize Logistic Regression
log_reg = LogisticRegression(random_state=42)

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test)

# Evaluate the model
print("Logistic Regression Evaluation:")
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg, zero_division=1))  # Handling undefined precision
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.2f}")


# # Support Vector Machine (SVM)

# In[52]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Initialize SVM
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
print("SVM Evaluation:")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, zero_division=1))  # Handle undefined precision
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")


# #  K-Nearest Neighbors (KNN)

# In[70]:


import joblib
from sklearn.neighbors import KNeighborsClassifier

# Initialize KNN
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model
print("KNN Evaluation:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")

joblib.dump(knn_model,'model_nn.joblib')


# In[ ]:


pip 
install 
joblib;
# Load the model and scaler
import joblib
model = joblib.load('model_nn.joblib')


# Example of making a prediction
new_data = pd.DataFrame({
    'footballer_id': [1],
    'postion': [0],  # Encoded value for 'midfielder'
    'value': [36],
    'game_workload': [178],
    'year': [2016],
    'month': [5],
    'day': [1]
})

# Preprocess the new data
new_data_scaled = scaler.transform(new_data)

# Make a prediction
prediction = model.predict(new_data_scaled)
print(f"Predicted Injury: {'Yes' if prediction[0] == 1 else 'No'}") 


# In[ ]:


import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load('model_nn.joblib')
 # Assuming scaler is also saved

# Get input from the user
footballer_id = int(input("Enter Footballer ID: "))
postion = input("Enter Position (e.g., 'midfielder', 'forward', 'defender'): ")
# You need to encode the position input as per your model's requirements.
# Assuming positions are encoded as integers; you might need to map them manually.
position_map = {'midfielder': 0, 'forward': 1, 'defender': 2}  # Example encoding map
encoded_position = position_map.get(postion.lower(), -1)

value = float(input("Enter Footballer Value: "))
game_workload = float(input("Enter Game Workload: "))
year = int(input("Enter Year: "))
month = int(input("Enter Month: "))
day = int(input("Enter Day: "))

# Check for invalid position input
if encoded_position == -1:
    print("Invalid position entered.")
else:
    # Prepare the new data for prediction
    new_data = pd.DataFrame({
        'footballer_id': [footballer_id],
        'postion': [encoded_position],  # Encoded position
        'value': [value],
        'game_workload': [game_workload],
        'year': [year],
        'month': [month],
        'day': [day]
    })

    # Preprocess the new data using the scaler
    new_data_scaled = scaler.transform(new_data)

    # Make a prediction
    prediction = model.predict(new_data_scaled)
    print(f"Predicted Injury: {'Yes' if prediction[0] == 1 else 'No'}")


# In[ ]:


pip 
install;
fastapi ;
uvicorn ;
joblib;
pandas;


# In[ ]:


pip 
install;
nbconvert;


# In[15]:


import nbformat
from nbconvert import PythonExporter

# Load the notebook
notebook_filename = "final2.ipynb"
with open(notebook_filename, 'r', encoding='utf-8') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to Python script
python_exporter = PythonExporter()
script, _ = python_exporter.from_notebook_node(notebook_content)

# Save the script
python_filename = "final2.py"
with open(python_filename, 'w', encoding='utf-8') as f:
    f.write(script)

print(f"Converted {notebook_filename} to {python_filename}")


# In[ ]:


pip 
install 
fastapi ;
uvicorn ;
joblib ;pandas


# In[ ]:


pip 
fastapi 
uvicorn 
pydantic 
joblib;


# In[2]:


import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# In[3]:


# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Save the trained model using Joblib
joblib.dump(rf_model, 'model_nn.joblib')






# In[4]:


import joblib
import numpy as np

# Load the trained model
rf_model = joblib.load('model_nn.joblib')

# Example data for prediction (you can replace this with new data)
X_new = np.array([[0.5, -1.2, 3.1, 0.7]])

# Make a prediction
prediction = rf_model.predict(X_new)

# Output the prediction
print(f"Predicted Class: {prediction[0]}")


# In[46]:


get_ipython().system('uvicorn final2:app --no-reload')


# In[ ]:





# In[50]:


get_ipython().system('uvicorn final2:app --reload')


# In[57]:


import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the trained model and scaler
model = joblib.load('model_nn.joblib')
# scaler = joblib.load('scaler.joblib')

# Define the input data schema using Pydantic
class InjuryPredictionInput(BaseModel):
    athlete_id: int
    postion: str
    value: float
    game_workload: int
    year: int
    month: int
    day: int

# Initialize FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict_injury(input_data: InjuryPredictionInput):
    try:
        # Convert input data to a DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Encode the 'postion' column (if necessary)
        input_df['postion'] = input_df['postion'].map({'midfielder': 0, 'attacker': 1})

        # Standardize the features
        input_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Return the prediction result
        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0][1]),
            "message": "Injury predicted successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sports Injury Prediction API!"}


# In[60]:


pip 
install
fastapi 
uvicorn


# In[62]:


get_ipython().system('uvicorn final2:app --reload')


# In[63]:


get_ipython().system('uvicorn final2:app --host 0.0.0.0 --port 8000 --reload')


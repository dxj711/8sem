import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Import machine learning models
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc


import optuna

# Streamlit app title
st.title("Proactive Maintenance Analysis")

# Read the dataset
filename = "predictive_maintenance_dataset.csv"
df = pd.read_csv(filename)

# EDA
st.header("Exploratory Data Analysis (EDA)")

# Display dataset shape
st.write("Dataset Shape:", df.shape)

# Drop duplicates
df.drop_duplicates(inplace=True)
st.write("Dataset Shape after dropping duplicates:", df.shape)

# Scatter plot
st.subheader("Scatter Plot between Metric7 and Metric8")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['metric7'], df['metric8'], alpha=0.5)
ax.set_title('Scatter Plot between Metric7 and Metric8')
ax.set_xlabel('Metric7')
ax.set_ylabel('Metric8')
ax.grid(True)
st.pyplot(fig)

# Log transformation
for num in ["2","3","4","7","8","9"]:
    df[f'metric{num}'] = np.log1p(df[f'metric{num}'])

# Scatter plot after log transformation
st.subheader("Scatter Plot between Metric7 and Metric8 after Log Transformation")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['metric7'], df['metric8'], alpha=0.5)
ax.set_title('Scatter Plot between Metric7 and Metric8')
ax.set_xlabel('Metric7')
ax.set_ylabel('Metric8')
ax.grid(True)
st.pyplot(fig)

# Drop metric8
df.drop("metric8", axis=1, inplace=True)

# Summarize data
st.subheader("Data Summary")
def summarize_data(df):
    st.write("Number of rows and columns:", df.shape)
    st.write("\nColumns in the dataset:", df.columns)
    st.write("\nData types and missing values:")
    st.write(df.info())
    st.write("\nSummary statistics for numerical columns:")
    st.write(df.describe())
    st.write("\nMissing values:")
    st.write(df.isnull().sum())
    st.write("\nUnique values in 'failure' column:")
    st.write(df['failure'].value_counts())

summarize_data(df)

# Device model extraction
df["device_model"] = df["device"].apply(lambda x: x[:4])
df["device_rest"] =  df["device"].apply(lambda x: x[4:])
df.drop("device", axis=1, inplace=True)

# Distribution plots
st.subheader("Distribution of Failure with respect to Device Model")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x="device_model", data=df.loc[df["failure"] == 1], ax=ax)
ax.set_title('Distribution of Failure (failure=1) with respect to Device')
st.pyplot(fig)

# Drop Z1F2
df.drop(df.loc[df["device_model"] == "Z1F2"].index, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop device_rest
df.drop("device_rest", axis=1, inplace=True)

# Histograms for failure=0
st.subheader("Distribution of Metrics for Failure=0")
fig, ax = plt.subplots(figsize=(20, 10))
mask = df.failure == 0
for i, col in enumerate(['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric9']):
    plt.subplot(2, 4, i + 1)
    sns.histplot(data=df.loc[mask], x=col, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
st.pyplot(fig)

# Histograms for failure=1
st.subheader("Distribution of Metrics for Failure=1")
fig, ax = plt.subplots(figsize=(20, 10))
mask = df.failure > 0
for i, col in enumerate(['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 'metric6', 'metric7', 'metric9']):
    plt.subplot(2, 4, i + 1)
    sns.histplot(data=df.loc[mask], x=col, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
st.pyplot(fig)

# Date transformations
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').dt.strftime('%Y-%m')
df['week'] = df['date'].dt.to_period('W').dt.strftime('%Y-%U')

# Line plot for failure over time by month
st.subheader("Failure over Time by Month")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df, x='month', y='failure', ax=ax)
plt.xticks(rotation=45)  # Fix: Rotate x-axis labels
ax.set_title("Failure over Time by Month")
st.pyplot(fig)

# Line plot for failure over time by week
st.subheader("Failure over Time by Week")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df, x='week', y='failure', ax=ax)
plt.xticks(rotation=45)  # Fix: Rotate x-axis labels
ax.set_title("Failure over Time by Week")
st.pyplot(fig)


# Correlation matrix
st.subheader("Correlation Matrix")
numeric_cols = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
ax.set_title("Correlation Matrix")
st.pyplot(fig)

# Distribution of failure
st.subheader("Distribution of 'failure'")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='failure', ax=ax)
ax.set_title("Distribution of 'failure'")
st.pyplot(fig)

# Feature engineering
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Distribution plots for day_of_week, day_of_month, and is_weekend
st.subheader("Distribution of Day of Week, Day of Month, and Weekend")
fig, ax = plt.subplots(figsize=(15, 5))
plt.subplot(131)
sns.countplot(data=df, x='day_of_week', palette='Set3')
plt.title("Distribution of day_of_week")
plt.xlabel("Day of Week")
plt.ylabel("Count")

plt.subplot(132)
sns.countplot(data=df, x='day_of_month', palette='Set3')
plt.title("Distribution of day_of_month")
plt.xlabel("Day of Month")
plt.ylabel("Count")

plt.subplot(133)
sns.countplot(data=df, x='is_weekend', palette='Set3')
plt.title("Distribution of is_weekend")
plt.xlabel("Weekend (1) or Weekday (0)")
plt.ylabel("Count")
plt.tight_layout()
st.pyplot(fig)

# Drop date column
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week
df = df.drop(['date'], axis=1)

# Feature selection using Random Forest
X = df.drop(columns=['failure'])
y = df['failure']
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
selected_features = feature_importances[feature_importances > 0.01].index.tolist()

# Filter dataset with important features
X_filtered = df[selected_features]
y_filtered = df['failure']


# Undersampling
X = df.copy()
Y = df["failure"]
X.drop("failure", axis=1, inplace=True)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, Y)
under_sample = X_resampled.copy()
under_sample["failure"] = y_resampled

# Display undersampled data
st.subheader("Undersampled Data")
st.write(under_sample.sample(10))

# Distribution of failure in undersampled data
st.subheader("Distribution of 'failure' in Undersampled Data")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=under_sample, x='failure', ax=ax)
ax.set_title("Distribution of 'failure'")
st.pyplot(fig)

# Undersampling for balancing data
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_filtered, y_filtered)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model evaluation
st.header("Model Evaluation")


# Reshape data for LSTM
x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_lstm = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


def evaluate_model(x_train, y_train, x_test, y_test):
    classifiers = [
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        ExtraTreesClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        GaussianNB(),
        BernoulliNB(),
        SVC(),
        LogisticRegression(),
        SGDClassifier(),
    ]

    classifier_names = [
        'GradientBoost',
        'RandomForest',
        'AdaBoost',
        'ExtraTrees',
        'DecisionTree',
        'KNeighbors',
        'GaussianNB',
        'BernoulliNB',
        'SVC',
        'LogisticRegression',
        'SGD',
    ]

    metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1'], index=classifier_names)

    for i, clf in enumerate(classifiers):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics.loc[classifier_names[i], 'Accuracy'] = accuracy
        metrics.loc[classifier_names[i], 'Precision'] = precision
        metrics.loc[classifier_names[i], 'Recall'] = recall
        metrics.loc[classifier_names[i], 'F1'] = f1

    metrics = metrics.sort_values(by='Accuracy', ascending=False)
    return metrics

metrics = evaluate_model(x_train, y_train, x_test, y_test)
st.write("Model Evaluation Metrics:")
st.write(metrics)

# Hyperparameter tuning using Optuna
st.header("Hyperparameter Tuning using Optuna")

def create_study(objective):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    best_f1 = study.best_value
    st.write(f'Best hyperparameters: {best_params}')
    st.write(f'Best f1 score: {best_f1}')
    return best_params





# GradientBoosting
def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_gb = create_study(objective_gb)
best_gb = GradientBoostingClassifier(**best_params_gb, random_state=42)
y_pred_gb = best_gb.fit(x_train, y_train).predict(x_test)

# RandomForest
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 150),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': trial.suggest_uniform('min_samples_split', 0.1, 1.0),
        'min_samples_leaf': trial.suggest_uniform('min_samples_leaf', 0.1, 0.5),
        'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt']),
    }
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_rf = create_study(objective_rf)
best_rf = RandomForestClassifier(**best_params_rf, random_state=42)
y_pred_rf = best_rf.fit(x_train, y_train).predict(x_test)


#Adaboost
def objective_ab(trial):
    # Define the hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 50, 200),  # Number of weak learners
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 1.0),  # Learning rate
        'algorithm': trial.suggest_categorical("algorithm", ["SAMME"]),  # Only 'SAMME' is allowed
    }

    # Create an AdaBoostClassifier with the suggested hyperparameters
    model = AdaBoostClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        algorithm=params['algorithm'],
        random_state=42
    )

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(x_test)

    # Calculate F1 score as the objective to maximize
    f1 = f1_score(y_test, y_pred)

    return f1

# Create study and optimize
best_params_ab = create_study(objective_ab)
best_ab = AdaBoostClassifier(**best_params_ab, random_state=42)
y_pred_ab = best_ab.fit(x_train, y_train).predict(x_test)

# ExtraTrees
def objective_etc(trial):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 1, 32),
        'min_samples_split': trial.suggest_float("min_samples_split", 0.1, 1.0),
        'min_samples_leaf': trial.suggest_float("min_samples_leaf", 0.1, 0.5),
    }
    clf = ExtraTreesClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_etc = create_study(objective_etc)
best_etc = ExtraTreesClassifier(**best_params_etc, random_state=42)
y_pred_etc = best_etc.fit(x_train, y_train).predict(x_test)

# Decision Tree
def objective_dt(trial):
    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
        'min_samples_split': trial.suggest_uniform('min_samples_split', 0.1, 1.0),
        'min_samples_leaf': trial.suggest_uniform('min_samples_leaf', 0.1, 0.5),
    }
    clf = DecisionTreeClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_dt = create_study(objective_dt)
best_dt = DecisionTreeClassifier(**best_params_dt, random_state=42)
y_pred_dt = best_dt.fit(x_train, y_train).predict(x_test)

# KNN
def objective_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 2),
    }
    clf = KNeighborsClassifier(**params)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_knn = create_study(objective_knn)
best_knn = KNeighborsClassifier(**best_params_knn)
y_pred_knn = best_knn.fit(x_train, y_train).predict(x_test)

# GaussianNB
best_gnb = GaussianNB()
y_pred_gnb = best_gnb.fit(x_train, y_train).predict(x_test)

# BernoulliNB
def objective_bnb(trial):
    params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-10, 1.0),
        'binarize': trial.suggest_float('binarize', 0.0, 1.0),
        'fit_prior': trial.suggest_categorical('fit_prior', [True, False]),
    }
    clf = BernoulliNB(**params)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_bnb = create_study(objective_bnb)
best_bnb = BernoulliNB(**best_params_bnb)
y_pred_bnb = best_bnb.fit(x_train, y_train).predict(x_test)

# SVC
def objective_svc(trial):
    params = {
        'C': trial.suggest_loguniform('C', 1e-3, 1e3),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'degree': trial.suggest_int('degree', 2, 5) if trial.params['kernel'] == 'poly' else 1,
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.params['kernel'] in ['rbf', 'poly', 'sigmoid'] else 'scale',
    }
    clf = SVC(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_svc = create_study(objective_svc)
best_svc = SVC(**best_params_svc)
y_pred_svc = best_svc.fit(x_train, y_train).predict(x_test)

# LogisticRegression
def objective_lr(trial):
    params = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e5),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
    }
    clf = LogisticRegression(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_lr = create_study(objective_lr)
best_lr = LogisticRegression(**best_params_lr)
y_pred_lr = best_lr.fit(x_train, y_train).predict(x_test)


# SGDClassifier
def objective_sgd(trial):
    # Define hyperparameters to optimize
    params = {
        'loss': trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber']),  # Updated 'log' to 'log_loss'
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
        'alpha': trial.suggest_loguniform('alpha', 1e-6, 1e-1),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
        'eta0': trial.suggest_loguniform('eta0', 1e-5, 1e-1),
    }

    # Initialize the classifier with hyperparameters
    clf = SGDClassifier(**params, random_state=42)

    # Train the classifier on the training data
    clf.fit(x_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(x_test)
    
    # Calculate F1 score as the objective to maximize
    f1 = f1_score(y_test, y_pred)

    return f1




# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train_lstm, y_train, epochs=20, batch_size=32, validation_data=(x_test_lstm, y_test))

# Model evaluation
y_pred_lstm = (model.predict(x_test_lstm) > 0.5).astype(int)


# Calculate evaluation metrics
accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
precision_lstm = precision_score(y_test, y_pred_lstm)
recall_lstm = recall_score(y_test, y_pred_lstm)
f1_lstm = f1_score(y_test, y_pred_lstm)

# Print evaluation metrics
st.write("LSTM Model Evaluation Metrics:")
st.write(f"Accuracy: {accuracy_lstm:.4f}")
st.write(f"Precision: {precision_lstm:.4f}")
st.write(f"Recall: {recall_lstm:.4f}")
st.write(f"F1 Score: {f1_lstm:.4f}")








# Create study and optimize
best_params_sgd = create_study(objective_sgd)
best_sgd = SGDClassifier(**best_params_sgd, random_state=42)
y_pred_sgd = best_sgd.fit(x_train, y_train).predict(x_test)

# Voting Classifier
voting_clf = VotingClassifier(estimators=[('gb', best_gb), ('rf', best_rf), ('ab', best_ab), ('etc', best_etc), ('dt', best_dt), ('knn', best_knn), ('gnb', best_gnb), ('bnb', best_bnb), ('svc', best_svc), ('lr', best_lr), ('sgd', best_sgd)], voting='hard')
voting_clf.fit(x_train, y_train)
y_pred_vh = voting_clf.predict(x_test)

# Model comparison
st.header("Model Comparison")

def calculate_evaluation_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def plot_confusion_matrix(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, linecolor="black", cbar=False, xticklabels=["Non-Failure", "Failure"], yticklabels=["Non-Failure", "Failure"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

best_model = ""
best_f1 = 0.0
best_precision = 0.0
best_recall = 0.0
best_accuracy = 0.0

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(3*5, 4*5))
models = [
    ("Model Gradient Boosting", y_pred_gb),
    ("Model Random Forest", y_pred_rf),
    ("Model AdaBoost", y_pred_ab),
    ("Model Extra Tree", y_pred_etc),
    ("Decison Tree", y_pred_dt),
    ("KNN", y_pred_knn),
    ("GaussianNB", y_pred_gnb),
    ("BernoulliNB", y_pred_bnb),
    ("SVC", y_pred_svc),
    ("LogisticRegression", y_pred_lr),
    ("SGDClassifier", y_pred_sgd),
    ("Hard Voting Classifier", y_pred_vh),
    ("LSTM Model", y_pred_lstm)
]

for (model_name, y_pred), ax in zip(models, axes.flatten()):
    plot_confusion_matrix(ax, y_test, y_pred, f"Confusion Matrix - {model_name}")
    precision, recall, f1, accuracy = calculate_evaluation_metrics(y_test, y_pred)
    st.write(f"\nModel: {model_name}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"Accuracy: {accuracy:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model_name
        best_precision = precision
        best_recall = recall
        best_accuracy = accuracy

plt.tight_layout()
st.pyplot(fig)

st.write("\n=====Best Model=====\n")
st.write(f"Model: {best_model}")
st.write(f"Precision: {best_precision:.4f}")
st.write(f"Recall: {best_recall:.4f}")
st.write(f"F1 Score: {best_f1:.4f}")
st.write(f"Accuracy: {best_accuracy:.4f}")


# Predictive Maintenance Scheduling
st.header("Predictive Maintenance Scheduling")

# Ensure the best model (identified earlier) is used for predictions
if best_model == "Model Gradient Boosting":
    best_final_model = best_gb
elif best_model == "Model Random Forest":
    best_final_model = best_rf
elif best_model == "Model AdaBoost":
    best_final_model = best_ab
elif best_model == "Model Extra Tree":
    best_final_model = best_etc
elif best_model == "Decison Tree":
    best_final_model = best_dt
elif best_model == "KNN":
    best_final_model = best_knn
elif best_model == "GaussianNB":
    best_final_model = best_gnb
elif best_model == "BernoulliNB":
    best_final_model = best_bnb
elif best_model == "SVC":
    best_final_model = best_svc
elif best_model == "LogisticRegression":
    best_final_model = best_lr
elif best_model == "SGDClassifier":
    best_final_model = best_sgd
elif best_model == "Hard Voting Classifier":
    best_final_model = voting_clf
else:
    st.write("No best model found. Please check model evaluation results.")
    best_final_model = None

# Predict failure probability and schedule maintenance
if best_final_model:
    try:
        # Drop 'failure' column if it exists before making predictions
        df_features = df.drop(columns=["failure"], errors="ignore")

        # Ensure the number of samples is correct
        if df_features.shape[0] == 0:
            st.write("Error: No valid data for prediction.")
        else:
            failure_probabilities = best_final_model.predict_proba(df_features)[:, 1]
            df["failure_probability"] = failure_probabilities

            # Define a threshold for scheduling maintenance
            maintenance_threshold = st.slider("Set Failure Probability Threshold for Maintenance", 0.5, 1.0, 0.8)

            upcoming_maintenance = df[df["failure_probability"] > maintenance_threshold]

            st.subheader("Machines Recommended for Maintenance")
            if not upcoming_maintenance.empty:
                st.write("The following machines have a high probability of failure and should be scheduled for maintenance:")
                st.dataframe(upcoming_maintenance[["failure_probability"] + list(df_features.columns)])
            else:
                st.write("No immediate maintenance required based on the selected threshold.")
    except Exception as e:
        st.write(f"Error during prediction: {str(e)}")



# ROC Curve
st.header("ROC Curve")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_ab)
roc_auc = auc(fpr, tpr)
st.write("AUC:", roc_auc)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

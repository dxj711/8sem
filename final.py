import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings("ignore")
import streamlit as st

# ----------------------- LSTM Integration Start -----------------------
# Import necessary PyTorch libraries
class CNN_BiLSTMAttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(CNN_BiLSTMAttentionNet, self).__init__()
        # Convolutional front-end to capture local patterns
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        # LSTM: note that after CNN, the feature dimension becomes 64
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        # Fully-connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (batch, seq_length, input_size)
        # Permute to (batch, input_size, seq_length) for CNN
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # Permute back: (batch, seq_length, 64)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)  # (batch, seq_length, hidden_size*2)
        # Apply attention over time steps
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_length, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        out = self.fc(context)
        return out


class LSTMClassifier:
    def __init__(self, input_size, sequence_length=20, hidden_size=256, num_layers=3,
                 dropout=0.4, epochs=200, batch_size=64, learning_rate=0.0002,
                 patience=15, verbose=True, device=None):
        self.input_size = input_size
        self.sequence_length = sequence_length  # This must match your evaluation slice
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.verbose = verbose
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN_BiLSTMAttentionNet(input_size, hidden_size, num_layers, dropout).to(self.device)

    def create_sequences(self, X, y=None, is_prediction=False):
        sequences = []
        X_array = np.array(X)
        total_samples = len(X_array)
        seq_len = self.sequence_length
        # Generate (total_samples - seq_len + 1) sequences
        if is_prediction:
            for i in range(total_samples - seq_len + 1):
                sequences.append(X_array[i:i + seq_len])
            return np.array(sequences)
        else:
            targets = []
            for i in range(total_samples - seq_len + 1):
                sequences.append(X_array[i:i + seq_len])
                targets.append(np.array(y)[i + seq_len - 1])
            return np.array(sequences), np.array(targets)

    def fit(self, X, y):
        X_seq, y_seq = self.create_sequences(X, y)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=self.verbose)
        
        best_loss = np.inf
        patience_counter = 0
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(loader.dataset)
            scheduler.step(epoch_loss)
            if self.verbose:
                st.write(f'Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}')
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                if self.verbose:
                    st.write(f"Early stopping at epoch {epoch+1}")
                break

    def predict_proba(self, X):
        self.model.eval()
        X_seq = self.create_sequences(X, is_prediction=True)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        # This produces (N - sequence_length + 1) predictions,
        # which should match y_test.values[sequence_length - 1:] in your evaluation.
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


# ----------------------- LSTM Integration End -----------------------

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

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Undersampling
X = df.copy()
Y = df["failure"]
X.drop("failure", axis=1, inplace=True)
from imblearn.under_sampling import RandomUnderSampler
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

# Train-test split and standardization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_norm = under_sample.drop(['failure'], axis=1)
y_norm = under_sample['failure']
x_train, x_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ----------------------- Model Evaluation including LSTM -----------------------
st.header("Model Evaluation")

def evaluate_model(x_train, y_train, x_test, y_test):
    # List of traditional classifiers
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
    
    classifiers = [
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        ExtraTreesClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        GaussianNB(),
        BernoulliNB(),
        SVC(probability=True),  # set probability=True for predict_proba
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
    
    # Append the LSTM model into the evaluation framework.
    sequence_length = 5  # Choose suitable sequence length for your data

    lstm_model_for_eval = LSTMClassifier(
        input_size=x_train.shape[1],
        sequence_length=sequence_length,
        epochs=50, 
        hidden_size=64, 
        num_layers=2, 
        dropout=0.3,
        patience=5,
        learning_rate=0.001, 
        verbose=False
    )
    classifiers.append(lstm_model_for_eval)
    classifier_names.append("LSTM")
    
    metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1'], index=classifier_names)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    for i, clf in enumerate(classifiers):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # Adjustment for LSTM predictions length mismatch
        if classifier_names[i] == "LSTM":
            y_test_aligned = y_test.values[sequence_length - 1:]  # fix length mismatch
        else:
            y_test_aligned = y_test.values

        accuracy = accuracy_score(y_test_aligned, y_pred)
        precision = precision_score(y_test_aligned, y_pred)
        recall = recall_score(y_test_aligned, y_pred)
        f1 = f1_score(y_test_aligned, y_pred)

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
import optuna

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
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_gb = create_study(objective_gb)
from sklearn.ensemble import GradientBoostingClassifier
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
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_rf = create_study(objective_rf)
from sklearn.ensemble import RandomForestClassifier
best_rf = RandomForestClassifier(**best_params_rf, random_state=42)
y_pred_rf = best_rf.fit(x_train, y_train).predict(x_test)


# Adaboost
def objective_ab(trial):
    from sklearn.ensemble import AdaBoostClassifier
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 50, 200),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 1.0),
        'algorithm': trial.suggest_categorical("algorithm", ["SAMME"]),
    }
    model = AdaBoostClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        algorithm=params['algorithm'],
        random_state=42
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_ab = create_study(objective_ab)
from sklearn.ensemble import AdaBoostClassifier
best_ab = AdaBoostClassifier(**best_params_ab, random_state=42)
y_pred_ab = best_ab.fit(x_train, y_train).predict(x_test)

# ExtraTrees
def objective_etc(trial):
    from sklearn.ensemble import ExtraTreesClassifier
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 1, 32),
        'min_samples_split': trial.suggest_float("min_samples_split", 0.1, 1.0),
        'min_samples_leaf': trial.suggest_float("min_samples_leaf", 0.1, 0.5),
    }
    clf = ExtraTreesClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_etc = create_study(objective_etc)
from sklearn.ensemble import ExtraTreesClassifier
best_etc = ExtraTreesClassifier(**best_params_etc, random_state=42)
y_pred_etc = best_etc.fit(x_train, y_train).predict(x_test)

# Decision Tree
def objective_dt(trial):
    from sklearn.tree import DecisionTreeClassifier
    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
        'min_samples_split': trial.suggest_uniform('min_samples_split', 0.1, 1.0),
        'min_samples_leaf': trial.suggest_uniform('min_samples_leaf', 0.1, 0.5),
    }
    clf = DecisionTreeClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_dt = create_study(objective_dt)
from sklearn.tree import DecisionTreeClassifier
best_dt = DecisionTreeClassifier(**best_params_dt, random_state=42)
y_pred_dt = best_dt.fit(x_train, y_train).predict(x_test)

# KNN
def objective_knn(trial):
    from sklearn.neighbors import KNeighborsClassifier
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 2),
    }
    clf = KNeighborsClassifier(**params)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_knn = create_study(objective_knn)
from sklearn.neighbors import KNeighborsClassifier
best_knn = KNeighborsClassifier(**best_params_knn)
y_pred_knn = best_knn.fit(x_train, y_train).predict(x_test)

# GaussianNB
from sklearn.naive_bayes import GaussianNB
best_gnb = GaussianNB()
y_pred_gnb = best_gnb.fit(x_train, y_train).predict(x_test)

# BernoulliNB
def objective_bnb(trial):
    from sklearn.naive_bayes import BernoulliNB
    params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-10, 1.0),
        'binarize': trial.suggest_float('binarize', 0.0, 1.0),
        'fit_prior': trial.suggest_categorical('fit_prior', [True, False]),
    }
    clf = BernoulliNB(**params)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_bnb = create_study(objective_bnb)
from sklearn.naive_bayes import BernoulliNB
best_bnb = BernoulliNB(**best_params_bnb)
y_pred_bnb = best_bnb.fit(x_train, y_train).predict(x_test)

# SVC
def objective_svc(trial):
    from sklearn.svm import SVC
    params = {
        'C': trial.suggest_loguniform('C', 1e-3, 1e3),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'degree': trial.suggest_int('degree', 2, 5) if trial.params['kernel'] == 'poly' else 1,
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.params['kernel'] in ['rbf', 'poly', 'sigmoid'] else 'scale',
    }
    clf = SVC(**params, random_state=42, probability=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_svc = create_study(objective_svc)
from sklearn.svm import SVC
best_svc = SVC(**best_params_svc, probability=True)
y_pred_svc = best_svc.fit(x_train, y_train).predict(x_test)

# LogisticRegression
def objective_lr(trial):
    from sklearn.linear_model import LogisticRegression
    params = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e5),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
    }
    clf = LogisticRegression(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_lr = create_study(objective_lr)
from sklearn.linear_model import LogisticRegression
best_lr = LogisticRegression(**best_params_lr)
y_pred_lr = best_lr.fit(x_train, y_train).predict(x_test)


# SGDClassifier
def objective_sgd(trial):
    from sklearn.linear_model import SGDClassifier
    params = {
        'loss': trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber']),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
        'alpha': trial.suggest_loguniform('alpha', 1e-6, 1e-1),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
        'eta0': trial.suggest_loguniform('eta0', 1e-5, 1e-1),
    }
    clf = SGDClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    return f1

best_params_sgd = create_study(objective_sgd)
from sklearn.linear_model import SGDClassifier
best_sgd = SGDClassifier(**best_params_sgd, random_state=42)
y_pred_sgd = best_sgd.fit(x_train, y_train).predict(x_test)

# Voting Classifier
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('gb', best_gb), ('rf', best_rf), ('ab', best_ab), ('etc', best_etc), ('dt', best_dt), ('knn', best_knn), ('gnb', best_gnb), ('bnb', best_bnb), ('svc', best_svc), ('lr', best_lr), ('sgd', best_sgd)], voting='hard')
voting_clf.fit(x_train, y_train)
y_pred_vh = voting_clf.predict(x_test)

# ------------------- Train LSTM separately for Model Comparison -------------------
# Train an instance of LSTMClassifier to obtain predictions for the comparison block.
lstm_model = LSTMClassifier(
    input_size=x_train.shape[1],
    sequence_length=5,  # Adjust if you used a different value elsewhere
    epochs=50, 
    hidden_size=64, 
    num_layers=2, 
    dropout=0.3,
    patience=5,
    learning_rate=0.001, 
    verbose=False
)

lstm_model.fit(x_train, y_train)
y_pred_lstm = lstm_model.predict(x_test)

# ----------------------- Model Comparison -----------------------
st.header("Model Comparison")

from sklearn.metrics import confusion_matrix
def calculate_evaluation_metrics(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def plot_confusion_matrix(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, linecolor="black", cbar=False, 
                xticklabels=["Non-Failure", "Failure"], yticklabels=["Non-Failure", "Failure"], ax=ax)
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
    ("LSTM", y_pred_lstm)
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
elif best_model == "LSTM":
    best_final_model = lstm_model
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
            # For models expecting probabilities, call predict_proba.
            # Note: For some classifiers, this may require using a scaler.
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
from sklearn.metrics import roc_curve, auc
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

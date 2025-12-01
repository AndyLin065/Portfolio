from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset: Credit Card Fraud Detection
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Alternative: Use sklearn's make_classification for demo

# Create synthetic fraud dataset (for demonstration)
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95, 0.05],  # Imbalanced dataset
    random_state=42
)

X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
y = pd.Series(y, name='is_fraud')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',  # Handle imbalanced data
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:\n", feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'], 
         feature_importance.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.show()


import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

# Exploratory Data Analysis (new cell)

sns.set(style="whitegrid")

# Basic summaries
print("Shapes:")
print(f"X: {X.shape}, X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y: {y.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}\n")

print("Class distribution (full dataset):")
print(y.value_counts().to_frame("count"))
print("\nClass distribution (train):")
print(y_train.value_counts().to_frame("count"))
print("\nMissing values (per column, full dataset):")
print(X.isnull().sum())

print("\nNumeric summary (top 10 features by importance):")
top10 = feature_importance['feature'].tolist()[:10]
print(X[top10].describe().T)

# Create joined dataframes for class-based plots
df = X.join(y)
df_train = X_train.join(y_train)

# 1) Feature importance (already computed) — plot top 10
plt.figure(figsize=(8, 5))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# 2) Distribution plots for top features
n = min(6, len(top10))
fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
for i, feat in enumerate(top10[:n]):
    ax_hist = axes[i, 0]
    ax_box = axes[i, 1]
    sns.histplot(df[feat], kde=True, ax=ax_hist, bins=40)
    ax_hist.set_title(f"Distribution: {feat}")
    sns.boxplot(x=y, y=X[feat], ax=ax_box)
    ax_box.set_title(f"Boxplot by class: {feat}")
plt.tight_layout()
plt.show()

# 3) Correlation matrix for top 10 features
corr = X[top10].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation (Top 10 Important Features)")
plt.tight_layout()
plt.show()

# 4) Pairplot for top 4 features (sampled to keep it fast)
pair_feats = top10[:4]
sample = df.sample(min(1000, len(df)), random_state=42)
sns.pairplot(sample, vars=pair_feats, hue='is_fraud', plot_kws={'alpha': 0.6, 's': 20})
plt.suptitle("Pairplot (sampled)", y=1.02)
plt.show()

# 5) Model evaluation visuals: ROC and Precision-Recall (on X_test / y_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax[0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
ax[0].set_xlabel("False Positive Rate")
ax[0].set_ylabel("True Positive Rate")
ax[0].set_title("ROC Curve")
ax[0].legend()

ax[1].step(recall, precision, where='post', label=f"AP = {avg_precision:.3f}")
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-Recall Curve")
ax[1].legend()

plt.tight_layout()
plt.show()

# 6) Confusion matrix on test set
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test)")
plt.tight_layout()
plt.show()

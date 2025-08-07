


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from data_loader import load_datasets
from feature_engineer import engineer_features
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw data
data = load_datasets()
results = data['results']
drivers = data['drivers']
constructors = data['constructors']
races = data['races']

# Merge the data into one DataFrame
results = results.merge(drivers, on='driverId')\
                 .merge(constructors, on='constructorId')\
                 .merge(races, on='raceId')

# Generate features
features_df = engineer_features(results)
features_df = features_df.dropna(subset=['positionOrder', 'grid', 'driver_form', 'track_history_avg',
                                          'constructor_avg_points', 'finish_rate'])

# Define X and y
X = features_df[['grid', 'driver_form', 'track_history_avg', 'constructor_avg_points', 'finish_rate']]
y = (features_df['positionOrder'] == 1).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

## The Model did not accurately predict actual wimmer which suggests class imbalance so will run agian with a balanced class
# Train model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
model_balanced = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
y_pred1 = model_balanced.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

# Confusion Matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Winner', 'Winner'], yticklabels=['Not Winner', 'Winner'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Feature Importances
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

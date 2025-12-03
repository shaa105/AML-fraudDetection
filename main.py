import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("/content/paysim dataset.csv")

# Drop ID columns for now (keep separately for graph)
account_cols = ["nameOrig", "nameDest"]

# Feature engineering
df['orig_balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
df['amount_log'] = np.log1p(df['amount'])
df['hour'] = df['step'] % 24
df['day'] = (df['step'] // 24).astype(int)

# Replace infinities and missing
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Target and features
y = df['isFraud'].astype(int)
features = ["amount","amount_log","orig_balance_change","dest_balance_change","hour","day","type"]
X = df[features]

def rule_structuring(row):
    # Small repeated deposits (less than 5000)
    return (row['amount'] < 5000, 5 if row['amount'] < 5000 else 0)

def rule_layering(row):
    # Money moved through multiple accounts (simulate)
    # We'll use transactions > 10000 as layering proxy
    return (row['amount'] > 10000, 8 if row['amount'] > 10000 else 0)

def rule_velocity(row):
    # Rapid transactions (simulate by step modulo)
    return (row['step'] % 10 == 0, 4 if row['step'] % 10 == 0 else 0)

def rule_large_amount(row):
    # Very large transaction
    return (row['amount'] > 200000, 10 if row['amount'] > 200000 else 0)

def apply_aml_rules(row):
    rules = [rule_structuring, rule_layering, rule_velocity, rule_large_amount]
    flags = {}
    total_score = 0
    for r in rules:
        flag, score = r(row)
        flags[r.__name__] = flag
        total_score += score
    return flags, total_score

# Apply rules to all transactions
rule_results = df.apply(lambda x: apply_aml_rules(x), axis=1)
df['rule_flags'] = [r[0] for r in rule_results]
df['rule_score'] = [r[1] for r in rule_results]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing
numeric_features = ["amount","amount_log","orig_balance_change","dest_balance_change","hour","day"]
categorical_features = ["type"]

numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ML pipeline
rf_pipeline = Pipeline([
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
])

# Train
rf_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = rf_pipeline.predict(X_test)
y_prob = rf_pipeline.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Save ML probabilities
df['ml_prob'] = rf_pipeline.predict_proba(X)[:,1]
def compute_risk_score(ml_prob, rule_score):
    ml_component = ml_prob * 100 * 0.6
    rule_component = (rule_score / 27) * 100 * 0.4  # max total rule score ~27
    return ml_component + rule_component

def classify_risk(score):
    if score >= 75:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    else:
        return "LOW"

df['risk_score'] = df.apply(lambda x: compute_risk_score(x['ml_prob'], x['rule_score']), axis=1)
df['risk_level'] = df['risk_score'].apply(classify_risk)
import networkx as nx

G = nx.DiGraph()

# Add transactions as edges
for idx, row in df.iterrows():
    G.add_edge(row['nameOrig'], row['nameDest'], weight=row['amount'], risk=row['risk_score'])

# Find high-risk hubs
centrality = nx.degree_centrality(G)
top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top suspicious accounts (hubs):", top_hubs)
import seaborn as sns
import matplotlib.pyplot as plt

# Top 25 risky accounts
top_accounts = df.groupby('nameOrig')['risk_score'].max().sort_values(ascending=False).head(25)

plt.figure(figsize=(10,8))
sns.heatmap(top_accounts.to_frame().T, annot=True, fmt=".1f", cmap="Oranges")
plt.title("Top Risky Accounts (Hybrid AML Score)")
plt.show()

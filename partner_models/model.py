import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import recall_score, precision_score, f1_score

# 1. Clean & Streamlined Tabular TLE Parser
def parse_tle_to_features(file_path):
    features = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    for i in range(0, len(lines) - 2, 3):
        try:
            line2 = lines[i+2]
            features.append({
                'inclination': float(line2[8:16]),
                'eccentricity': float("0." + line2[26:33]),
                'mean_motion': float(line2[52:63])
            })
        except Exception:
            continue
    return pd.DataFrame(features)

print("--- Step 1: Parsing TLE Datasets ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
starlink_file = next((f for f in os.listdir(script_dir) if "STARLINK" in f and "1008" in f), "STARLINK-1008 TLE.txt")
debris_file = next((f for f in os.listdir(script_dir) if "COSMOS" in f), "COSMOS 2551 Debris.txt")

df_starlink = parse_tle_to_features(os.path.join(script_dir, starlink_file))
df_debris = parse_tle_to_features(os.path.join(script_dir, debris_file))

# 2. Expand Pairwise Interactions for Training Stability
print("\n--- Step 2: Structuring Pairwise Space Interactions ---")
# Using a larger sample size gives the models enough statistical variation to learn boundaries
sample_size = min(3000, len(df_starlink), len(df_debris))
df_starlink_sample = df_starlink.sample(n=sample_size, random_state=42).reset_index(drop=True).add_prefix('starlink_')
df_debris_sample = df_debris.sample(n=sample_size, random_state=42).reset_index(drop=True).add_prefix('debris_')
df_pairs = pd.concat([df_starlink_sample, df_debris_sample], axis=1)

# 3. Mathematically Define High-Risk Conjunction Target
np.random.seed(42)
orbital_delta = (
    abs(df_pairs['starlink_inclination'] - df_pairs['debris_inclination']) * 10 +
    abs(df_pairs['starlink_eccentricity'] - df_pairs['debris_eccentricity']) * 500
)
# Create a tight distribution where roughly 10-15% of cases are close conjunction encounters
threshold = np.percentile(orbital_delta, 12.0)
df_pairs['is_high_risk'] = (orbital_delta <= threshold).astype(int)

print(f"-> Total Dataset Pairs: {len(df_pairs)}")
print(f"-> Low-Risk Cases (Class 0): {list(df_pairs['is_high_risk']).count(0)}")
print(f"-> High-Risk Conjunctions (Class 1): {list(df_pairs['is_high_risk']).count(1)}")

# 4. Data Preparation & Split
X = df_pairs.drop(columns=['is_high_risk'])
y = df_pairs['is_high_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize Models with Class Balancing to resolve skewness
lr_model = LogisticRegression(class_weight="balanced", random_state=42)
svc_model = SVC(kernel='rbf', probability=True, class_weight="balanced", random_state=42)
hybrid_ensemble = VotingClassifier(estimators=[('lr', lr_model), ('svc', svc_model)], voting='soft')

models = {
    "Logistic Regression (Baseline)": lr_model,
    "Support Vector Machine (SVC)": svc_model,
    "Hybrid Ensemble Model (LR + SVC)": hybrid_ensemble
}

# --- STEP 4: Data Split ---
X = df_pairs.drop(columns=['is_high_risk'])
y = df_pairs['is_high_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- STEP 5: Standardization / Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =================================================================================
# --- PASTE THE SHOWCASE CODE BLOCK HERE (Right after Step 5) ---
# =================================================================================
print("\n======================= SUBMISSION DEMO: SPLIT DATA MATRICES =======================")
print(f"Total Rows split for Training (80%): {X_train_scaled.shape[0]} samples")
print(f"Total Rows split for Testing (20%):  {X_test_scaled.shape[0]} samples")

print("\n1. TRAINING FEATURE MATRIX (X_train_scaled) - First 3 Rows:")
print(pd.DataFrame(X_train_scaled, columns=X.columns).head(3).to_string(index=False))

print("\n2. TRAINING GROUND-TRUTH LABELS (y_train) - First 15 Values:")
print(list(y_train[:15]))

print("\n3. TESTING FEATURE MATRIX (X_test_scaled) - First 3 Rows:")
print(pd.DataFrame(X_test_scaled, columns=X.columns).head(3).to_string(index=False))

print("\n4. TESTING HIDDEN TARGET LABELS (y_test) - First 15 Values:")
print(list(y_test[:15]))
print("====================================================================================\n")


# 7. Execution Loop
results = []
print("\n--- Step 3: Running Comparative ML Suite ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    recall = recall_score(y_test, y_pred, zero_division=0) * 100
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    
    results.append({
        "Method": name, 
        "Recall (Safety %)": f"{recall:.2f}%", 
        "Precision": f"{precision:.2f}%", 
        "F1-Score": f"{f1:.2f}%"
    })

# 8. Output Final Report Table
print("\n==================== FINAL PROJECT COMPARISON TABLE ====================")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print("========================================================================\n")

df_results.to_csv(os.path.join(script_dir, "model_comparison_results.csv"), index=False)
# --- SHOW SIMPLIFIED DATA TO TEACHER ---
print("\n==================== TECHNICAL DATA SIMPLIFICATION (df.head) ====================")
print("This shows how raw text TLE lines were converted into clean numeric rows and columns:")
print(df_pairs.head())
print("=================================================================================\n")



##XGBoost 

# 6. Initialize Core Models, XGBoost, and the Hybrid Voting Ensemble
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

lr_model = LogisticRegression(class_weight="balanced", random_state=42)
svc_model = SVC(kernel='rbf', probability=True, class_weight="balanced", random_state=42)
xgb_model = XGBClassifier(scale_pos_weight=7.0, eval_metric='logloss', random_state=42)

hybrid_ensemble = VotingClassifier(
    estimators=[('lr', lr_model), ('svc', svc_model), ('xgb', xgb_model)], 
    voting='soft'
)

models = {
    "Logistic Regression (Baseline)": lr_model,
    "Support Vector Machine (SVC)": svc_model,
    "XGBoost (Advanced Boosting)": xgb_model,
    "Hybrid Ensemble Model (Voting)": hybrid_ensemble
}

# 7. Model Execution Loop
results = []
plot_data = []

print("\n--- Step 3: Running Comparative ML Suite ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    recall = recall_score(y_test, y_pred, zero_division=0) * 100
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    
    results.append({
        "Method": name, "Recall (Safety %)": f"{recall:.2f}%", 
        "Precision": f"{precision:.2f}%", "F1-Score": f"{f1:.2f}%"
    })
    
    plot_data.append([name, "Recall", recall])
    plot_data.append([name, "Precision", precision])
    plot_data.append([name, "F1-Score", f1])

# 8. Display Final Report Table
print("\n==================== FINAL PROJECT COMPARISON TABLE ====================")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print("========================================================================\n")

# 9. GENERATE METRIC COMPARISON GRAPH
print("Generating Performance Evaluation Plot...")
df_plot = pd.DataFrame(plot_data, columns=["Model", "Metric", "Percentage Score (%)"])

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

ax = sns.barplot(x="Model", y="Percentage Score (%)", hue="Metric", data=df_plot, palette="Set2")

plt.title("Satellite Collision Avoidance: Model Performance Comparison", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Machine Learning Approach", fontsize=12, labelpad=10)
plt.ylabel("Score (%)", fontsize=12)
plt.ylim(0, 115)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', xytext=(0, 3),
                    textcoords='offset points')

plt.tight_layout()
print("Opening graph window... Close the popup screen to exit program.")
plt.show()

# ==============================================================================
# TEMA: REGRESIA LINIARĂ ȘI CLASIFICARE
# Cerințe: 1. Predicția bacșișului (tip) - problema 7.1 Tips
#          2. Clasificare rock vs classical - problema 7.12 Music
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setare stil pentru grafice
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "="*80)
print(" TEMA: IMPLEMENTARE MODEL LINIAR - REGRESIA LINIARĂ ȘI CLASIFICARE")
print(" Cook, Swayne (2007) - Problema 7.1 Tips & Problema 7.12 Music")
print("="*80 + "\n")

# ==============================================================================
# PROBLEMA 7.1: PREDICȚIA BACȘIȘULUI (TIP) PENTRU O MASĂ LA RESTAURANT
# Dataset: tips.csv (Cook, Swayne, 2007, pp 153)
# Model: Linear Regression
# ==============================================================================

print("\n" + "█"*80)
print("█ PROBLEMA 7.1: PREDICȚIA BACȘIȘULUI (TIP) - pp 153 (Cook, Swayne, 2007)")
print("█"*80 + "\n")

# Încărcare dataset
print("📥 Se încarcă dataset-ul tips.csv...\n")
tips_data = pd.read_csv('tips.csv')

print("📊 Informații despre dataset:")
print(tips_data.info())

print("\n📋 Primele 10 înregistrări:")
print(tips_data.head(10))

print(f"\n📈 Dimensiuni dataset: {tips_data.shape[0]} observații, {tips_data.shape[1]} variabile")

print("\n📉 Statistici descriptive pentru variabilele cheie:")
print(tips_data[['total_bill', 'tip']].describe())

# Verificare valori lipsă
print(f"\n🔍 Valori lipsă: {tips_data.isnull().sum().sum()}")

# Analiza corelației
correlation = tips_data['total_bill'].corr(tips_data['tip'])
print(f"\n📊 Corelație Pearson între total_bill și tip: {correlation:.4f}")

# Preparare date pentru regresie liniară
X_tips = tips_data[['total_bill']].values  # Feature: suma totală a mesei
y_tips = tips_data['tip'].values            # Target: bacșișul

# Împărțire date: 80% antrenare, 20% test
X_train_tips, X_test_tips, y_train_tips, y_test_tips = train_test_split(
    X_tips, y_tips, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n🔀 Împărțire date:")
print(f"   - Antrenare: {len(X_train_tips)} observații ({len(X_train_tips)/len(X_tips)*100:.1f}%)")
print(f"   - Test: {len(X_test_tips)} observații ({len(X_test_tips)/len(X_tips)*100:.1f}%)")

# Antrenare model de regresie liniară
print("\n🤖 Se antrenează modelul de regresie liniară...")
model_tips = LinearRegression()
model_tips.fit(X_train_tips, y_train_tips)

# Predicții
y_pred_tips_train = model_tips.predict(X_train_tips)
y_pred_tips_test = model_tips.predict(X_test_tips)

# Evaluare model
r2_train = r2_score(y_train_tips, y_pred_tips_train)
r2_test = r2_score(y_test_tips, y_pred_tips_test)
rmse_test = np.sqrt(mean_squared_error(y_test_tips, y_pred_tips_test))
mae_test = mean_absolute_error(y_test_tips, y_pred_tips_test)

print("\n" + "─"*80)
print("REZULTATE MODEL REGRESIE LINIARĂ - PREDICȚIA BACȘIȘULUI")
print("─"*80)

print(f"\n📐 Ecuația modelului liniar:")
print(f"   tip = {model_tips.intercept_:.4f} + {model_tips.coef_[0]:.4f} × total_bill")

print(f"\n💡 Interpretare:")
print(f"   - Intercept (β₀): {model_tips.intercept_:.4f}")
print(f"     → Bacșișul estimat când total_bill = 0")
print(f"   - Slope (β₁): {model_tips.coef_[0]:.4f}")
print(f"     → Pentru fiecare dolar adițional la masa, bacșișul crește cu ${model_tips.coef_[0]:.4f}")
print(f"     → Rata de bacșiș: {model_tips.coef_[0]*100:.2f}%")

print(f"\n📊 Metrice de performanță:")
print(f"   ✓ R² (antrenare): {r2_train:.4f} ({r2_train*100:.2f}% variabilitate explicată)")
print(f"   ✓ R² (test):      {r2_test:.4f} ({r2_test*100:.2f}% variabilitate explicată)")
print(f"   ✓ RMSE (test):    ${rmse_test:.4f}")
print(f"   ✓ MAE (test):     ${mae_test:.4f}")

# Exemple de predicții
print(f"\n🔮 Exemple de predicții:")
example_bills = np.array([[10.0], [20.0], [30.0], [50.0]])
example_predictions = model_tips.predict(example_bills)
for bill, pred in zip(example_bills, example_predictions):
    print(f"   - Masa de ${bill[0]:.2f} → Bacșiș estimat: ${pred:.2f}")

print("─"*80)

# Vizualizare Problema 7.1
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Grafic 1: Scatter plot cu dreapta de regresie
axes[0, 0].scatter(X_train_tips, y_train_tips, color='#2E86AB', alpha=0.5, s=60, label='Date antrenare')
axes[0, 0].scatter(X_test_tips, y_test_tips, color='#F18F01', alpha=0.7, s=60, label='Date test')
# Linie de regresie
x_line = np.linspace(X_tips.min(), X_tips.max(), 100).reshape(-1, 1)
y_line = model_tips.predict(x_line)
axes[0, 0].plot(x_line, y_line, color='#A23B72', linewidth=3, label='Dreapta de regresie', zorder=5)
axes[0, 0].set_xlabel('Suma totală a mesei - total_bill ($)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Bacșișul - tip ($)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Problema 7.1: Predicția bacșișului (Cook, Swayne, 2007)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Grafic 2: Reziduuri vs Predicții
residuals_test = y_test_tips - y_pred_tips_test
axes[0, 1].scatter(y_pred_tips_test, residuals_test, color='#F18F01', alpha=0.7, s=60)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Linie zero')
axes[0, 1].set_xlabel('Predicții ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Reziduuri ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Analiza reziduurilor - Verificare asumții', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Grafic 3: Distribuția reziduurilor
axes[1, 0].hist(residuals_test, bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Reziduuri ($)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frecvență', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribuția reziduurilor (test normalitate)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Grafic 4: Valori reale vs Predicții
axes[1, 1].scatter(y_test_tips, y_pred_tips_test, color='#A23B72', alpha=0.7, s=60)
# Linie perfectă (y=x)
min_val = min(y_test_tips.min(), y_pred_tips_test.min())
max_val = max(y_test_tips.max(), y_pred_tips_test.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicție perfectă')
axes[1, 1].set_xlabel('Valori reale ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Predicții ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_title(f'Valori reale vs Predicții (R²={r2_test:.4f})', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('problema_7_1_tips_regresie.png', dpi=150, bbox_inches='tight')
print("\n✓ Grafic salvat: problema_7_1_tips_regresie.png\n")
plt.show()

# ==============================================================================
# PROBLEMA 7.12: DISCRIMINAREA ÎNTRE ROCK ȘI CLASSICAL
# Dataset: features_30_sec.csv (caracteristici audio)
# Model: Logistic Regression
# ==============================================================================

print("\n" + "█"*80)
print("█ PROBLEMA 7.12: DISCRIMINARE ROCK vs CLASSICAL - pp 171 (Cook, Swayne, 2007)")
print("█"*80 + "\n")

# Încărcare dataset muzică
print("🎵 Se încarcă dataset-ul features_30_sec.csv...\n")
music_data = pd.read_csv('features_30_sec.csv')

print("📊 Informații despre dataset:")
print(music_data.info())

print("\n📋 Primele 5 înregistrări:")
print(music_data.head())

# Verificare coloane disponibile
print(f"\n📋 Coloane disponibile: {list(music_data.columns)}")

# Filtrare doar genurile Rock și Classical
print("\n🎸 Se filtrează doar genurile 'rock' și 'classical'...")
music_filtered = music_data[music_data['label'].isin(['rock', 'classical'])].copy()

print(f"\n📊 Distribuția claselor:")
print(music_filtered['label'].value_counts())
print(f"   - Total piese: {len(music_filtered)}")

# Selectare features pentru clasificare
# Alegem caracteristici audio relevante
feature_cols = ['tempo', 'spectral_centroid_mean', 'spectral_bandwidth_mean',
                'rolloff_mean', 'zero_crossing_rate_mean']

# Verificăm dacă coloanele există, altfel folosim altele
available_features = [col for col in feature_cols if col in music_filtered.columns]

if len(available_features) < 3:
    # Fallback: folosim primele coloane numerice
    numeric_cols = music_filtered.select_dtypes(include=[np.number]).columns.tolist()
    available_features = [col for col in numeric_cols if col not in ['label', 'filename']][:5]

print(f"\n✓ Features selectate pentru clasificare: {available_features}")

# Preparare date
X_music = music_filtered[available_features].values
le = LabelEncoder()
y_music = le.fit_transform(music_filtered['label'])  # rock=0, classical=1

print(f"\n📊 Statistici features:")
print(music_filtered[available_features].describe())

# Împărțire date
X_train_music, X_test_music, y_train_music, y_test_music = train_test_split(
    X_music, y_music, test_size=0.2, random_state=42, stratify=y_music
)

print(f"\n🔀 Împărțire date:")
print(f"   - Antrenare: {len(X_train_music)} observații")
print(f"   - Test: {len(X_test_music)} observații")

# Normalizare features (important pentru regresie logistică)
scaler = StandardScaler()
X_train_music_scaled = scaler.fit_transform(X_train_music)
X_test_music_scaled = scaler.transform(X_test_music)

# Antrenare model de clasificare (Regresie Logistică)
print("\n🤖 Se antrenează modelul de clasificare (Regresie Logistică)...")
model_music = LogisticRegression(random_state=42, max_iter=2000, solver='lbfgs')
model_music.fit(X_train_music_scaled, y_train_music)

# Predicții
y_pred_music_train = model_music.predict(X_train_music_scaled)
y_pred_music_test = model_music.predict(X_test_music_scaled)
y_pred_proba = model_music.predict_proba(X_test_music_scaled)

# Evaluare model
accuracy_train = accuracy_score(y_train_music, y_pred_music_train)
accuracy_test = accuracy_score(y_test_music, y_pred_music_test)
cm = confusion_matrix(y_test_music, y_pred_music_test)

print("\n" + "─"*80)
print("REZULTATE MODEL CLASIFICARE - DISCRIMINARE ROCK vs CLASSICAL")
print("─"*80)

print(f"\n📐 Parametrii modelului de regresie logistică:")
print(f"   - Intercept (β₀): {model_music.intercept_[0]:.4f}")
print(f"\n   - Coeficienți (βᵢ):")
for i, (feat, coef) in enumerate(zip(available_features, model_music.coef_[0])):
    print(f"     {feat}: {coef:.4f}")

print(f"\n🎯 Performanță model:")
print(f"   ✓ Accuracy (antrenare): {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
print(f"   ✓ Accuracy (test):      {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")

print(f"\n📊 Matrice de confuzie (date test):")
print(f"   ╔═════════════════╦═══════════════════╗")
print(f"   ║                 ║    Predicții      ║")
print(f"   ╠═════════════════╬═════════╦═════════╣")
print(f"   ║                 ║  Rock   ║Classical║")
print(f"   ╠═════════════════╬═════════╬═════════╣")
print(f"   ║ Real Rock       ║  {cm[0,0]:4d}   ║  {cm[0,1]:4d}   ║")
print(f"   ║ Real Classical  ║  {cm[1,0]:4d}   ║  {cm[1,1]:4d}   ║")
print(f"   ╚═════════════════╩═════════╩═════════╝")

print(f"\n   Interpretare:")
print(f"   - True Negatives (TN):  {cm[0,0]} - Rock clasificat corect ca Rock")
print(f"   - False Positives (FP): {cm[0,1]} - Rock clasificat greșit ca Classical")
print(f"   - False Negatives (FN): {cm[1,0]} - Classical clasificat greșit ca Rock")
print(f"   - True Positives (TP):  {cm[1,1]} - Classical clasificat corect ca Classical")

print(f"\n📈 Raport de clasificare detaliat:")
print(classification_report(y_test_music, y_pred_music_test,
                           target_names=['Rock', 'Classical'],
                           digits=4))

print("─"*80)

# Vizualizare Problema 7.12
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Grafic 1: Matrice de confuzie
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0, 0],
           xticklabels=['Rock', 'Classical'], yticklabels=['Rock', 'Classical'],
           cbar_kws={'label': 'Număr observații'}, linewidths=3, linecolor='black',
           annot_kws={'size': 16, 'weight': 'bold'})
axes[0, 0].set_title('Problema 7.12: Matrice de confuzie - Rock vs Classical',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Adevărată etichetă', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Predicția modelului', fontsize=11, fontweight='bold')

# Grafic 2: Distribuția probabilităților
prob_rock = y_pred_proba[y_test_music == 0][:, 0]
prob_classical = y_pred_proba[y_test_music == 1][:, 1]

axes[0, 1].hist(prob_rock, alpha=0.6, bins=15, label='Rock (prob. Rock)',
               color='#2E86AB', edgecolor='black', linewidth=1.5)
axes[0, 1].hist(prob_classical, alpha=0.6, bins=15, label='Classical (prob. Classical)',
               color='#A23B72', edgecolor='black', linewidth=1.5)
axes[0, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
axes[0, 1].set_xlabel('Probabilitate de clasificare', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frecvență', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribuția probabilităților de clasificare', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Grafic 3: Feature importance (coeficienți absoluti)
feature_importance = np.abs(model_music.coef_[0])
sorted_idx = np.argsort(feature_importance)
axes[1, 0].barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='#F18F01', edgecolor='black')
axes[1, 0].set_yticks(range(len(sorted_idx)))
axes[1, 0].set_yticklabels([available_features[i] for i in sorted_idx], fontsize=9)
axes[1, 0].set_xlabel('Importanță (|coeficient|)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Importanța caracteristicilor audio în clasificare', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Grafic 4: Scatter plot 2 features principale
if len(available_features) >= 2:
    idx1, idx2 = 0, 1
    axes[1, 1].scatter(X_test_music[y_test_music == 0, idx1],
                      X_test_music[y_test_music == 0, idx2],
                      color='#2E86AB', alpha=0.6, s=80, label='Rock', edgecolor='black')
    axes[1, 1].scatter(X_test_music[y_test_music == 1, idx1],
                      X_test_music[y_test_music == 1, idx2],
                      color='#A23B72', alpha=0.6, s=80, label='Classical', edgecolor='black')
    axes[1, 1].set_xlabel(available_features[idx1], fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel(available_features[idx2], fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Separare claselor în spațiul features', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('problema_7_12_music_clasificare.png', dpi=150, bbox_inches='tight')
print("\n✓ Grafic salvat: problema_7_12_music_clasificare.png\n")
plt.show()

# ==============================================================================
# SUMAR FINAL
# ==============================================================================

print("\n" + "="*80)
print(" SUMAR FINAL - REZULTATE TEMA")
print("="*80)

print(f"\n📊 PROBLEMA 7.1 - PREDICȚIA BACȘIȘULUI:")
print(f"   ✓ Model: Regresie Liniară")
print(f"   ✓ Dataset: tips.csv ({len(tips_data)} observații)")
print(f"   ✓ Ecuație: tip = {model_tips.intercept_:.4f} + {model_tips.coef_[0]:.4f} × total_bill")
print(f"   ✓ R² Score: {r2_test:.4f} ({r2_test*100:.2f}%)")
print(f"   ✓ RMSE: ${rmse_test:.4f}")

print(f"\n🎵 PROBLEMA 7.12 - DISCRIMINARE ROCK vs CLASSICAL:")
print(f"   ✓ Model: Regresie Logistică")
print(f"   ✓ Dataset: features_30_sec.csv ({len(music_filtered)} piese)")
print(f"   ✓ Features: {len(available_features)} caracteristici audio")
print(f"   ✓ Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print(f"   ✓ Clasificări corecte: {cm[0,0] + cm[1,1]}/{len(y_test_music)}")

print("\n" + "="*80)
print("✅ ANALIZA COMPLETĂ! Ambele probleme au fost rezolvate cu succes!")
print("="*80 + "\n")


# ==============================================================================
# Descriere: ~ralizare Tema 2
# - pentru o reprezentare cât mai reală am ales seturi de date reale pe care să le folosesc la realizarea cerințelor
# - am folosit seturi de date de pe site-ul: https://www.kaggle.com/
# -tips: https://www.kaggle.com/datasets/hnazari8665/tipscsv
# -audio: https://www.kaggle.com/code/andradaolteanu/work-w-audio-data-visualise-classify-recommend/notebook
# ==============================================================================

# ==============================================================================
# Rulaj:
# C:\Users\Daniel\PycharmProjects\.venv\Scripts\python.exe "C:\Users\Daniel\PycharmProjects\2 TI\main.py"
#
# ================================================================================
#  TEMA: IMPLEMENTARE MODEL LINIAR - REGRESIA LINIARĂ ȘI CLASIFICARE
#  Cook, Swayne (2007) - Problema 7.1 Tips & Problema 7.12 Music
# ================================================================================
#
#
# ████████████████████████████████████████████████████████████████████████████████
# █ PROBLEMA 7.1: PREDICȚIA BACȘIȘULUI (TIP) - pp 153 (Cook, Swayne, 2007)
# ████████████████████████████████████████████████████████████████████████████████
#
# 📥 Se încarcă dataset-ul tips.csv...
#
# 📊 Informații despre dataset:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 244 entries, 0 to 243
# Data columns (total 11 columns):
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   total_bill        244 non-null    float64
#  1   tip               244 non-null    float64
#  2   sex               244 non-null    object
#  3   smoker            244 non-null    object
#  4   day               244 non-null    object
#  5   time              244 non-null    object
#  6   size              244 non-null    int64
#  7   price_per_person  244 non-null    float64
#  8   Payer Name        244 non-null    object
#  9   CC Number         244 non-null    int64
#  10  Payment ID        244 non-null    object
# dtypes: float64(3), int64(2), object(6)
# memory usage: 21.1+ KB
# None
#
# 📋 Primele 10 înregistrări:
#    total_bill   tip     sex  ...          Payer Name         CC Number Payment ID
# 0       16.99  1.01  Female  ...  Christy Cunningham  3560325168603410    Sun2959
# 1       10.34  1.66    Male  ...      Douglas Tucker  4478071379779230    Sun4608
# 2       21.01  3.50    Male  ...      Travis Walters  6011812112971322    Sun4458
# 3       23.68  3.31    Male  ...    Nathaniel Harris  4676137647685994    Sun5260
# 4       24.59  3.61  Female  ...        Tonya Carter  4832732618637221    Sun2251
# 5       25.29  4.71    Male  ...          Erik Smith   213140353657882    Sun9679
# 6        8.77  2.00    Male  ...  Kristopher Johnson  2223727524230344    Sun5985
# 7       26.88  3.12    Male  ...         Robert Buck  3514785077705092    Sun8157
# 8       15.04  1.96    Male  ...     Joseph Mcdonald  3522866365840377    Sun6820
# 9       14.78  3.23    Male  ...       Jerome Abbott  3532124519049786    Sun3775
#
# [10 rows x 11 columns]
#
# 📈 Dimensiuni dataset: 244 observații, 11 variabile
#
# 📉 Statistici descriptive pentru variabilele cheie:
#        total_bill         tip
# count  244.000000  244.000000
# mean    19.785943    2.998279
# std      8.902412    1.383638
# min      3.070000    1.000000
# 25%     13.347500    2.000000
# 50%     17.795000    2.900000
# 75%     24.127500    3.562500
# max     50.810000   10.000000
#
# 🔍 Valori lipsă: 0
#
# 📊 Corelație Pearson între total_bill și tip: 0.6757
#
# 🔀 Împărțire date:
#    - Antrenare: 195 observații (79.9%)
#    - Test: 49 observații (20.1%)
#
# 🤖 Se antrenează modelul de regresie liniară...
#
# ────────────────────────────────────────────────────────────────────────────────
# REZULTATE MODEL REGRESIE LINIARĂ - PREDICȚIA BACȘIȘULUI
# ────────────────────────────────────────────────────────────────────────────────
#
# 📐 Ecuația modelului liniar:
#    tip = 0.9252 + 0.1070 × total_bill
#
# 💡 Interpretare:
#    - Intercept (β₀): 0.9252
#      → Bacșișul estimat când total_bill = 0
#    - Slope (β₁): 0.1070
#      → Pentru fiecare dolar adițional la masa, bacșișul crește cu $0.1070
#      → Rata de bacșiș: 10.70%
#
# 📊 Metrice de performanță:
#    ✓ R² (antrenare): 0.4310 (43.10% variabilitate explicată)
#    ✓ R² (test):      0.5449 (54.49% variabilitate explicată)
#    ✓ RMSE (test):    $0.7542
#    ✓ MAE (test):     $0.6209
#
# 🔮 Exemple de predicții:
#    - Masa de $10.00 → Bacșiș estimat: $1.99
#    - Masa de $20.00 → Bacșiș estimat: $3.06
#    - Masa de $30.00 → Bacșiș estimat: $4.13
#    - Masa de $50.00 → Bacșiș estimat: $6.27
# ────────────────────────────────────────────────────────────────────────────────
#
# ✓ Grafic salvat: problema_7_1_tips_regresie.png
#
#
# ████████████████████████████████████████████████████████████████████████████████
# █ PROBLEMA 7.12: DISCRIMINARE ROCK vs CLASSICAL - pp 171 (Cook, Swayne, 2007)
# ████████████████████████████████████████████████████████████████████████████████
#
# 🎵 Se încarcă dataset-ul features_30_sec.csv...
#
# 📊 Informații despre dataset:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 60 columns):
#  #   Column                   Non-Null Count  Dtype
# ---  ------                   --------------  -----
#  0   filename                 1000 non-null   object
#  1   length                   1000 non-null   int64
#  2   chroma_stft_mean         1000 non-null   float64
#  3   chroma_stft_var          1000 non-null   float64
#  4   rms_mean                 1000 non-null   float64
#  5   rms_var                  1000 non-null   float64
#  6   spectral_centroid_mean   1000 non-null   float64
#  7   spectral_centroid_var    1000 non-null   float64
#  8   spectral_bandwidth_mean  1000 non-null   float64
#  9   spectral_bandwidth_var   1000 non-null   float64
#  10  rolloff_mean             1000 non-null   float64
#  11  rolloff_var              1000 non-null   float64
#  12  zero_crossing_rate_mean  1000 non-null   float64
#  13  zero_crossing_rate_var   1000 non-null   float64
#  14  harmony_mean             1000 non-null   float64
#  15  harmony_var              1000 non-null   float64
#  16  perceptr_mean            1000 non-null   float64
#  17  perceptr_var             1000 non-null   float64
#  18  tempo                    1000 non-null   float64
#  19  mfcc1_mean               1000 non-null   float64
#  20  mfcc1_var                1000 non-null   float64
#  21  mfcc2_mean               1000 non-null   float64
#  22  mfcc2_var                1000 non-null   float64
#  23  mfcc3_mean               1000 non-null   float64
#  24  mfcc3_var                1000 non-null   float64
#  25  mfcc4_mean               1000 non-null   float64
#  26  mfcc4_var                1000 non-null   float64
#  27  mfcc5_mean               1000 non-null   float64
#  28  mfcc5_var                1000 non-null   float64
#  29  mfcc6_mean               1000 non-null   float64
#  30  mfcc6_var                1000 non-null   float64
#  31  mfcc7_mean               1000 non-null   float64
#  32  mfcc7_var                1000 non-null   float64
#  33  mfcc8_mean               1000 non-null   float64
#  34  mfcc8_var                1000 non-null   float64
#  35  mfcc9_mean               1000 non-null   float64
#  36  mfcc9_var                1000 non-null   float64
#  37  mfcc10_mean              1000 non-null   float64
#  38  mfcc10_var               1000 non-null   float64
#  39  mfcc11_mean              1000 non-null   float64
#  40  mfcc11_var               1000 non-null   float64
#  41  mfcc12_mean              1000 non-null   float64
#  42  mfcc12_var               1000 non-null   float64
#  43  mfcc13_mean              1000 non-null   float64
#  44  mfcc13_var               1000 non-null   float64
#  45  mfcc14_mean              1000 non-null   float64
#  46  mfcc14_var               1000 non-null   float64
#  47  mfcc15_mean              1000 non-null   float64
#  48  mfcc15_var               1000 non-null   float64
#  49  mfcc16_mean              1000 non-null   float64
#  50  mfcc16_var               1000 non-null   float64
#  51  mfcc17_mean              1000 non-null   float64
#  52  mfcc17_var               1000 non-null   float64
#  53  mfcc18_mean              1000 non-null   float64
#  54  mfcc18_var               1000 non-null   float64
#  55  mfcc19_mean              1000 non-null   float64
#  56  mfcc19_var               1000 non-null   float64
#  57  mfcc20_mean              1000 non-null   float64
#  58  mfcc20_var               1000 non-null   float64
#  59  label                    1000 non-null   object
# dtypes: float64(57), int64(1), object(2)
# memory usage: 468.9+ KB
# None
#
# 📋 Primele 5 înregistrări:
#           filename  length  chroma_stft_mean  ...  mfcc20_mean  mfcc20_var  label
# 0  blues.00000.wav  661794          0.350088  ...     1.221291   46.936035  blues
# 1  blues.00001.wav  661794          0.340914  ...     0.531217   45.786282  blues
# 2  blues.00002.wav  661794          0.363637  ...    -2.231258   30.573025  blues
# 3  blues.00003.wav  661794          0.404785  ...    -3.407448   31.949339  blues
# 4  blues.00004.wav  661794          0.308526  ...   -11.703234   55.195160  blues
#
# [5 rows x 60 columns]
#
# 📋 Coloane disponibile: ['filename', 'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo', 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var', 'label']
#
# 🎸 Se filtrează doar genurile 'rock' și 'classical'...
#
# 📊 Distribuția claselor:
# label
# classical    100
# rock         100
# Name: count, dtype: int64
#    - Total piese: 200
#
# ✓ Features selectate pentru clasificare: ['tempo', 'spectral_centroid_mean', 'spectral_bandwidth_mean', 'rolloff_mean', 'zero_crossing_rate_mean']
#
# 📊 Statistici features:
#             tempo  ...  zero_crossing_rate_mean
# count  200.000000  ...               200.000000
# mean   124.091696  ...                 0.093426
# std     29.943959  ...                 0.034610
# min     67.999589  ...                 0.031534
# 25%    103.359375  ...                 0.068789
# 50%    120.250355  ...                 0.087253
# 75%    143.554688  ...                 0.118678
# max    234.907670  ...                 0.187548
#
# [8 rows x 5 columns]
#
# 🔀 Împărțire date:
#    - Antrenare: 160 observații
#    - Test: 40 observații
#
# 🤖 Se antrenează modelul de clasificare (Regresie Logistică)...
#
# ────────────────────────────────────────────────────────────────────────────────
# REZULTATE MODEL CLASIFICARE - DISCRIMINARE ROCK vs CLASSICAL
# ────────────────────────────────────────────────────────────────────────────────
#
# 📐 Parametrii modelului de regresie logistică:
#    - Intercept (β₀): 0.4222
#
#    - Coeficienți (βᵢ):
#      tempo: -0.1454
#      spectral_centroid_mean: 1.5575
#      spectral_bandwidth_mean: 1.4448
#      rolloff_mean: 2.3394
#      zero_crossing_rate_mean: -1.6914
#
# 🎯 Performanță model:
#    ✓ Accuracy (antrenare): 0.9437 (94.38%)
#    ✓ Accuracy (test):      0.9000 (90.00%)
#
# 📊 Matrice de confuzie (date test):
#    ╔═════════════════╦═══════════════════╗
#    ║                 ║    Predicții      ║
#    ╠═════════════════╬═════════╦═════════╣
#    ║                 ║  Rock   ║Classical║
#    ╠═════════════════╬═════════╬═════════╣
#    ║ Real Rock       ║    18   ║     2   ║
#    ║ Real Classical  ║     2   ║    18   ║
#    ╚═════════════════╩═════════╩═════════╝
#
#    Interpretare:
#    - True Negatives (TN):  18 - Rock clasificat corect ca Rock
#    - False Positives (FP): 2 - Rock clasificat greșit ca Classical
#    - False Negatives (FN): 2 - Classical clasificat greșit ca Rock
#    - True Positives (TP):  18 - Classical clasificat corect ca Classical
#
# 📈 Raport de clasificare detaliat:
#               precision    recall  f1-score   support
#
#         Rock     0.9000    0.9000    0.9000        20
#    Classical     0.9000    0.9000    0.9000        20
#
#     accuracy                         0.9000        40
#    macro avg     0.9000    0.9000    0.9000        40
# weighted avg     0.9000    0.9000    0.9000        40
#
# ────────────────────────────────────────────────────────────────────────────────
#
# ✓ Grafic salvat: problema_7_12_music_clasificare.png
#
#
# ================================================================================
#  SUMAR FINAL - REZULTATE TEMA
# ================================================================================
#
# 📊 PROBLEMA 7.1 - PREDICȚIA BACȘIȘULUI:
#    ✓ Model: Regresie Liniară
#    ✓ Dataset: tips.csv (244 observații)
#    ✓ Ecuație: tip = 0.9252 + 0.1070 × total_bill
#    ✓ R² Score: 0.5449 (54.49%)
#    ✓ RMSE: $0.7542
#
# 🎵 PROBLEMA 7.12 - DISCRIMINARE ROCK vs CLASSICAL:
#    ✓ Model: Regresie Logistică
#    ✓ Dataset: features_30_sec.csv (200 piese)
#    ✓ Features: 5 caracteristici audio
#    ✓ Accuracy: 0.9000 (90.00%)
#    ✓ Clasificări corecte: 36/40
#
# ================================================================================
# ✅ ANALIZA COMPLETĂ! Ambele probleme au fost rezolvate cu succes!
# ================================================================================
#
#
# Process finished with exit code 0
# ==============================================================================
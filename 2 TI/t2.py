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
from sklearn.preprocessing import StandardScaler
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
tips_data = pd.read_csv('tips_IT.csv')

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
print(f"     → Pentru fiecare dolar adițional la masă, bacșișul crește cu ${model_tips.coef_[0]:.4f}")
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
    print(f"   - Masă de ${bill[0]:.2f} → Bacșiș estimat: ${pred:.2f}")

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
# Dataset: artists.csv (caracteristici audio)
# Model: Logistic Regression
# ==============================================================================

print("\n" + "█"*80)
print("█ PROBLEMA 7.12: DISCRIMINARE ROCK vs CLASSICAL - pp 171 (Cook, Swayne, 2007)")
print("█"*80 + "\n")

# Încărcare dataset muzică
print("🎵 Se încarcă dataset-ul artists.csv...\n")
music_data = pd.read_csv('music-sub_IT.csv')

print("📊 Informații despre dataset:")
print(music_data.info())

print("\n📋 Primele 10 înregistrări:")
print(music_data.head(10))

print(f"\n📈 Dimensiuni dataset: {music_data.shape[0]} piese, {music_data.shape[1]} coloane")

# Afișare distribuția genurilor
print("\n🎸 Distribuția genurilor muzicale:")
print(music_data['Type'].value_counts())

# Filtrare doar genurile Rock și Classical
print("\n🔍 Se filtrează doar genurile 'Rock' și 'Classical'...")
music_filtered = music_data[music_data['Type'].isin(['Rock', 'Classical'])].copy()

print(f"\n📊 Distribuția claselor după filtrare:")
print(music_filtered['Type'].value_counts())
print(f"   - Total piese: {len(music_filtered)}")

# Selectare features pentru clasificare (caracteristici audio)
feature_cols = ['LVar', 'LAve', 'LMax', 'LFEner', 'LFreq']

print(f"\n✓ Features selectate pentru clasificare:")
for feat in feature_cols:
    print(f"   - {feat}")

print("\n📉 Statistici descriptive pentru features:")
print(music_filtered[feature_cols].describe())

# Preparare date
X_music = music_filtered[feature_cols].values

# Convertim Type în valori numerice: Rock=0, Classical=1
y_music = (music_filtered['Type'] == 'Classical').astype(int).values

print(f"\n🎯 Encoding-ul claselor:")
print(f"   - Rock = 0 ({np.sum(y_music == 0)} piese)")
print(f"   - Classical = 1 ({np.sum(y_music == 1)} piese)")

# Împărțire date cu stratificare
X_train_music, X_test_music, y_train_music, y_test_music = train_test_split(
    X_music, y_music, test_size=0.2, random_state=42, stratify=y_music
)

print(f"\n🔀 Împărțire date:")
print(f"   - Antrenare: {len(X_train_music)} observații")
print(f"     → Rock: {np.sum(y_train_music == 0)}, Classical: {np.sum(y_train_music == 1)}")
print(f"   - Test: {len(X_test_music)} observații")
print(f"     → Rock: {np.sum(y_test_music == 0)}, Classical: {np.sum(y_test_music == 1)}")

# Normalizare features (crucial pentru regresie logistică!)
scaler = StandardScaler()
X_train_music_scaled = scaler.fit_transform(X_train_music)
X_test_music_scaled = scaler.transform(X_test_music)

print("\n✓ Features normalizate (StandardScaler)")

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
print(f"\n   - Coeficienți (βᵢ) pentru features:")
for feat, coef in zip(feature_cols, model_music.coef_[0]):
    print(f"     • {feat:10s}: {coef:8.4f}")

print(f"\n💡 Interpretare coeficienți:")
print(f"   - Coeficient pozitiv → crește probabilitatea de Classical")
print(f"   - Coeficient negativ → crește probabilitatea de Rock")

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

# Calcul metrici suplimentare
precision_rock = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
recall_rock = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
precision_classical = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall_classical = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0

print(f"\n📈 Metrici detaliate:")
print(f"   Rock:")
print(f"     - Precision: {precision_rock:.4f} - Din predicțiile Rock, {precision_rock*100:.1f}% sunt corecte")
print(f"     - Recall:    {recall_rock:.4f} - Din Rock-urile reale, {recall_rock*100:.1f}% sunt detectate")
print(f"   Classical:")
print(f"     - Precision: {precision_classical:.4f} - Din predicțiile Classical, {precision_classical*100:.1f}% sunt corecte")
print(f"     - Recall:    {recall_classical:.4f} - Din Classical-urile reale, {recall_classical*100:.1f}% sunt detectate")

print(f"\n📋 Raport de clasificare complet:")
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
axes[0, 0].set_title('Problema 7.12: Matrice de Confuzie - Rock vs Classical',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Adevărată etichetă', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Predicția modelului', fontsize=11, fontweight='bold')

# Grafic 2: Distribuția probabilităților
prob_rock = y_pred_proba[y_test_music == 0][:, 0]
prob_classical = y_pred_proba[y_test_music == 1][:, 1]

axes[0, 1].hist(prob_rock, alpha=0.6, bins=10, label='Rock (prob. Rock)',
               color='#2E86AB', edgecolor='black', linewidth=1.5)
axes[0, 1].hist(prob_classical, alpha=0.6, bins=10, label='Classical (prob. Classical)',
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
colors = ['#F18F01' if model_music.coef_[0][i] > 0 else '#2E86AB' for i in sorted_idx]
axes[1, 0].barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=colors, edgecolor='black')
axes[1, 0].set_yticks(range(len(sorted_idx)))
axes[1, 0].set_yticklabels([feature_cols[i] for i in sorted_idx], fontsize=10)
axes[1, 0].set_xlabel('Importanță (|coeficient|)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Importanța caracteristicilor audio în clasificare', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')
# Legendă pentru culori
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#F18F01', label='Favorizează Classical'),
                   Patch(facecolor='#2E86AB', label='Favorizează Rock')]
axes[1, 0].legend(handles=legend_elements, fontsize=9)

# Grafic 4: Scatter plot 2 features principale (LVar vs LFreq)
idx1, idx2 = 0, 4  # LVar și LFreq
axes[1, 1].scatter(X_test_music[y_test_music == 0, idx1],
                  X_test_music[y_test_music == 0, idx2],
                  color='#2E86AB', alpha=0.7, s=100, label='Rock', edgecolor='black', linewidth=1.5)
axes[1, 1].scatter(X_test_music[y_test_music == 1, idx1],
                  X_test_music[y_test_music == 1, idx2],
                  color='#A23B72', alpha=0.7, s=100, label='Classical', edgecolor='black', linewidth=1.5)
axes[1, 1].set_xlabel(feature_cols[idx1], fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel(feature_cols[idx2], fontsize=11, fontweight='bold')
axes[1, 1].set_title('Separare claselor: LVar vs LFreq', fontsize=12, fontweight='bold')
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
print(f"   ✓ MAE: ${mae_test:.4f}")

print(f"\n🎵 PROBLEMA 7.12 - DISCRIMINARE ROCK vs CLASSICAL:")
print(f"   ✓ Model: Regresie Logistică")
print(f"   ✓ Dataset: artists.csv ({len(music_filtered)} piese Rock/Classical)")
print(f"   ✓ Features: {len(feature_cols)} caracteristici audio (LVar, LAve, LMax, LFEner, LFreq)")
print(f"   ✓ Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print(f"   ✓ Clasificări corecte: {cm[0,0] + cm[1,1]}/{len(y_test_music)}")
print(f"   ✓ Erori: {cm[0,1] + cm[1,0]}/{len(y_test_music)}")

print("\n📈 Comparație performanță:")
print(f"   - Regresie liniară (tips): R² = {r2_test:.4f}")
print(f"   - Clasificare (music): Accuracy = {accuracy_test:.4f}")

print("\n" + "="*80)
print("✅ ANALIZA COMPLETĂ! Ambele probleme au fost rezolvate cu succes!")
print("   Grafice salvate:")
print("   • problema_7_1_tips_regresie.png")
print("   • problema_7_12_music_clasificare.png")
print("="*80 + "\n")


# ================================================================================================================
# Rulaj:
#
# C:\Users\Daniel\PycharmProjects\.venv\Scripts\python.exe "C:\Users\Daniel\PycharmProjects\2 TI\t2.py"
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
# Data columns (total 7 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   total_bill  244 non-null    float64
#  1   tip         244 non-null    float64
#  2   sex         244 non-null    object
#  3   smoker      244 non-null    object
#  4   day         244 non-null    object
#  5   time        244 non-null    object
#  6   size        244 non-null    int64
# dtypes: float64(2), int64(1), object(4)
# memory usage: 13.5+ KB
# None
#
# 📋 Primele 10 înregistrări:
#    total_bill   tip     sex smoker  day    time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# 2       21.01  3.50    Male     No  Sun  Dinner     3
# 3       23.68  3.31    Male     No  Sun  Dinner     2
# 4       24.59  3.61  Female     No  Sun  Dinner     4
# 5       25.29  4.71    Male     No  Sun  Dinner     4
# 6        8.77  2.00    Male     No  Sun  Dinner     2
# 7       26.88  3.12    Male     No  Sun  Dinner     4
# 8       15.04  1.96    Male     No  Sun  Dinner     2
# 9       14.78  3.23    Male     No  Sun  Dinner     2
#
# 📈 Dimensiuni dataset: 244 observații, 7 variabile
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
#      → Pentru fiecare dolar adițional la masă, bacșișul crește cu $0.1070
#      → Rata de bacșiș: 10.70%
#
# 📊 Metrice de performanță:
#    ✓ R² (antrenare): 0.4310 (43.10% variabilitate explicată)
#    ✓ R² (test):      0.5449 (54.49% variabilitate explicată)
#    ✓ RMSE (test):    $0.7542
#    ✓ MAE (test):     $0.6209
#
# 🔮 Exemple de predicții:
#    - Masă de $10.00 → Bacșiș estimat: $1.99
#    - Masă de $20.00 → Bacșiș estimat: $3.06
#    - Masă de $30.00 → Bacșiș estimat: $4.13
#    - Masă de $50.00 → Bacșiș estimat: $6.27
# ────────────────────────────────────────────────────────────────────────────────
#
# ✓ Grafic salvat: problema_7_1_tips_regresie.png
#
#
# ████████████████████████████████████████████████████████████████████████████████
# █ PROBLEMA 7.12: DISCRIMINARE ROCK vs CLASSICAL - pp 171 (Cook, Swayne, 2007)
# ████████████████████████████████████████████████████████████████████████████████
#
# 🎵 Se încarcă dataset-ul artists.csv...
#
# 📊 Informații despre dataset:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 62 entries, 0 to 61
# Data columns (total 8 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   Unnamed: 0  62 non-null     object
#  1   Artist      62 non-null     object
#  2   Type        62 non-null     object
#  3   LVar        62 non-null     float64
#  4   LAve        62 non-null     float64
#  5   LMax        62 non-null     int64
#  6   LFEner      62 non-null     float64
#  7   LFreq       62 non-null     float64
# dtypes: float64(4), int64(1), object(3)
# memory usage: 4.0+ KB
# None
#
# 📋 Primele 10 înregistrări:
#        Unnamed: 0 Artist  Type  ...   LMax     LFEner      LFreq
# 0   Dancing Queen   Abba  Rock  ...  29921  105.92095   59.57379
# 1      Knowing Me   Abba  Rock  ...  27626  102.83616   58.48031
# 2   Take a Chance   Abba  Rock  ...  26372  102.32488  124.59397
# 3       Mamma Mia   Abba  Rock  ...  28898  101.61648   48.76513
# 4     Lay All You   Abba  Rock  ...  27940  100.30076   74.02039
# 5   Super Trouper   Abba  Rock  ...  25531  100.24848   81.40140
# 6  I Have A Dream   Abba  Rock  ...  14699  104.59686  305.18689
# 7      The Winner   Abba  Rock  ...   8928  104.34921  277.66056
# 8           Money   Abba  Rock  ...  22962  102.24066  165.15799
# 9             SOS   Abba  Rock  ...  15517  104.36243  146.73700
#
# [10 rows x 8 columns]
#
# 📈 Dimensiuni dataset: 62 piese, 8 coloane
#
# 🎸 Distribuția genurilor muzicale:
# Type
# Rock         32
# Classical    27
# New wave      3
# Name: count, dtype: int64
#
# 🔍 Se filtrează doar genurile 'Rock' și 'Classical'...
#
# 📊 Distribuția claselor după filtrare:
# Type
# Rock         32
# Classical    27
# Name: count, dtype: int64
#    - Total piese: 59
#
# ✓ Features selectate pentru clasificare:
#    - LVar
#    - LAve
#    - LMax
#    - LFEner
#    - LFreq
#
# 📉 Statistici descriptive pentru features:
#                LVar        LAve         LMax      LFEner       LFreq
# count  5.900000e+01   59.000000     59.00000   59.000000   59.000000
# mean   2.071054e+07   -7.606237  22812.59322  104.068546  238.356477
# std    2.685230e+07   48.401751   8737.73016    5.605229  178.082421
# min    2.936083e+05  -98.062924   2985.00000   83.881950   41.405150
# 25%    3.117825e+06   -6.172145  17294.00000  101.642950  103.754710
# 50%    8.368953e+06   -2.474896  24633.00000  104.350050  176.534410
# 75%    2.685274e+07    2.399587  30011.50000  108.321380  322.787765
# max    1.294722e+08  216.231759  32766.00000  114.002290  877.772430
#
# 🎯 Encoding-ul claselor:
#    - Rock = 0 (32 piese)
#    - Classical = 1 (27 piese)
#
# 🔀 Împărțire date:
#    - Antrenare: 47 observații
#      → Rock: 25, Classical: 22
#    - Test: 12 observații
#      → Rock: 7, Classical: 5
#
# ✓ Features normalizate (StandardScaler)
#
# 🤖 Se antrenează modelul de clasificare (Regresie Logistică)...
#
# ────────────────────────────────────────────────────────────────────────────────
# REZULTATE MODEL CLASIFICARE - DISCRIMINARE ROCK vs CLASSICAL
# ────────────────────────────────────────────────────────────────────────────────
#
# 📐 Parametrii modelului de regresie logistică:
#    - Intercept (β₀): -0.2271
#
#    - Coeficienți (βᵢ) pentru features:
#      • LVar      :  -1.3128
#      • LAve      :   1.4183
#      • LMax      :  -0.9869
#      • LFEner    :   0.1402
#      • LFreq     :   1.2548
#
# 💡 Interpretare coeficienți:
#    - Coeficient pozitiv → crește probabilitatea de Classical
#    - Coeficient negativ → crește probabilitatea de Rock
#
# 🎯 Performanță model:
#    ✓ Accuracy (antrenare): 0.9574 (95.74%)
#    ✓ Accuracy (test):      0.8333 (83.33%)
#
# 📊 Matrice de confuzie (date test):
#    ╔═════════════════╦═══════════════════╗
#    ║                 ║    Predicții      ║
#    ╠═════════════════╬═════════╦═════════╣
#    ║                 ║  Rock   ║Classical║
#    ╠═════════════════╬═════════╬═════════╣
#    ║ Real Rock       ║     5   ║     2   ║
#    ║ Real Classical  ║     0   ║     5   ║
#    ╚═════════════════╩═════════╩═════════╝
#
#    Interpretare:
#    - True Negatives (TN):  5 - Rock clasificat corect ca Rock
#    - False Positives (FP): 2 - Rock clasificat greșit ca Classical
#    - False Negatives (FN): 0 - Classical clasificat greșit ca Rock
#    - True Positives (TP):  5 - Classical clasificat corect ca Classical
#
# 📈 Metrici detaliate:
#    Rock:
#      - Precision: 1.0000 - Din predicțiile Rock, 100.0% sunt corecte
#      - Recall:    0.7143 - Din Rock-urile reale, 71.4% sunt detectate
#    Classical:
#      - Precision: 0.7143 - Din predicțiile Classical, 71.4% sunt corecte
#      - Recall:    1.0000 - Din Classical-urile reale, 100.0% sunt detectate
#
# 📋 Raport de clasificare complet:
#               precision    recall  f1-score   support
#
#         Rock     1.0000    0.7143    0.8333         7
#    Classical     0.7143    1.0000    0.8333         5
#
#     accuracy                         0.8333        12
#    macro avg     0.8571    0.8571    0.8333        12
# weighted avg     0.8810    0.8333    0.8333        12
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
#    ✓ MAE: $0.6209
#
# 🎵 PROBLEMA 7.12 - DISCRIMINARE ROCK vs CLASSICAL:
#    ✓ Model: Regresie Logistică
#    ✓ Dataset: artists.csv (59 piese Rock/Classical)
#    ✓ Features: 5 caracteristici audio (LVar, LAve, LMax, LFEner, LFreq)
#    ✓ Accuracy: 0.8333 (83.33%)
#    ✓ Clasificări corecte: 10/12
#    ✓ Erori: 2/12
#
# 📈 Comparație performanță:
#    - Regresie liniară (tips): R² = 0.5449
#    - Clasificare (music): Accuracy = 0.8333
#
# ================================================================================
# ✅ ANALIZA COMPLETĂ! Ambele probleme au fost rezolvate cu succes!
#    Grafice salvate:
#    • problema_7_1_tips_regresie.png
#    • problema_7_12_music_clasificare.png
# ================================================================================
#
#
# Process finished with exit code 0
# ================================================================================================================
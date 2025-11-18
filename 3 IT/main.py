import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Setăm seed-ul pentru reproducibilitate
np.random.seed(42)

# 1. ÎNCĂRCAREA ȘI PREPROCESAREA DATELOR
print("=" * 60)
print("ÎNCĂRCAREA ȘI PREPROCESAREA DATELOR")
print("=" * 60)

# Citim datele
df = pd.read_csv('breast-cancer-wisconsin.csv')

# Curățăm numele coloanelor (eliminăm spațiile)
df.columns = df.columns.str.strip()
print(f"Dimensiunea dataset-ului: {df.shape}")
print(f"Coloane: {list(df.columns)}")
print(f"\nPrimele rânduri:\n{df.head()}")

# Eliminăm coloana Id (non-informativă)
df = df.drop('Id', axis=1)
print(f"\nDupă eliminarea Id: {df.shape}")

# Verificăm valorile lipsă
print(f"\nValori lipsă în fiecare coloană:")
print(df.isin(['?']).sum())

# Înlocuim '?' cu NaN și apoi cu media coloanei
df = df.replace('?', np.nan)
for col in df.columns[:-1]:  # Toate coloanele except Class
    df[col] = pd.to_numeric(df[col])
    if df[col].isnull().any():
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)
        print(f"Coloana '{col}': înlocuite cu media {mean_value:.2f}")

# Convertim clasa: 2 (benign) -> 0, 4 (malignant) -> 1
df['Class'] = df['Class'].astype(int)
df['Class'] = df['Class'].map({2: 0, 4: 1})

print(f"\nDistribuția claselor:")
print(df['Class'].value_counts())
print(f"Benign (0): {(df['Class'] == 0).sum()}, Malignant (1): {(df['Class'] == 1).sum()}")

# Separăm features și target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"\nFeature names: {list(X.columns)}")

# 2. EXPERIMENTARE CU DIFERITE SCALERE ȘI PARAMETRI
print("\n" + "=" * 60)
print("EXPERIMENTARE CU DIFERITE CONFIGURAȚII")
print("=" * 60)

# Configurații de testat
scalers_config = {
    'Standard Scaler': StandardScaler(),
    'MinMax Scaler': MinMaxScaler(),
    'No Scaling': None
}

kernels_config = ['linear', 'rbf', 'poly']

results_summary = []

# Rulăm 30 de experimente pentru fiecare configurație
n_runs = 30

for scaler_name, scaler in scalers_config.items():
    for kernel in kernels_config:
        print(f"\n--- {scaler_name} + Kernel {kernel} ---")
        accuracies = []

        for run in range(n_runs):
            # Split train-test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=run, stratify=y
            )

            # Scaling
            if scaler is not None:
                scaler_fitted = scaler.fit(X_train)
                X_train_scaled = scaler_fitted.transform(X_train)
                X_test_scaled = scaler_fitted.transform(X_test)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values

            # Antrenare SVM
            if kernel == 'rbf':
                svm = SVC(kernel=kernel, gamma='scale', random_state=42)
            else:
                svm = SVC(kernel=kernel, random_state=42)

            svm.fit(X_train_scaled, y_train)

            # Predicție
            y_pred = svm.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        results_summary.append({
            'Scaler': scaler_name,
            'Kernel': kernel,
            'Mean Accuracy': mean_acc,
            'Std Accuracy': std_acc
        })

        print(f"Acuratețe medie: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")

# Afișăm cel mai bun rezultat
results_df = pd.DataFrame(results_summary)
best_result = results_df.loc[results_df['Mean Accuracy'].idxmax()]
print("\n" + "=" * 60)
print("CEA MAI BUNĂ CONFIGURAȚIE")
print("=" * 60)
print(best_result)

# 3. ANALIZĂ DETALIATĂ CU CEA MAI BUNĂ CONFIGURAȚIE
print("\n" + "=" * 60)
print("ANALIZĂ DETALIATĂ")
print("=" * 60)

best_scaler_name = best_result['Scaler']
best_kernel = best_result['Kernel']

# Recreăm cel mai bun scaler
if best_scaler_name == 'Standard Scaler':
    best_scaler = StandardScaler()
elif best_scaler_name == 'MinMax Scaler':
    best_scaler = MinMaxScaler()
else:
    best_scaler = None

# Split final
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
if best_scaler is not None:
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)
else:
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

# Antrenare model final
if best_kernel == 'rbf':
    final_svm = SVC(kernel=best_kernel, gamma='scale', random_state=42)
else:
    final_svm = SVC(kernel=best_kernel, random_state=42)

final_svm.fit(X_train_scaled, y_train)

# Predicții
y_pred = final_svm.predict(X_test_scaled)

# Metrici
print(f"\nAccuracy pe test set: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nMatricea de confuzie:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nRaport de clasificare:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# 4. VARIABLE IMPORTANCE (Permutation Importance)
print("\n" + "=" * 60)
print("IMPORTANȚA VARIABILELOR")
print("=" * 60)

perm_importance = permutation_importance(
    final_svm, X_test_scaled, y_test, n_repeats=10, random_state=42
)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\nImportanța features (top 5):")
print(importance_df.head())

# 5. VIZUALIZĂRI
print("\n" + "=" * 60)
print("GENERARE VIZUALIZĂRI")
print("=" * 60)

# Vizualizare 1: Compararea performanței pentru diferite configurații
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
pivot_data = results_df.pivot(index='Kernel', columns='Scaler', values='Mean Accuracy')
sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy'})
plt.title('Compararea Acurateței pentru Diferite Configurații')
plt.ylabel('Kernel')
plt.xlabel('Scaler')

# Vizualizare 2: Importanța variabilelor
plt.subplot(1, 2, 2)
top_features = importance_df.head(9)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importanță')
plt.title('Top 9 Feature Importance')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('svm_analysis_1.png', dpi=300, bbox_inches='tight')
print("Salvat: svm_analysis_1.png")

# Vizualizare 3: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title(f'Matricea de Confuzie\n{best_scaler_name} + Kernel {best_kernel}')
plt.ylabel('Adevărat')
plt.xlabel('Prezis')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Salvat: confusion_matrix.png")

# Vizualizare 4: Support Vectors pentru cele mai importante 2 variabile
plt.figure(figsize=(10, 8))

# Găsim indicii celor mai importante 2 features
top_2_indices = [X.columns.get_loc(importance_df.iloc[i]['Feature']) for i in range(2)]
feature1_name = importance_df.iloc[0]['Feature']
feature2_name = importance_df.iloc[1]['Feature']

# Plotăm datele de antrenare
plt.subplot(2, 1, 1)
plt.scatter(X_train_scaled[:, top_2_indices[0]],
            X_train_scaled[:, top_2_indices[1]],
            c=y_train, cmap='coolwarm', alpha=0.6, edgecolors='k')
plt.xlabel(feature1_name)
plt.ylabel(feature2_name)
plt.title('Date de Antrenare')
plt.colorbar(label='Clasă')

# Plotăm support vectors
support_vectors = final_svm.support_vectors_
plt.scatter(support_vectors[:, top_2_indices[0]],
            support_vectors[:, top_2_indices[1]],
            s=200, linewidth=1.5, facecolors='none', edgecolors='green',
            label='Support Vectors')
plt.legend()

# Plotăm datele de test
plt.subplot(2, 1, 2)
plt.scatter(X_test_scaled[:, top_2_indices[0]],
            X_test_scaled[:, top_2_indices[1]],
            c=y_test, cmap='coolwarm', alpha=0.6, edgecolors='k')
plt.xlabel(feature1_name)
plt.ylabel(feature2_name)
plt.title('Date de Test')
plt.colorbar(label='Clasă')

plt.tight_layout()
plt.savefig('support_vectors.png', dpi=300, bbox_inches='tight')
print("Salvat: support_vectors.png")

# 6. STATISTICI FINALE
print("\n" + "=" * 60)
print("STATISTICI FINALE")
print("=" * 60)
print(f"Număr total de support vectors: {len(final_svm.support_vectors_)}")
print(f"Procent din datele de antrenare: {len(final_svm.support_vectors_) / len(X_train) * 100:.2f}%")
print(f"\nSupport vectors per clasă:")
print(f"Clasa 0 (Benign): {np.sum(y_train.iloc[final_svm.support_] == 0)}")
print(f"Clasa 1 (Malignant): {np.sum(y_train.iloc[final_svm.support_] == 1)}")

print("\n" + "=" * 60)
print("ANALIZĂ COMPLETĂ - SUCCES!")
print("=" * 60)
print("\nFișiere generate:")
print("1. svm_analysis_1.png - Compararea configurațiilor și importanța features")
print("2. confusion_matrix.png - Matricea de confuzie")
print("3. support_vectors.png - Vizualizarea support vectors")
print("\nConcluzii:")
print(f"- Cea mai bună configurație: {best_scaler_name} + Kernel {best_kernel}")
print(f"- Acuratețe: {best_result['Mean Accuracy']:.4f} ± {best_result['Std Accuracy']:.4f}")
print(f"- Cele mai importante variabile: {importance_df.iloc[0]['Feature']}, {importance_df.iloc[1]['Feature']}")
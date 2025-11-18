"""
Implementare Completă - Tema Cursurile 6 și 7
Wine Dataset: Performance Evaluation, Feature Selection & Parameter Tuning

Autor: [Numele tău]
Dataset: Wine Recognition Data (178 samples, 13 features, 3 classes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif,
    SequentialFeatureSelector, chi2
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, cohen_kappa_score, roc_curve, auc, roc_auc_score
)

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# 1. ÎNCĂRCAREA ȘI PREGĂTIREA DATELOR
# ==============================================================================

def load_wine_data():
    """Încarcă Wine dataset din fișier sau sklearn"""
    try:
        # Încarcă din fișier wine.data
        data = pd.read_csv('wine.data', header=None)

        # Numele coloanelor conform wine.names
        columns = ['Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash',
                   'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
                   'Proanthocyanins', 'Color_intensity', 'Hue',
                   'OD280_OD315_of_diluted_wines', 'Proline']
        data.columns = columns

        X = data.drop('Class', axis=1).values
        y = data['Class'].values
        feature_names = columns[1:]

        print("✓ Date încărcate din fișier wine.data")

    except FileNotFoundError:
        # Alternativ: încarcă din sklearn
        from sklearn.datasets import load_wine
        wine = load_wine()
        X = wine.data
        y = wine.target
        feature_names = wine.feature_names
        print("✓ Date încărcate din sklearn.datasets")

    # Standardizare date
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)} (counts: {np.bincount(y)})")
    print(f"Features: {len(feature_names)}\n")

    return X_scaled, y, feature_names

# ==============================================================================
# 2. PERFORMANCE EVALUATION (CURS 6)
# ==============================================================================

def random_subsampling_cv(model, X, y, n_runs=30, test_size=0.33):
    """
    Cross-validation prin random subsampling cu 30 runs

    Returns:
        Dictionary cu toate metricile
    """
    accuracies = []
    f1_scores = []
    kappa_scores = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    print(f"Rulare {n_runs} iterații random subsampling...")

    for i in range(n_runs):
        # Split aleatoriu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i
        )

        # Antrenare și predicție
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrici
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_test, y_pred)

        accuracies.append(acc)
        f1_scores.append(f1)
        kappa_scores.append(kappa)

        # Salvare predicții pentru confusion matrix
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Probabilități pentru ROC (dacă modelul le suportă)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            all_y_proba.append((y_test, y_proba))

    results = {
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'kappa_scores': kappa_scores,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'all_y_proba': all_y_proba,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1': np.mean(f1_scores),
        'mean_kappa': np.mean(kappa_scores)
    }

    return results

def print_performance_metrics(results, model_name):
    """Afișează metricile de performanță"""
    print(f"\n{'='*60}")
    print(f"REZULTATE PERFORMANȚĂ - {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"F1-Score: {results['mean_f1']:.4f}")
    print(f"Cohen's Kappa: {results['mean_kappa']:.4f}")

    # Interpretare Cohen's Kappa
    kappa = results['mean_kappa']
    if kappa < 0:
        interpretation = "No agreement"
    elif kappa < 0.2:
        interpretation = "Slight agreement"
    elif kappa < 0.4:
        interpretation = "Fair agreement"
    elif kappa < 0.6:
        interpretation = "Moderate agreement"
    elif kappa < 0.8:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    print(f"  → Interpretare: {interpretation}")

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plotare confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.show()

    print(f"\nConfusion Matrix:\n{cm}")
    return cm

def plot_roc_curves(y_proba_list, model_name, n_classes=3):
    """
    Plotare ROC curves pentru clasificare multiclasă (One-vs-Rest)
    """
    # Concatenare toate predicțiile
    y_true_all = np.concatenate([yt for yt, _ in y_proba_list])
    y_proba_all = np.vstack([yp for _, yp in y_proba_list])

    # Binarizare etichete pentru One-vs-Rest
    # Clasele în Wine sunt 1, 2, 3 (nu 0, 1, 2)
    unique_classes = np.unique(y_true_all)
    y_true_bin = label_binarize(y_true_all, classes=unique_classes)

    # Calcul ROC pentru fiecare clasă
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green']
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'Class {i+1} (AUC = {roc_auc[i]:.3f})')

    # Linia diagonală (clasificator aleatoriu)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves (One-vs-Rest) - {model_name}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.show()

    print(f"\nAUC Scores:")
    for i in range(n_classes):
        print(f"  Class {i+1}: {roc_auc[i]:.4f}")
    print(f"  Mean AUC: {np.mean(list(roc_auc.values())):.4f}")

    return roc_auc

def wilcoxon_test(results1, results2, name1, name2):
    """
    Test Wilcoxon pentru compararea a două modele
    """
    print(f"\n{'='*60}")
    print(f"WILCOXON SIGNED-RANK TEST")
    print(f"Comparare: {name1} vs {name2}")
    print(f"{'='*60}")

    scores1 = results1['accuracies']
    scores2 = results2['accuracies']

    # Test Wilcoxon
    statistic, p_value = wilcoxon(scores1, scores2)

    print(f"Mean Accuracy {name1}: {np.mean(scores1):.4f}")
    print(f"Mean Accuracy {name2}: {np.mean(scores2):.4f}")
    print(f"Wilcoxon Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")

    # Interpretare
    alpha = 0.05
    if p_value <= alpha:
        print(f"\n✓ REZULTAT: Diferență statistică semnificativă (p ≤ {alpha})")
        print("  → Respingem ipoteza nulă H0")
        print("  → Cele două modele au performanțe diferite")

        if np.mean(scores1) > np.mean(scores2):
            print(f"  → {name1} este superior lui {name2}")
        else:
            print(f"  → {name2} este superior lui {name1}")
    else:
        print(f"\n✗ REZULTAT: Nu există diferență statistică semnificativă (p > {alpha})")
        print("  → Nu respingem ipoteza nulă H0")
        print("  → Cele două modele au performanțe similare")

    return statistic, p_value

# ==============================================================================
# 3. FEATURE SELECTION (CURS 7)
# ==============================================================================

def variance_threshold_selection(X, y, feature_names, threshold=0.1):
    """
    FILTER METHOD 1: Variance Threshold
    Elimină features cu varianță foarte mică
    """
    print(f"\n{'='*60}")
    print("FEATURE SELECTION - Variance Threshold")
    print(f"{'='*60}")

    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)

    # Feature-uri selectate
    selected_features = [feature_names[i] for i in range(len(feature_names))
                        if selector.get_support()[i]]

    print(f"Threshold: {threshold}")
    print(f"Features originale: {X.shape[1]}")
    print(f"Features selectate: {X_selected.shape[1]}")
    print(f"Features eliminate: {X.shape[1] - X_selected.shape[1]}")
    print(f"\nFeatures păstrate: {selected_features}")

    return X_selected, selected_features, selector

def selectkbest_selection(X, y, feature_names, k=8, score_func=mutual_info_classif):
    """
    FILTER METHOD 2: SelectKBest cu Mutual Information
    Selectează top K features bazat pe information gain
    """
    print(f"\n{'='*60}")
    print("FEATURE SELECTION - SelectKBest (Mutual Information)")
    print(f"{'='*60}")

    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)

    # Scoruri și features selectate
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    # Sortare după scor
    feature_scores = list(zip(feature_names, scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"K (număr features): {k}")
    print(f"\nTop {k} features selectate:")
    for i, (feat, score) in enumerate(feature_scores[:k], 1):
        print(f"  {i}. {feat}: {score:.4f}")

    # Vizualizare scoruri
    plt.figure(figsize=(10, 6))
    features, scores_list = zip(*feature_scores)
    plt.barh(range(len(features)), scores_list, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Mutual Information Score')
    plt.title('Feature Importance - Mutual Information')
    plt.axvline(x=sorted(scores_list, reverse=True)[k-1], color='red',
                linestyle='--', label=f'Top {k} threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_selection_mutual_info.png', dpi=300)
    plt.show()

    return X_selected, selected_features, selector

def sequential_feature_selection(X, y, feature_names, model, n_features=8, direction='forward'):
    """
    WRAPPER METHOD: Sequential Feature Selection
    Greedy search pentru găsirea subset-ului optimal
    """
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION - Sequential ({direction.upper()})")
    print(f"{'='*60}")

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction=direction,
        cv=5,
        n_jobs=-1
    )

    print(f"Antrenare {direction} selection (poate dura câteva minute)...")
    X_selected = sfs.fit_transform(X, y)

    # Features selectate
    selected_indices = sfs.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    print(f"Features selectate ({n_features}): {selected_features}")

    return X_selected, selected_features, sfs

def compare_feature_selection_methods(X, y, feature_names):
    """
    Compară toate metodele de feature selection
    """
    print(f"\n{'='*70}")
    print("COMPARARE METODE FEATURE SELECTION")
    print(f"{'='*70}")

    # Model de bază pentru evaluare
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 1. Original (toate features)
    scores_original = cross_val_score(base_model, X, y, cv=5, scoring='accuracy')

    # 2. Variance Threshold
    X_var, _, _ = variance_threshold_selection(X, y, feature_names, threshold=0.1)
    scores_var = cross_val_score(base_model, X_var, y, cv=5, scoring='accuracy')

    # 3. SelectKBest
    X_kbest, _, _ = selectkbest_selection(X, y, feature_names, k=8)
    scores_kbest = cross_val_score(base_model, X_kbest, y, cv=5, scoring='accuracy')

    # 4. Sequential Forward Selection
    X_sfs, _, _ = sequential_feature_selection(
        X, y, feature_names,
        RandomForestClassifier(n_estimators=50, random_state=42),
        n_features=8, direction='forward'
    )
    scores_sfs = cross_val_score(base_model, X_sfs, y, cv=5, scoring='accuracy')

    # Tabel comparativ
    results_df = pd.DataFrame({
        'Method': ['Original (13 features)', 'Variance Threshold',
                   'SelectKBest (MI)', 'Sequential Forward'],
        'Features': [X.shape[1], X_var.shape[1], X_kbest.shape[1], X_sfs.shape[1]],
        'Mean Accuracy': [scores_original.mean(), scores_var.mean(),
                         scores_kbest.mean(), scores_sfs.mean()],
        'Std Accuracy': [scores_original.std(), scores_var.std(),
                        scores_kbest.std(), scores_sfs.std()]
    })

    print("\n" + "="*70)
    print(results_df.to_string(index=False))
    print("="*70)

    # Vizualizare
    plt.figure(figsize=(10, 6))
    methods = results_df['Method']
    means = results_df['Mean Accuracy']
    stds = results_df['Std Accuracy']

    plt.bar(range(len(methods)), means, yerr=stds, capsize=5,
            color=['gray', 'lightblue', 'lightgreen', 'orange'])
    plt.xticks(range(len(methods)), methods, rotation=15, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Comparație Metode Feature Selection')
    plt.ylim([0.8, 1.0])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png', dpi=300)
    plt.show()

    return results_df

# ==============================================================================
# 4. PARAMETER TUNING (CURS 7)
# ==============================================================================

def grid_search_tuning(X, y, model_type='neural_network'):
    """
    Grid Search pentru tuning parametri
    """
    print(f"\n{'='*60}")
    print(f"PARAMETER TUNING - Grid Search ({model_type.upper()})")
    print(f"{'='*60}")

    if model_type == 'neural_network':
        model = MLPClassifier(max_iter=1000, random_state=42)
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'bagging':
        model = BaggingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 20, 50, 100],
            'max_samples': [0.5, 0.7, 1.0],
            'max_features': [0.5, 0.7, 1.0]
        }
    else:
        raise ValueError(f"Model type '{model_type}' not supported")

    print(f"Parametri testați:")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")

    print(f"\nTotal combinații: {np.prod([len(v) for v in param_grid.values()])}")
    print("Rulare Grid Search (poate dura câteva minute)...")

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    print(f"\n{'='*60}")
    print("REZULTATE GRID SEARCH")
    print(f"{'='*60}")
    print(f"Best Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")

    # Top 5 configurații
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')

    print(f"\nTop 5 configurații:")
    for i, row in results_df.head(5).iterrows():
        print(f"\n  Rank {int(row['rank_test_score'])}:")
        print(f"    Score: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"    Params: {row['params']}")

    return grid_search.best_estimator_, grid_search.best_params_

# ==============================================================================
# 5. MAIN - RULARE COMPLETĂ
# ==============================================================================

def main():
    """
    Funcție principală - execută toate analizele
    """
    print("="*70)
    print(" WINE DATASET - MACHINE LEARNING ANALYSIS ".center(70))
    print(" Cursurile 6 & 7: Performance Evaluation, Feature Selection & Tuning ".center(70))
    print("="*70)

    # ========== ÎNCĂRCARE DATE ==========
    X, y, feature_names = load_wine_data()

    # ========== CURS 6: PERFORMANCE EVALUATION ==========
    print("\n" + "="*70)
    print(" PARTEA 1: PERFORMANCE EVALUATION (CURS 6) ".center(70))
    print("="*70)

    # Definire modele
    models = {
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50),
                                        max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Evaluare fiecare model
    all_results = {}
    for name, model in models.items():
        print(f"\n{'─'*70}")
        print(f"Evaluare model: {name}")
        print(f"{'─'*70}")

        results = random_subsampling_cv(model, X, y, n_runs=30)
        all_results[name] = results

        print_performance_metrics(results, name)
        plot_confusion_matrix(results['all_y_true'], results['all_y_pred'], name)

        if results['all_y_proba']:
            plot_roc_curves(results['all_y_proba'], name, n_classes=3)

    # Comparare celor mai bune 2 modele (Wilcoxon test)
    sorted_models = sorted(all_results.items(),
                          key=lambda x: x[1]['mean_accuracy'],
                          reverse=True)

    best_model1_name, best_model1_results = sorted_models[0]
    best_model2_name, best_model2_results = sorted_models[1]

    wilcoxon_test(best_model1_results, best_model2_results,
                  best_model1_name, best_model2_name)

    # ========== CURS 7: FEATURE SELECTION ==========
    print("\n\n" + "="*70)
    print(" PARTEA 2: FEATURE SELECTION (CURS 7) ".center(70))
    print("="*70)

    # Comparare metode
    comparison_df = compare_feature_selection_methods(X, y, feature_names)

    # ========== CURS 7: PARAMETER TUNING ==========
    print("\n\n" + "="*70)
    print(" PARTEA 3: PARAMETER TUNING (CURS 7) ".center(70))
    print("="*70)

    # Grid Search pentru Neural Network
    best_nn, best_params_nn = grid_search_tuning(X, y, model_type='neural_network')

    # Grid Search pentru Random Forest
    best_rf, best_params_rf = grid_search_tuning(X, y, model_type='random_forest')

    # Evaluare finală cu parametri optimizați
    print(f"\n{'='*70}")
    print("EVALUARE FINALĂ - MODELE OPTIMIZATE")
    print(f"{'='*70}")

    optimized_models = {
        'Neural Network (Optimized)': best_nn,
        'Random Forest (Optimized)': best_rf
    }

    for name, model in optimized_models.items():
        results = random_subsampling_cv(model, X, y, n_runs=30)
        print_performance_metrics(results, name)

    print("\n" + "="*70)
    print(" ANALIZĂ COMPLETATĂ CU SUCCES! ".center(70))
    print("="*70)
    print("\nFișiere generate:")
    print("  - confusion_matrix_*.png")
    print("  - roc_curve_*.png")
    print("  - feature_selection_*.png")

# ==============================================================================
# RULARE
# ==============================================================================

if __name__ == "__main__":
    main()
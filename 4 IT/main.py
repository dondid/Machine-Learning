"""
Implementare Completă - Metode Ansamblu
Decision Trees, Bagging, Boosting, Random Forests și Neural Networks
Vizualizări pas cu pas pentru înțelegere profundă
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

# Setare stil pentru ploturi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class EnsembleAnalysis:
    def __init__(self, data_path, target_column, task_type='classification'):
        """
        Inițializare analiză

        Parameters:
        -----------
        data_path : str - calea către fișierul CSV
        target_column : str - numele coloanei țintă
        task_type : str - 'classification' sau 'regression'
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type
        self.results = {}
        self.models = {}

        print(f"{'='*80}")
        print(f"ANALIZĂ METODE ANSAMBLU - {task_type.upper()}")
        print(f"{'='*80}\n")

    def load_and_preprocess(self):
        """Încărcare și preprocesare date"""
        print("📊 PASUL 1: ÎNCĂRCARE ȘI PREPROCESARE DATE")
        print("-" * 80)

        # Încărcare date
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Date încărcate: {self.df.shape[0]} rânduri, {self.df.shape[1]} coloane")

        # Afișare informații despre date
        print(f"\n📋 Primele 5 rânduri:")
        print(self.df.head())

        print(f"\n📈 Statistici descriptive pentru {self.target_column}:")
        print(self.df[self.target_column].describe())

        # Verificare valori lipsă
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Valori lipsă detectate:")
            print(missing[missing > 0])
            self.df = self.df.dropna()
            print(f"✓ Valori lipsă eliminate. Rânduri rămase: {self.df.shape[0]}")

        # Separare caracteristici și țintă
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Selectare doar coloane numerice pentru caracteristici
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        print(f"\n✓ Caracteristici numerice selectate: {len(numeric_cols)}")
        print(f"  Coloane: {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}")

        # Normalizare pentru Neural Network
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split train/test
        test_size = 0.3
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.X_train_scaled, self.X_test_scaled = train_test_split(
            X_scaled, test_size=test_size, random_state=42
        )[0:2]

        print(f"\n✓ Split date: {(1-test_size)*100:.0f}% antrenare, {test_size*100:.0f}% testare")
        print(f"  Train: {self.X_train.shape[0]} exemple")
        print(f"  Test:  {self.X_test.shape[0]} exemple")

        # Vizualizare distribuție țintă
        self._plot_target_distribution()

    def _plot_target_distribution(self):
        """Vizualizare distribuție variabilă țintă"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        if self.task_type == 'classification':
            # Pentru clasificare
            train_counts = self.y_train.value_counts()
            test_counts = self.y_test.value_counts()

            axes[0].bar(train_counts.index, train_counts.values, alpha=0.7, color='steelblue')
            axes[0].set_title('Distribuție Clasă - Set Antrenare', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Clasă')
            axes[0].set_ylabel('Frecvență')
            axes[0].grid(alpha=0.3)

            axes[1].bar(test_counts.index, test_counts.values, alpha=0.7, color='coral')
            axes[1].set_title('Distribuție Clasă - Set Testare', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Clasă')
            axes[1].set_ylabel('Frecvență')
            axes[1].grid(alpha=0.3)
        else:
            # Pentru regresie
            axes[0].hist(self.y_train, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0].set_title('Distribuție Țintă - Set Antrenare', fontsize=14, fontweight='bold')
            axes[0].set_xlabel(self.target_column)
            axes[0].set_ylabel('Frecvență')
            axes[0].axvline(self.y_train.mean(), color='red', linestyle='--', label=f'Medie: {self.y_train.mean():.2f}')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].hist(self.y_test, bins=30, alpha=0.7, color='coral', edgecolor='black')
            axes[1].set_title('Distribuție Țintă - Set Testare', fontsize=14, fontweight='bold')
            axes[1].set_xlabel(self.target_column)
            axes[1].set_ylabel('Frecvență')
            axes[1].axvline(self.y_test.mean(), color='red', linestyle='--', label=f'Medie: {self.y_test.mean():.2f}')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('01_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')  # Închide toate figurile pentru a elibera memoria
        print("\n✓ Grafic salvat: 01_target_distribution.png\n")

    def train_decision_tree(self):
        """PASUL 2: Decision Tree (Baseline)"""
        print(f"\n{'='*80}")
        print("🌳 PASUL 2: DECISION TREE (BASELINE)")
        print("-" * 80)

        if self.task_type == 'classification':
            model = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=10)
        else:
            model = DecisionTreeRegressor(max_depth=5, random_state=42, min_samples_split=10)

        # Antrenare
        print("🔄 Antrenare model...")
        model.fit(self.X_train, self.y_train)

        # Predicții
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        # Evaluare
        self._evaluate_model('Decision Tree', model, y_pred_train, y_pred_test)

        # Vizualizare arbore
        self._plot_decision_tree(model)

        # Salvare model
        self.models['Decision Tree'] = model

    def _plot_decision_tree(self, model):
        """Vizualizare arbore de decizie"""
        plt.figure(figsize=(20, 10))
        plot_tree(model,
                  feature_names=self.X_train.columns,
                  filled=True,
                  rounded=True,
                  fontsize=10,
                  max_depth=3)  # Limităm adâncimea pentru vizualizare
        plt.title('Decision Tree - Structură (primele 3 niveluri)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('02_decision_tree_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print("✓ Grafic salvat: 02_decision_tree_structure.png")

    def train_bagging(self):
        """PASUL 3: Bagging"""
        print(f"\n{'='*80}")
        print("🎒 PASUL 3: BAGGING (Bootstrap AGGregatING)")
        print("-" * 80)
        print("📖 Concept: Antrenează mai mulți arbori pe subseturi diferite (cu replacement)")
        print("           și combină predicțiile prin vot/medie")

        n_estimators = 25

        if self.task_type == 'classification':
            base_estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
            model = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        else:
            base_estimator = DecisionTreeRegressor(max_depth=5, random_state=42)
            model = BaggingRegressor(
                estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )

        # Antrenare cu vizualizare progres
        print(f"\n🔄 Antrenare {n_estimators} bags...")
        model.fit(self.X_train, self.y_train)

        print(f"✓ Antrenare completă!")
        print(f"  OOB Score: {model.oob_score_:.4f}")

        # Predicții
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        # Evaluare
        self._evaluate_model('Bagging', model, y_pred_train, y_pred_test)

        # Vizualizare diversitate bags
        self._plot_bagging_analysis(model)

        # Salvare model
        self.models['Bagging'] = model

    def _plot_bagging_analysis(self, model):
        """Analiză și vizualizare Bagging"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Predicții individuale vs ansamblu
        n_samples = min(50, len(self.X_test))  # Reducem la 50 pentru vizualizare clară
        sample_indices = np.random.choice(len(self.X_test), n_samples, replace=False)
        X_sample = self.X_test.iloc[sample_indices]
        y_sample = self.y_test.iloc[sample_indices]

        # Predicții de la fiecare estimator (folosind toate features-urile din ansamblu)
        # BaggingRegressor/Classifier gestionează intern feature selection
        predictions = np.array([
            model.estimators_[i].predict(X_sample.iloc[:, model.estimators_features_[i]])
            for i in range(len(model.estimators_))
        ])
        ensemble_pred = model.predict(X_sample)

        # Plot scatter cu medie și std pentru fiecare exemplu
        mean_preds = np.mean(predictions, axis=0)
        std_preds = np.std(predictions, axis=0)

        x_pos = np.arange(n_samples)
        axes[0, 0].fill_between(x_pos, mean_preds - std_preds, mean_preds + std_preds,
                                alpha=0.3, color='lightblue', label='±1 std (diversitate)')
        axes[0, 0].plot(x_pos, ensemble_pred, 'r-', linewidth=2,
                       label='Predicție Ansamblu', marker='o', markersize=4)
        axes[0, 0].plot(x_pos, y_sample.values, 'g--', linewidth=2,
                       label='Valoare Reală', marker='s', markersize=4)
        axes[0, 0].set_title(f'Diversitatea Predicțiilor ({n_samples} exemple)',
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Index Exemplu')
        axes[0, 0].set_ylabel('Predicție')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Distribuția predicțiilor pentru un exemplu
        example_idx = 0
        example_predictions = predictions[:, example_idx]
        axes[0, 1].hist(example_predictions, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 1].axvline(ensemble_pred[example_idx], color='red', linestyle='--',
                          linewidth=2, label=f'Ansamblu: {ensemble_pred[example_idx]:.2f}')
        axes[0, 1].axvline(y_sample.values[example_idx], color='green', linestyle='--',
                          linewidth=2, label=f'Real: {y_sample.values[example_idx]:.2f}')
        axes[0, 1].set_title(f'Distribuție Predicții pentru Exemplul #{example_idx}', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Predicție')
        axes[0, 1].set_ylabel('Frecvență Estimatori')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Acuratețe/Eroare pe măsură ce adăugăm estimatori
        scores = []
        for n in range(1, len(model.estimators_) + 1):
            if self.task_type == 'classification':
                # Vot majoritar pentru primii n estimatori
                preds = np.array([
                    model.estimators_[i].predict(self.X_test.iloc[:, model.estimators_features_[i]])
                    for i in range(n)
                ])
                ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, preds)
                score = accuracy_score(self.y_test, ensemble_pred)
            else:
                # Medie pentru primii n estimatori
                preds = np.array([
                    model.estimators_[i].predict(self.X_test.iloc[:, model.estimators_features_[i]])
                    for i in range(n)
                ])
                ensemble_pred = np.mean(preds, axis=0)
                score = -mean_squared_error(self.y_test, ensemble_pred)
            scores.append(score)

        axes[1, 0].plot(range(1, len(scores) + 1), scores, 'b-', linewidth=2, marker='o')
        axes[1, 0].set_title('Performanță vs Număr Estimatori', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Număr Estimatori')
        axes[1, 0].set_ylabel('Accuracy' if self.task_type == 'classification' else 'Negative MSE')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].axhline(scores[-1], color='red', linestyle='--', alpha=0.5,
                          label=f'Final: {scores[-1]:.4f}')
        axes[1, 0].legend()

        # 4. Importanță caracteristici (medie peste toate bag-urile)
        if hasattr(model.estimators_[0], 'feature_importances_'):
            # Calculăm importanța pentru toate features-urile
            all_importances = np.zeros(len(self.X_train.columns))

            for i, estimator in enumerate(model.estimators_):
                # Obținem feature indices pentru acest estimator
                feature_indices = model.estimators_features_[i]
                # Obținem importanțele pentru features-urile selectate
                importances = estimator.feature_importances_
                # Adăugăm la importanța totală
                all_importances[feature_indices] += importances

            # Normalizăm (împărțim la numărul de ori când fiecare feature a fost folosit)
            feature_counts = np.zeros(len(self.X_train.columns))
            for feature_indices in model.estimators_features_:
                feature_counts[feature_indices] += 1

            # Evităm împărțirea la 0
            all_importances = np.divide(all_importances, feature_counts,
                                       where=feature_counts != 0, out=np.zeros_like(all_importances))

            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': all_importances
            }).sort_values('importance', ascending=False).head(15)

            axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue')
            axes[1, 1].set_yticks(range(len(feature_importance)))
            axes[1, 1].set_yticklabels(feature_importance['feature'])
            axes[1, 1].set_title('Top 15 Caracteristici (Medie Bags)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Importanță Medie')
            axes[1, 1].invert_yaxis()
            axes[1, 1].grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('03_bagging_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print("✓ Grafic salvat: 03_bagging_analysis.png")

    def train_boosting(self):
        """PASUL 4: AdaBoost (Adaptive Boosting)"""
        print(f"\n{'='*80}")
        print("🚀 PASUL 4: ADABOOST (Adaptive Boosting)")
        print("-" * 80)
        print("📖 Concept: Antrenează estimatori secvențial, fiecare corectând")
        print("           erorile precedentului prin ajustarea ponderilor exemplelor")

        n_estimators = 50

        if self.task_type == 'classification':
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=n_estimators,
                learning_rate=1.0,
                random_state=42
            )
        else:
            model = AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=3),
                n_estimators=n_estimators,
                learning_rate=1.0,
                random_state=42
            )

        print(f"\n🔄 Antrenare {n_estimators} iterații boosting...")
        model.fit(self.X_train, self.y_train)
        print("✓ Antrenare completă!")

        # Predicții
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        # Evaluare
        self._evaluate_model('AdaBoost', model, y_pred_train, y_pred_test)

        # Vizualizare proces boosting
        self._plot_boosting_analysis(model)

        # Salvare model
        self.models['AdaBoost'] = model

    def _plot_boosting_analysis(self, model):
        """Analiză și vizualizare AdaBoost"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Evoluția erorii pe iterații
        train_errors = []
        test_errors = []

        for i, (estimator, weight) in enumerate(zip(model.estimators_, model.estimator_weights_)):
            # Predicție train
            if self.task_type == 'classification':
                staged_pred_train = model.predict(self.X_train)
                staged_pred_test = model.predict(self.X_test)
                train_errors.append(1 - accuracy_score(self.y_train, staged_pred_train))
                test_errors.append(1 - accuracy_score(self.y_test, staged_pred_test))
            else:
                staged_pred_train = model.predict(self.X_train)
                staged_pred_test = model.predict(self.X_test)
                train_errors.append(mean_squared_error(self.y_train, staged_pred_train))
                test_errors.append(mean_squared_error(self.y_test, staged_pred_test))

        axes[0, 0].plot(range(1, len(train_errors) + 1), train_errors, 'b-',
                       linewidth=2, label='Train', marker='o', markersize=4)
        axes[0, 0].plot(range(1, len(test_errors) + 1), test_errors, 'r-',
                       linewidth=2, label='Test', marker='s', markersize=4)
        axes[0, 0].set_title('Evoluția Erorii pe Iterații Boosting', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Iterație')
        axes[0, 0].set_ylabel('Error Rate' if self.task_type == 'classification' else 'MSE')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. Ponderi estimatori
        axes[0, 1].bar(range(len(model.estimator_weights_)), model.estimator_weights_,
                      color='steelblue', alpha=0.7)
        axes[0, 1].set_title('Ponderi Estimatori în Ansamblu', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Index Estimator')
        axes[0, 1].set_ylabel('Pondere')
        axes[0, 1].grid(alpha=0.3, axis='y')

        # 3. Performanță cumulativă
        cumulative_scores = []
        for n in range(1, len(model.estimators_) + 1):
            if self.task_type == 'classification':
                # Predicție weighted voting
                predictions = np.array([est.predict(self.X_test) for est in model.estimators_[:n]])
                weights = model.estimator_weights_[:n]
                weighted_pred = np.average(predictions, axis=0, weights=weights)
                final_pred = (weighted_pred > 0.5).astype(int) if len(np.unique(self.y_test)) == 2 else weighted_pred.round()
                score = accuracy_score(self.y_test, final_pred)
            else:
                predictions = np.array([est.predict(self.X_test) for est in model.estimators_[:n]])
                weights = model.estimator_weights_[:n]
                weighted_pred = np.average(predictions, axis=0, weights=weights)
                score = r2_score(self.y_test, weighted_pred)
            cumulative_scores.append(score)

        axes[1, 0].plot(range(1, len(cumulative_scores) + 1), cumulative_scores,
                       'g-', linewidth=2, marker='o', markersize=4)
        axes[1, 0].set_title('Performanță Cumulativă (Test Set)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Număr Estimatori')
        axes[1, 0].set_ylabel('Accuracy' if self.task_type == 'classification' else 'R² Score')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].axhline(cumulative_scores[-1], color='red', linestyle='--', alpha=0.5,
                          label=f'Final: {cumulative_scores[-1]:.4f}')
        axes[1, 0].legend()

        # 4. Importanță caracteristici
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'],
                           color='coral')
            axes[1, 1].set_yticks(range(len(feature_importance)))
            axes[1, 1].set_yticklabels(feature_importance['feature'])
            axes[1, 1].set_title('Top 15 Caracteristici Importante', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Importanță')
            axes[1, 1].invert_yaxis()
            axes[1, 1].grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('04_boosting_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print("✓ Grafic salvat: 04_boosting_analysis.png")

    def train_random_forest(self):
        """PASUL 5: Random Forest"""
        print(f"\n{'='*80}")
        print("🌲 PASUL 5: RANDOM FOREST")
        print("-" * 80)
        print("📖 Concept: Bagging + selecție aleatoare caracteristici la fiecare split")
        print("           pentru diversitate maximă între arbori")

        n_estimators = 100

        if self.task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )

        print(f"\n🔄 Antrenare {n_estimators} arbori...")
        model.fit(self.X_train, self.y_train)
        print("✓ Antrenare completă!")
        print(f"  OOB Score: {model.oob_score_:.4f}")

        # Predicții
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        # Evaluare
        self._evaluate_model('Random Forest', model, y_pred_train, y_pred_test)

        # Vizualizare analiză
        self._plot_random_forest_analysis(model)

        # Salvare model
        self.models['Random Forest'] = model

    def _plot_random_forest_analysis(self, model):
        """Analiză și vizualizare Random Forest"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Importanță caracteristici cu incertitudine
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        # Calculare incertitudine (std între arbori)
        importances_per_tree = np.array([tree.feature_importances_ for tree in model.estimators_])
        feature_std = np.std(importances_per_tree, axis=0)
        importance_std = [feature_std[list(self.X_train.columns).index(feat)]
                         for feat in feature_importance['feature']]

        ax1 = fig.add_subplot(gs[0, :2])
        bars = ax1.barh(range(len(feature_importance)), feature_importance['importance'],
                       xerr=importance_std, color='forestgreen', alpha=0.7, capsize=3)
        ax1.set_yticks(range(len(feature_importance)))
        ax1.set_yticklabels(feature_importance['feature'])
        ax1.set_title('Top 20 Caracteristici Importante (cu Incertitudine)',
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('Importanță (Gini)')
        ax1.invert_yaxis()
        ax1.grid(alpha=0.3, axis='x')

        # 2. Distribuție profunzime arbori
        ax2 = fig.add_subplot(gs[0, 2])
        tree_depths = [tree.get_depth() for tree in model.estimators_]
        ax2.hist(tree_depths, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(tree_depths), color='red', linestyle='--', linewidth=2,
                   label=f'Medie: {np.mean(tree_depths):.1f}')
        ax2.set_title('Distribuție Profunzime Arbori', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Profunzime')
        ax2.set_ylabel('Frecvență')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Performanță vs număr arbori
        ax3 = fig.add_subplot(gs[1, 0])
        scores = []
        for n in range(10, len(model.estimators_) + 1, 5):
            if self.task_type == 'classification':
                preds = np.array([est.predict(self.X_test) for est in model.estimators_[:n]])
                ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, preds)
                score = accuracy_score(self.y_test, ensemble_pred)
            else:
                preds = np.array([est.predict(self.X_test) for est in model.estimators_[:n]])
                ensemble_pred = np.mean(preds, axis=0)
                score = r2_score(self.y_test, ensemble_pred)
            scores.append((n, score))

        scores = np.array(scores)
        ax3.plot(scores[:, 0], scores[:, 1], 'b-', linewidth=2, marker='o')
        ax3.set_title('Convergență Performanță', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Număr Arbori')
        ax3.set_ylabel('Accuracy' if self.task_type == 'classification' else 'R² Score')
        ax3.grid(alpha=0.3)
        ax3.axhline(scores[-1, 1], color='red', linestyle='--', alpha=0.5)

        # 4. Distribuție număr noduri frunză
        ax4 = fig.add_subplot(gs[1, 1])
        n_leaves = [tree.get_n_leaves() for tree in model.estimators_]
        ax4.hist(n_leaves, bins=20, color='coral', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(n_leaves), color='red', linestyle='--', linewidth=2,
                   label=f'Medie: {np.mean(n_leaves):.1f}')
        ax4.set_title('Distribuție Noduri Frunză', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Număr Noduri Frunză')
        ax4.set_ylabel('Frecvență')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. OOB vs Test Score pe parcurs
        ax5 = fig.add_subplot(gs[1, 2])
        oob_scores = []
        test_scores_rf = []
        for n in range(10, len(model.estimators_) + 1, 5):
            if self.task_type == 'classification':
                temp_model = RandomForestClassifier(
                    n_estimators=n, max_depth=10, oob_score=True,
                    random_state=42, warm_start=False
                )
            else:
                temp_model = RandomForestRegressor(
                    n_estimators=n, max_depth=10, oob_score=True,
                    random_state=42, warm_start=False
                )
            temp_model.fit(self.X_train, self.y_train)
            oob_scores.append(temp_model.oob_score_)

            pred = temp_model.predict(self.X_test)
            if self.task_type == 'classification':
                test_scores_rf.append(accuracy_score(self.y_test, pred))
            else:
                test_scores_rf.append(r2_score(self.y_test, pred))

        x_range = range(10, len(model.estimators_) + 1, 5)
        ax5.plot(x_range, oob_scores, 'b-', linewidth=2, marker='o', label='OOB Score')
        ax5.plot(x_range, test_scores_rf, 'r-', linewidth=2, marker='s', label='Test Score')
        ax5.set_title('OOB vs Test Score', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Număr Arbori')
        ax5.set_ylabel('Score')
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. Diversitate predicții (pentru câteva exemple)
        ax6 = fig.add_subplot(gs[2, :])
        n_samples_show = 30
        sample_idx = np.random.choice(len(self.X_test), n_samples_show, replace=False)
        X_sample = self.X_test.iloc[sample_idx]
        y_sample = self.y_test.iloc[sample_idx]

        # Predicții individuale
        individual_preds = np.array([tree.predict(X_sample) for tree in model.estimators_[:20]])
        ensemble_pred = model.predict(X_sample)

        # Box plot pentru fiecare exemplu
        positions = range(n_samples_show)
        bp = ax6.boxplot([individual_preds[:, i] for i in range(n_samples_show)],
                         positions=positions, widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='blue', linewidth=2))

        ax6.plot(positions, ensemble_pred, 'ro-', linewidth=2, markersize=6,
                label='Predicție RF', alpha=0.8)
        ax6.plot(positions, y_sample.values, 'g^--', linewidth=2, markersize=6,
                label='Valoare Reală', alpha=0.8)
        ax6.set_title('Diversitate Predicții RF (20 arbori, 30 exemple)',
                     fontsize=12, fontweight='bold')
        ax6.set_xlabel('Index Exemplu')
        ax6.set_ylabel('Predicție')
        ax6.legend(loc='best')
        ax6.grid(alpha=0.3, axis='y')

        plt.savefig('05_random_forest_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print("✓ Grafic salvat: 05_random_forest_analysis.png")

    def train_neural_network(self):
        """PASUL 6: Neural Network (Black-box model)"""
        print(f"\n{'='*80}")
        print("🧠 PASUL 6: NEURAL NETWORK (Black-box Model)")
        print("-" * 80)
        print("📖 Concept: Rețea neuronală cu straturi ascunse pentru detectarea")
        print("           pattern-urilor complexe și non-lineare")

        if self.task_type == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False
            )

        print(f"\n🔄 Antrenare Neural Network...")
        print(f"  Arhitectură: Input → 100 → 50 → 25 → Output")
        print(f"  Activare: ReLU, Optimizer: Adam")

        model.fit(self.X_train_scaled, self.y_train)

        print(f"✓ Antrenare completă!")
        print(f"  Iterații: {model.n_iter_}")
        print(f"  Loss final: {model.loss_:.6f}")

        # Predicții
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)

        # Evaluare
        self._evaluate_model('Neural Network', model, y_pred_train, y_pred_test)

        # Vizualizare analiză
        self._plot_neural_network_analysis(model)

        # Salvare model
        self.models['Neural Network'] = model

    def _plot_neural_network_analysis(self, model):
        """Analiză și vizualizare Neural Network"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Evoluția loss-ului
        axes[0, 0].plot(model.loss_curve_, 'b-', linewidth=2, label='Training Loss')
        if hasattr(model, 'validation_scores_'):
            axes[0, 0].plot(model.validation_scores_, 'r-', linewidth=2, label='Validation Score')
        axes[0, 0].set_title('Evoluția Loss-ului în Timpul Antrenării',
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Iterație (Epoch)')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_yscale('log')

        # 2. Distribuția ponderilor în straturi
        layer_names = []
        layer_weights = []
        for i, coef in enumerate(model.coefs_):
            layer_names.append(f'Layer {i}-{i+1}')
            layer_weights.append(coef.flatten())

        axes[0, 1].violinplot(layer_weights, positions=range(len(layer_weights)),
                             showmeans=True, showmedians=True)
        axes[0, 1].set_xticks(range(len(layer_names)))
        axes[0, 1].set_xticklabels(layer_names, rotation=45)
        axes[0, 1].set_title('Distribuția Ponderilor pe Straturi',
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Valoare Pondere')
        axes[0, 1].grid(alpha=0.3, axis='y')
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)

        # 3. Predicții vs Valori Reale
        y_pred = model.predict(self.X_test_scaled)

        if self.task_type == 'classification':
            # Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.y_test, y_pred)

            # Normalizare
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            im = axes[1, 0].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            axes[1, 0].set_title('Matrice de Confuzie (Normalizată)',
                                fontsize=12, fontweight='bold')

            # Adăugare valori în celule
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1, 0].text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                                   ha="center", va="center",
                                   color="white" if cm_normalized[i, j] > thresh else "black")

            axes[1, 0].set_ylabel('Clasă Reală')
            axes[1, 0].set_xlabel('Clasă Prezisă')
            fig.colorbar(im, ax=axes[1, 0])
        else:
            # Scatter plot pentru regresie
            axes[1, 0].scatter(self.y_test, y_pred, alpha=0.5, s=30, color='steelblue')

            # Linie ideală
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val],
                           'r--', linewidth=2, label='Predicție Perfectă')

            axes[1, 0].set_title('Predicții vs Valori Reale (Test Set)',
                                fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Valoare Reală')
            axes[1, 0].set_ylabel('Valoare Prezisă')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        # 4. Distribuția erorilor de predicție
        if self.task_type == 'regression':
            errors = self.y_test - y_pred
            axes[1, 1].hist(errors, bins=30, color='coral', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2,
                              label=f'Zero Error')
            axes[1, 1].axvline(errors.mean(), color='blue', linestyle='--', linewidth=2,
                              label=f'Medie: {errors.mean():.2f}')
            axes[1, 1].set_title('Distribuția Erorilor de Predicție',
                                fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Error (Real - Prezis)')
            axes[1, 1].set_ylabel('Frecvență')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        else:
            # Pentru clasificare: confidence distribution
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(self.X_test_scaled)
                max_probas = np.max(probas, axis=1)

                axes[1, 1].hist(max_probas, bins=30, color='green', alpha=0.7, edgecolor='black')
                axes[1, 1].axvline(max_probas.mean(), color='red', linestyle='--',
                                  linewidth=2, label=f'Medie: {max_probas.mean():.3f}')
                axes[1, 1].set_title('Distribuția Confidence Scores',
                                    fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Max Probability')
                axes[1, 1].set_ylabel('Frecvență')
                axes[1, 1].legend()
                axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('06_neural_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print("✓ Grafic salvat: 06_neural_network_analysis.png")

    def _evaluate_model(self, model_name, model, y_pred_train, y_pred_test):
        """Evaluare model și salvare rezultate"""
        print(f"\n📊 EVALUARE {model_name.upper()}")
        print("-" * 60)

        if self.task_type == 'classification':
            train_acc = accuracy_score(self.y_train, y_pred_train)
            test_acc = accuracy_score(self.y_test, y_pred_test)

            print(f"  Accuracy Train: {train_acc:.4f}")
            print(f"  Accuracy Test:  {test_acc:.4f}")
            print(f"  Overfitting:    {(train_acc - test_acc):.4f}")

            # Cross-validation
            if model_name != 'Neural Network':  # NN necesită date scaled
                cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                           cv=5, scoring='accuracy')
                print(f"  CV Score (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            print(f"\n  Classification Report (Test Set):")
            print(classification_report(self.y_test, y_pred_test))

            self.results[model_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'overfitting': train_acc - test_acc
            }
        else:
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)

            print(f"  MSE Train:  {train_mse:.4f}")
            print(f"  MSE Test:   {test_mse:.4f}")
            print(f"  RMSE Test:  {np.sqrt(test_mse):.4f}")
            print(f"  MAE Test:   {test_mae:.4f}")
            print(f"  R² Train:   {train_r2:.4f}")
            print(f"  R² Test:    {test_r2:.4f}")
            print(f"  Overfitting (R²): {(train_r2 - test_r2):.4f}")

            # Cross-validation
            if model_name != 'Neural Network':
                cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                           cv=5, scoring='r2')
                print(f"  CV R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            self.results[model_name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'overfitting': train_r2 - test_r2
            }

    def compare_all_models(self):
        """Comparație finală între toate modelele"""
        print(f"\n{'='*80}")
        print("📊 COMPARAȚIE FINALĂ - TOATE MODELELE")
        print("=" * 80)

        # Creare DataFrame cu rezultate
        if self.task_type == 'classification':
            comparison_df = pd.DataFrame({
                'Model': list(self.results.keys()),
                'Train Accuracy': [self.results[m]['train_accuracy'] for m in self.results],
                'Test Accuracy': [self.results[m]['test_accuracy'] for m in self.results],
                'Overfitting': [self.results[m]['overfitting'] for m in self.results]
            })

            print("\n" + comparison_df.to_string(index=False))

            # Identificare cel mai bun model
            best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
            best_score = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Test Accuracy']
            print(f"\n🏆 CEL MAI BUN MODEL: {best_model}")
            print(f"   Test Accuracy: {best_score:.4f}")

        else:
            comparison_df = pd.DataFrame({
                'Model': list(self.results.keys()),
                'Train R²': [self.results[m]['train_r2'] for m in self.results],
                'Test R²': [self.results[m]['test_r2'] for m in self.results],
                'Test MSE': [self.results[m]['test_mse'] for m in self.results],
                'Test MAE': [self.results[m]['test_mae'] for m in self.results],
                'Overfitting': [self.results[m]['overfitting'] for m in self.results]
            })

            print("\n" + comparison_df.to_string(index=False))

            # Identificare cel mai bun model
            best_model = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Model']
            best_score = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Test R²']
            print(f"\n🏆 CEL MAI BUN MODEL: {best_model}")
            print(f"   Test R²: {best_score:.4f}")

        # Vizualizare comparație
        self._plot_model_comparison(comparison_df)

        # Salvare rezultate în CSV
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print("\n✓ Rezultate salvate: model_comparison_results.csv")

    def _plot_model_comparison(self, comparison_df):
        """Vizualizare comparație modele"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        models = comparison_df['Model'].values
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        if self.task_type == 'classification':
            # 1. Accuracy Comparison
            x = np.arange(len(models))
            width = 0.35

            axes[0, 0].bar(x - width/2, comparison_df['Train Accuracy'], width,
                          label='Train', color='steelblue', alpha=0.8)
            axes[0, 0].bar(x + width/2, comparison_df['Test Accuracy'], width,
                          label='Test', color='coral', alpha=0.8)
            axes[0, 0].set_xlabel('Model')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Comparație Accuracy: Train vs Test',
                                fontsize=13, fontweight='bold')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3, axis='y')

            # 2. Test Accuracy Ranking
            sorted_df = comparison_df.sort_values('Test Accuracy', ascending=True)
            axes[0, 1].barh(range(len(sorted_df)), sorted_df['Test Accuracy'], color=colors)
            axes[0, 1].set_yticks(range(len(sorted_df)))
            axes[0, 1].set_yticklabels(sorted_df['Model'])
            axes[0, 1].set_xlabel('Test Accuracy')
            axes[0, 1].set_title('Ranking Modele după Test Accuracy',
                                fontsize=13, fontweight='bold')
            axes[0, 1].grid(alpha=0.3, axis='x')

            # Adăugare valori pe bare
            for i, v in enumerate(sorted_df['Test Accuracy']):
                axes[0, 1].text(v + 0.005, i, f'{v:.4f}', va='center')

        else:
            # 1. R² Comparison
            x = np.arange(len(models))
            width = 0.35

            axes[0, 0].bar(x - width/2, comparison_df['Train R²'], width,
                          label='Train', color='steelblue', alpha=0.8)
            axes[0, 0].bar(x + width/2, comparison_df['Test R²'], width,
                          label='Test', color='coral', alpha=0.8)
            axes[0, 0].set_xlabel('Model')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_title('Comparație R² Score: Train vs Test',
                                fontsize=13, fontweight='bold')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3, axis='y')

            # 2. Test R² Ranking
            sorted_df = comparison_df.sort_values('Test R²', ascending=True)
            axes[0, 1].barh(range(len(sorted_df)), sorted_df['Test R²'], color=colors)
            axes[0, 1].set_yticks(range(len(sorted_df)))
            axes[0, 1].set_yticklabels(sorted_df['Model'])
            axes[0, 1].set_xlabel('Test R²')
            axes[0, 1].set_title('Ranking Modele după Test R²',
                                fontsize=13, fontweight='bold')
            axes[0, 1].grid(alpha=0.3, axis='x')

            for i, v in enumerate(sorted_df['Test R²']):
                axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center')

        # 3. Overfitting Comparison
        axes[1, 0].bar(models, comparison_df['Overfitting'], color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Overfitting (Train - Test)')
        axes[1, 0].set_title('Comparație Overfitting', fontsize=13, fontweight='bold')
        axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].grid(alpha=0.3, axis='y')

        # 4. Overall Performance Heatmap
        if self.task_type == 'classification':
            heatmap_data = comparison_df[['Train Accuracy', 'Test Accuracy']].T
        else:
            heatmap_data = comparison_df[['Train R²', 'Test R²', 'Test MSE']].T

        heatmap_data.columns = models

        im = axes[1, 1].imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_xticks(np.arange(len(models)))
        axes[1, 1].set_yticks(np.arange(len(heatmap_data.index)))
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(heatmap_data.index)
        axes[1, 1].set_title('Heatmap Performanță', fontsize=13, fontweight='bold')

        # Adăugare valori în celule
        for i in range(len(heatmap_data.index)):
            for j in range(len(models)):
                text = axes[1, 1].text(j, i, f'{heatmap_data.iloc[i, j]:.3f}',
                                      ha="center", va="center", color="black", fontsize=10)

        fig.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig('07_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print("\n✓ Grafic salvat: 07_model_comparison.png")

    def run_complete_analysis(self):
        """Rulare analiză completă"""
        self.load_and_preprocess()
        self.train_decision_tree()
        self.train_bagging()
        self.train_boosting()
        self.train_random_forest()
        self.train_neural_network()
        self.compare_all_models()

        print(f"\n{'='*80}")
        print("✅ ANALIZĂ COMPLETĂ FINALIZATĂ!")
        print("=" * 80)
        print("\n📁 Fișiere generate:")
        print("  - 01_target_distribution.png")
        print("  - 02_decision_tree_structure.png")
        print("  - 03_bagging_analysis.png")
        print("  - 04_boosting_analysis.png")
        print("  - 05_random_forest_analysis.png")
        print("  - 06_neural_network_analysis.png")
        print("  - 07_model_comparison.png")
        print("  - model_comparison_results.csv")
        print("\n🎓 Analiză educativă completă cu vizualizări pas cu pas!")


# ============================================================================
# EXEMPLU DE UTILIZARE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" METODE ANSAMBLU - ANALIZĂ COMPLETĂ ")
    print("="*80 + "\n")

    print("Selectați setul de date:")
    print("1. Car Price (Regresie)")
    print("2. Customer Purchasing Behavior (Regresie/Clasificare)")

    choice = input("\nAlegeți (1 sau 2): ").strip()

    if choice == '1':
        # ANALIZĂ CAR PRICE (REGRESIE)
        print("\n🚗 Analiză Car Price Dataset (Regresie)")
        print("-" * 80)

        analyzer = EnsembleAnalysis(
            data_path='CarPrice.csv',
            target_column='price',
            task_type='regression'
        )
        analyzer.run_complete_analysis()

    elif choice == '2':
        # ANALIZĂ CUSTOMER PURCHASING BEHAVIOR
        print("\n👥 Analiză Customer Purchasing Behavior Dataset")
        print("-" * 80)
        print("\nSelectați tipul de analiză:")
        print("a. Regresie - Predicție purchase_amount")
        print("b. Clasificare - Predicție loyalty_score categorii (Low/Medium/High)")

        sub_choice = input("\nAlegeți (a sau b): ").strip().lower()

        if sub_choice == 'a':
            # Regresie
            analyzer = EnsembleAnalysis(
                data_path='Customer Purchasing Behaviors.csv',
                target_column='purchase_amount',
                task_type='regression'
            )
            analyzer.run_complete_analysis()

        elif sub_choice == 'b':
            # Clasificare - trebuie să creăm categorii din loyalty_score
            print("\n📊 Preprocesare: Creare categorii loyalty...")
            df = pd.read_csv('Customer Purchasing Behaviors.csv')

            # Creare categorii loyalty
            df['loyalty_category'] = pd.cut(df['loyalty_score'],
                                           bins=[0, 4, 7, 10],
                                           labels=['Low', 'Medium', 'High'])

            # Salvare fișier temporar
            df.to_csv('customer_classification_temp.csv', index=False)
            print("✓ Categorii create: Low (0-4), Medium (4-7), High (7-10)")

            analyzer = EnsembleAnalysis(
                data_path='customer_classification_temp.csv',
                target_column='loyalty_category',
                task_type='classification'
            )
            analyzer.run_complete_analysis()

        else:
            print("❌ Opțiune invalidă!")
    else:
        print("❌ Opțiune invalidă!")

    print("\n" + "="*80)
    print(" ANALIZĂ FINALIZATĂ CU SUCCES! ")
    print("="*80)
    print("\n💡 CONCLUZII EDUCATIVE:")
    print("-" * 80)
    print("""
🌳 DECISION TREE (Baseline):
   - Model simplu și interpretabil
   - Tendință de overfitting pe date complexe
   - Util pentru înțelegerea structurii datelor

🎒 BAGGING:
   - Reduce variația prin combinarea mai multor arbori
   - Fiecare arbore antrenat pe subset diferit (bootstrap)
   - OOB score oferă estimare generalizare fără set validare separat
   - Îmbunătățește stabilitatea față de un singur arbore

🚀 ADABOOST:
   - Antrenare secvențială - fiecare model corectează erorile precedentului
   - Ajustare ponderi exemple greșit clasificate
   - Performanță crescută prin focus pe exemple dificile
   - Sensibil la zgomot și outliers

🌲 RANDOM FOREST:
   - Combină bagging cu selecție aleatoare de features
   - Diversitate maximă între arbori
   - Foarte robust și performant
   - Feature importance pentru interpretabilitate
   - Adesea cel mai bun model "out-of-the-box"

🧠 NEURAL NETWORK:
   - Model black-box cu capacitate mare de învățare
   - Detectează pattern-uri non-lineare complexe
   - Necesită scaling și mai multe date
   - Risc de overfitting dacă nu e regularizat corespunzător
   - Performanță variabilă în funcție de arhitectură și hiperparametri

📊 BEST PRACTICES:
   1. Încercați întotdeauna mai multe modele
   2. Random Forest - excelent punct de pornire
   3. Folosiți cross-validation pentru estimare robustă
   4. Monitorizați overfitting-ul (train vs test)
   5. Feature importance pentru înțelegere business
   6. Ensemble methods > single models (de obicei)
    """)

    print("\n🎯 RECOMANDĂRI URMĂTORII PAȘI:")
    print("-" * 80)
    print("""
1. 📚 STUDIU APROFUNDAT:
   - Experimentați cu hiperparametri (GridSearchCV, RandomizedSearchCV)
   - Încercați XGBoost/LightGBM pentru gradient boosting avansat
   - Stacking - combinare modele diferite în ansamblu

2. 🔍 FEATURE ENGINEERING:
   - Creați features noi din cele existente
   - Analizați interacțiuni între features
   - Selectare features optimă

3. 📊 VALIDARE EXTINSĂ:
   - Stratified K-Fold pentru date nebalansate
   - Time series split pentru date temporale
   - Analiză erori pentru înțelegere pattern-uri

4. 🚀 DEPLOYMENT:
   - Salvare model final (pickle/joblib)
   - Monitorizare performanță în producție
   - Reantrenare periodicăA

5. 🎓 EXPLICABILITATE:
   - SHAP values pentru interpretare predicții
   - LIME pentru explicații locale
   - Partial Dependence Plots
    """)

    print("\n✨ Mult succes în continuarea proiectului!")
    print("="*80 + "\n")
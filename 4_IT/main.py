import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Setare stil pentru grafice
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# FUNCȚIE PENTRU ANTRENARE ȘI EVALUARE MODELE
# ============================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, dataset_name):
    """
    Antrenează toate cele 5 modele și returnează rezultatele
    """
    print(f"\n{'=' * 80}")
    print(f"ANTRENARE MODELE PENTRU: {dataset_name}")
    print(f"{'=' * 80}")

    results = {}
    predictions = {}

    # 1. DECISION TREE
    print("\n1. Decision Tree...")
    dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    results['Decision Tree'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        'MAE': mean_absolute_error(y_test, y_pred_dt),
        'R²': r2_score(y_test, y_pred_dt),
        'model': dt_model
    }
    predictions['Decision Tree'] = y_pred_dt

    # 2. BAGGING
    print("2. Bagging...")
    bagging_model = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=5),
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
        oob_score=True
    )
    bagging_model.fit(X_train, y_train)
    y_pred_bag = bagging_model.predict(X_test)

    results['Bagging'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_bag)),
        'MAE': mean_absolute_error(y_test, y_pred_bag),
        'R²': r2_score(y_test, y_pred_bag),
        'OOB': bagging_model.oob_score_,
        'model': bagging_model
    }
    predictions['Bagging'] = y_pred_bag

    # 3. BOOSTING (AdaBoost)
    print("3. Boosting (AdaBoost)...")
    boosting_model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=50,
        learning_rate=0.1,
        random_state=42
    )
    boosting_model.fit(X_train, y_train)
    y_pred_boost = boosting_model.predict(X_test)

    results['Boosting'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_boost)),
        'MAE': mean_absolute_error(y_test, y_pred_boost),
        'R²': r2_score(y_test, y_pred_boost),
        'model': boosting_model
    }
    predictions['Boosting'] = y_pred_boost

    # 4. RANDOM FOREST
    print("4. Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42,
        oob_score=True
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    results['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'R²': r2_score(y_test, y_pred_rf),
        'OOB': rf_model.oob_score_,
        'model': rf_model
    }
    predictions['Random Forest'] = y_pred_rf

    # 5. NEURAL NETWORK
    print("5. Neural Network...")
    nn_model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_model.fit(X_train_scaled, y_train)
    y_pred_nn = nn_model.predict(X_test_scaled)

    results['Neural Network'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
        'MAE': mean_absolute_error(y_test, y_pred_nn),
        'R²': r2_score(y_test, y_pred_nn),
        'model': nn_model
    }
    predictions['Neural Network'] = y_pred_nn

    return results, predictions


# ============================================================================
# DATASET 1: CAR PRICE
# ============================================================================

print("\n" + "=" * 80)
print("DATASET 1: CAR PRICE - Predicția Prețului Mașinilor")
print("=" * 80)

# Încărcare date
df_car = pd.read_csv('CarPrice.csv')
print(f"\n✓ Date încărcate: {df_car.shape[0]} mașini, {df_car.shape[1]} coloane")

# Preprocesare
df_car_processed = df_car.drop(['car_ID', 'CarName'], axis=1)

# Codificare variabile categoriale
categorical_cols_car = ['fueltype', 'aspiration', 'doornumber', 'carbody',
                        'drivewheel', 'enginelocation', 'enginetype',
                        'cylindernumber', 'fuelsystem']

for col in categorical_cols_car:
    le = LabelEncoder()
    df_car_processed[col] = le.fit_transform(df_car_processed[col])

# Separare features și target
X_car = df_car_processed.drop('price', axis=1)
y_car = df_car_processed['price']

# Split train/test
X_car_train, X_car_test, y_car_train, y_car_test = train_test_split(
    X_car, y_car, test_size=0.2, random_state=42
)

# Scalare pentru NN
scaler_car = StandardScaler()
X_car_train_scaled = scaler_car.fit_transform(X_car_train)
X_car_test_scaled = scaler_car.transform(X_car_test)

print(f"✓ Train: {X_car_train.shape[0]} | Test: {X_car_test.shape[0]}")

# Antrenare modele
results_car, predictions_car = train_and_evaluate_models(
    X_car_train, X_car_test, y_car_train, y_car_test,
    X_car_train_scaled, X_car_test_scaled,
    "Car Price"
)

# ============================================================================
# DATASET 2: CUSTOMER PURCHASING BEHAVIORS
# ============================================================================

print("\n" + "=" * 80)
print("DATASET 2: CUSTOMER PURCHASING BEHAVIORS - Predicția Sumei Achiziționate")
print("=" * 80)

# Încărcare date
df_customer = pd.read_csv('Customer Purchasing Behaviors.csv')
print(f"\n✓ Date încărcate: {df_customer.shape[0]} clienți, {df_customer.shape[1]} coloane")

# Preprocesare
df_customer_processed = df_customer.drop(['user_id'], axis=1)

# Codificare variabilă categorială (region)
le_region = LabelEncoder()
df_customer_processed['region'] = le_region.fit_transform(df_customer_processed['region'])

# Separare features și target (vom prezice purchase_amount)
X_customer = df_customer_processed.drop('purchase_amount', axis=1)
y_customer = df_customer_processed['purchase_amount']

# Split train/test
X_customer_train, X_customer_test, y_customer_train, y_customer_test = train_test_split(
    X_customer, y_customer, test_size=0.2, random_state=42
)

# Scalare pentru NN
scaler_customer = StandardScaler()
X_customer_train_scaled = scaler_customer.fit_transform(X_customer_train)
X_customer_test_scaled = scaler_customer.transform(X_customer_test)

print(f"✓ Train: {X_customer_train.shape[0]} | Test: {X_customer_test.shape[0]}")

# Antrenare modele
results_customer, predictions_customer = train_and_evaluate_models(
    X_customer_train, X_customer_test, y_customer_train, y_customer_test,
    X_customer_train_scaled, X_customer_test_scaled,
    "Customer Purchasing"
)

# ============================================================================
# COMPARAȚIE REZULTATE - AMBELE DATASET-URI
# ============================================================================

print("\n" + "=" * 80)
print("REZULTATE FINALE - COMPARAȚIE AMBELE DATASET-URI")
print("=" * 80)

# Tabel comparativ Car Price
print("\n📊 DATASET 1: CAR PRICE")
print("-" * 80)
df_results_car = pd.DataFrame({
    'Model': list(results_car.keys()),
    'RMSE': [results_car[m]['RMSE'] for m in results_car.keys()],
    'MAE': [results_car[m]['MAE'] for m in results_car.keys()],
    'R²': [results_car[m]['R²'] for m in results_car.keys()]
})
print(df_results_car.to_string(index=False))

best_car = df_results_car.loc[df_results_car['R²'].idxmax()]
print(f"\n🏆 Cel mai bun model: {best_car['Model']} (R² = {best_car['R²']:.4f})")

# Tabel comparativ Customer
print("\n📊 DATASET 2: CUSTOMER PURCHASING BEHAVIORS")
print("-" * 80)
df_results_customer = pd.DataFrame({
    'Model': list(results_customer.keys()),
    'RMSE': [results_customer[m]['RMSE'] for m in results_customer.keys()],
    'MAE': [results_customer[m]['MAE'] for m in results_customer.keys()],
    'R²': [results_customer[m]['R²'] for m in results_customer.keys()]
})
print(df_results_customer.to_string(index=False))

best_customer = df_results_customer.loc[df_results_customer['R²'].idxmax()]
print(f"\n🏆 Cel mai bun model: {best_customer['Model']} (R² = {best_customer['R²']:.4f})")

# ============================================================================
# VIZUALIZĂRI COMPARATIVE
# ============================================================================

print("\n" + "=" * 80)
print("GENERARE VIZUALIZĂRI COMPARATIVE")
print("=" * 80)

# 1. DECISION TREES - Ambele Dataset-uri
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Car Price Tree
plot_tree(results_car['Decision Tree']['model'],
          feature_names=X_car.columns,
          filled=True, rounded=True,
          fontsize=8, max_depth=3, ax=axes[0])
axes[0].set_title('Decision Tree - Car Price\n(R² = {:.4f})'.format(
    results_car['Decision Tree']['R²']), fontsize=14, fontweight='bold', pad=20)

# Customer Tree
plot_tree(results_customer['Decision Tree']['model'],
          feature_names=X_customer.columns,
          filled=True, rounded=True,
          fontsize=8, max_depth=3, ax=axes[1])
axes[1].set_title('Decision Tree - Customer Purchasing\n(R² = {:.4f})'.format(
    results_customer['Decision Tree']['R²']), fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('decision_trees_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Arborii de decizie salvați în 'decision_trees_comparison.png'")
plt.show()

# 2. COMPARAȚIE METRICI - Ambele Dataset-uri
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

datasets = [
    ('Car Price', df_results_car),
    ('Customer Purchasing', df_results_customer)
]

for row, (name, df) in enumerate(datasets):
    # RMSE
    axes[row, 0].bar(df['Model'], df['RMSE'], color='steelblue', alpha=0.7)
    axes[row, 0].set_ylabel('RMSE', fontsize=11)
    axes[row, 0].set_title(f'{name} - RMSE', fontsize=12, fontweight='bold')
    axes[row, 0].tick_params(axis='x', rotation=45)
    axes[row, 0].grid(axis='y', alpha=0.3)

    # MAE
    axes[row, 1].bar(df['Model'], df['MAE'], color='coral', alpha=0.7)
    axes[row, 1].set_ylabel('MAE', fontsize=11)
    axes[row, 1].set_title(f'{name} - MAE', fontsize=12, fontweight='bold')
    axes[row, 1].tick_params(axis='x', rotation=45)
    axes[row, 1].grid(axis='y', alpha=0.3)

    # R²
    axes[row, 2].bar(df['Model'], df['R²'], color='seagreen', alpha=0.7)
    axes[row, 2].set_ylabel('R²', fontsize=11)
    axes[row, 2].set_title(f'{name} - R²', fontsize=12, fontweight='bold')
    axes[row, 2].tick_params(axis='x', rotation=45)
    axes[row, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_comparison_both_datasets.png', dpi=300, bbox_inches='tight')
print("✓ Comparația metricilor salvată în 'metrics_comparison_both_datasets.png'")
plt.show()

# 3. PREDICȚII VS VALORI REALE - Car Price
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (name, y_pred) in enumerate(predictions_car.items()):
    r2 = results_car[name]['R²']
    axes[idx].scatter(y_car_test, y_pred, alpha=0.6, s=50, color='steelblue')
    axes[idx].plot([y_car_test.min(), y_car_test.max()],
                   [y_car_test.min(), y_car_test.max()],
                   'r--', lw=2, label='Perfect prediction')
    axes[idx].set_xlabel('Valori Reale', fontsize=11)
    axes[idx].set_ylabel('Predicții', fontsize=11)
    axes[idx].set_title(f'{name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

fig.delaxes(axes[5])
fig.suptitle('Car Price - Predicții vs Valori Reale', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('predictions_car_price.png', dpi=300, bbox_inches='tight')
print("✓ Predicții Car Price salvate în 'predictions_car_price.png'")
plt.show()

# 4. PREDICȚII VS VALORI REALE - Customer Purchasing
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (name, y_pred) in enumerate(predictions_customer.items()):
    r2 = results_customer[name]['R²']
    axes[idx].scatter(y_customer_test, y_pred, alpha=0.6, s=50, color='coral')
    axes[idx].plot([y_customer_test.min(), y_customer_test.max()],
                   [y_customer_test.min(), y_customer_test.max()],
                   'r--', lw=2, label='Perfect prediction')
    axes[idx].set_xlabel('Valori Reale', fontsize=11)
    axes[idx].set_ylabel('Predicții', fontsize=11)
    axes[idx].set_title(f'{name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

fig.delaxes(axes[5])
fig.suptitle('Customer Purchasing - Predicții vs Valori Reale', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('predictions_customer_purchasing.png', dpi=300, bbox_inches='tight')
print("✓ Predicții Customer Purchasing salvate în 'predictions_customer_purchasing.png'")
plt.show()

# 5. IMPORTANȚA CARACTERISTICILOR - Random Forest
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Car Price
feature_importance_car = pd.DataFrame({
    'feature': X_car.columns,
    'importance': results_car['Random Forest']['model'].feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[0].barh(range(len(feature_importance_car)),
             feature_importance_car['importance'],
             color='steelblue', alpha=0.7)
axes[0].set_yticks(range(len(feature_importance_car)))
axes[0].set_yticklabels(feature_importance_car['feature'])
axes[0].set_xlabel('Importanță', fontsize=12)
axes[0].set_title('Top 10 Caracteristici - Car Price', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Customer Purchasing
feature_importance_customer = pd.DataFrame({
    'feature': X_customer.columns,
    'importance': results_customer['Random Forest']['model'].feature_importances_
}).sort_values('importance', ascending=False)

axes[1].barh(range(len(feature_importance_customer)),
             feature_importance_customer['importance'],
             color='coral', alpha=0.7)
axes[1].set_yticks(range(len(feature_importance_customer)))
axes[1].set_yticklabels(feature_importance_customer['feature'])
axes[1].set_xlabel('Importanță', fontsize=12)
axes[1].set_title('Toate Caracteristicile - Customer Purchasing', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance_both_datasets.png', dpi=300, bbox_inches='tight')
print("✓ Importanța caracteristicilor salvată în 'feature_importance_both_datasets.png'")
plt.show()

# ============================================================================
# SUMAR FINAL
# ============================================================================

print("\n" + "=" * 80)
print("✅ ANALIZĂ COMPLETĂ PENTRU AMBELE DATASET-URI!")
print("=" * 80)

print("\n📁 Fișiere generate:")
print("  • decision_trees_comparison.png")
print("  • metrics_comparison_both_datasets.png")
print("  • predictions_car_price.png")
print("  • predictions_customer_purchasing.png")
print("  • feature_importance_both_datasets.png")

print("\n🎯 Concluzii:")
print(f"  Car Price: Cel mai bun model = {best_car['Model']} (R² = {best_car['R²']:.4f})")
print(f"  Customer: Cel mai bun model = {best_customer['Model']} (R² = {best_customer['R²']:.4f})")

print("\n" + "=" * 80)
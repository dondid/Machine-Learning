<img width="1914" height="1141" alt="image" src="https://github.com/user-attachments/assets/d2f5f008-f799-43e5-8fe2-7ce5d75f65a4" />

<img width="3600" height="2400" alt="roc_curve_Random_Forest" src="https://github.com/user-attachments/assets/160d99ed-0030-45e1-a326-b888d0e123c9" />
<img width="3600" height="2400" alt="roc_curve_Neural_Network" src="https://github.com/user-attachments/assets/0f926458-a55a-48f3-b052-242c2e80f4af" />
<img width="3600" height="2400" alt="roc_curve_Decision_Tree" src="https://github.com/user-attachments/assets/20663a37-abed-407e-bf31-8c7dabf35f31" />
<img width="3000" height="1800" alt="feature_selection_mutual_info" src="https://github.com/user-attachments/assets/ccc0e134-fdeb-45ca-bea0-d7d588820bcc" />
<img width="3000" height="1800" alt="feature_selection_comparison" src="https://github.com/user-attachments/assets/dbad795a-4c14-4cc7-a13f-d4a68fbe32f0" />
<img width="3000" height="2400" alt="confusion_matrix_Random_Forest" src="https://github.com/user-attachments/assets/2a3a21ba-12ac-49fe-9539-9a3391d16c88" />
<img width="3000" height="2400" alt="confusion_matrix_Neural_Network" src="https://github.com/user-attachments/assets/47e83def-ac39-4f48-8b56-9e86782afd26" />
<img width="3000" height="2400" alt="confusion_matrix_Decision_Tree" src="https://github.com/user-attachments/assets/8a38c5c0-b246-42d4-ae7e-6f378d8f8b28" />

======================================================================
              CAR EVALUATION - MACHINE LEARNING ANALYSIS              
 Cursurile 6 & 7: Performance Evaluation, Feature Selection & Tuning  
======================================================================
✓ Date încărcate din UCI Repository

Dataset shape: (1728, 7)

Distribuție clase:
class
unacc    1210
acc       384
good       69
vgood      65
Name: count, dtype: int64

Primer 5 rânduri:
  buying  maint doors persons lug_boot safety  class
0  vhigh  vhigh     2       2    small    low  unacc
1  vhigh  vhigh     2       2    small    med  unacc
2  vhigh  vhigh     2       2    small   high  unacc
3  vhigh  vhigh     2       2      med    low  unacc
4  vhigh  vhigh     2       2      med    med  unacc

✓ Encodare variabile categoriale...
  buying: ['high', 'low', 'med', 'vhigh']
  maint: ['high', 'low', 'med', 'vhigh']
  doors: ['2', '3', '4', '5more']
  persons: ['2', '4', 'more']
  lug_boot: ['big', 'med', 'small']
  safety: ['high', 'low', 'med']

Target classes: ['acc', 'good', 'unacc', 'vgood']
Encoded as: [0 1 2 3]

✓ Date pregătite: 1728 samples, 6 features, 4 classes


======================================================================
              PARTEA 1: PERFORMANCE EVALUATION (CURS 6)               
======================================================================

──────────────────────────────────────────────────────────────────────
Evaluare model: Neural Network
──────────────────────────────────────────────────────────────────────
Rulare 30 iterații random subsampling...

============================================================
REZULTATE PERFORMANȚĂ - Neural Network
============================================================
Accuracy: 0.9831 ± 0.0063
F1-Score: 0.9831
Cohen's Kappa: 0.9629
  → Interpretare: Almost perfect agreement

Confusion Matrix:
[[ 3682    14   107     7]
 [   36   647     0     7]
 [   79    14 11907     0]
 [   23     3     0   604]]

Accuracy per clasă:
  acc: 0.9664
  good: 0.9377
  unacc: 0.9922
  vgood: 0.9587

AUC Scores:
  acc: 0.9987
  good: 0.9994
  unacc: 0.9992
  vgood: 0.9999
  Mean AUC: 0.9993

──────────────────────────────────────────────────────────────────────
Evaluare model: Decision Tree
──────────────────────────────────────────────────────────────────────
Rulare 30 iterații random subsampling...

============================================================
REZULTATE PERFORMANȚĂ - Decision Tree
============================================================
Accuracy: 0.9642 ± 0.0084
F1-Score: 0.9644
Cohen's Kappa: 0.9218
  → Interpretare: Almost perfect agreement

Confusion Matrix:
[[ 3521   121   141    27]
 [   49   608    10    23]
 [  171    18 11811     0]
 [   30    24     0   576]]

Accuracy per clasă:
  acc: 0.9241
  good: 0.8812
  unacc: 0.9842
  vgood: 0.9143

AUC Scores:
  acc: 0.9695
  good: 0.9802
  unacc: 0.9825
  vgood: 0.9792
  Mean AUC: 0.9779

──────────────────────────────────────────────────────────────────────
Evaluare model: Random Forest
──────────────────────────────────────────────────────────────────────
Rulare 30 iterații random subsampling...

============================================================
REZULTATE PERFORMANȚĂ - Random Forest
============================================================
Accuracy: 0.9705 ± 0.0082
F1-Score: 0.9705
Cohen's Kappa: 0.9355
  → Interpretare: Almost perfect agreement

Confusion Matrix:
[[ 3608    63   110    29]
 [   97   579     1    13]
 [  153     8 11839     0]
 [   30     2     0   598]]

Accuracy per clasă:
  acc: 0.9470
  good: 0.8391
  unacc: 0.9866
  vgood: 0.9492

AUC Scores:
  acc: 0.9941
  good: 0.9972
  unacc: 0.9991
  vgood: 0.9997
  Mean AUC: 0.9975

============================================================
WILCOXON SIGNED-RANK TEST
Comparare: Neural Network vs Random Forest
============================================================
Mean Accuracy Neural Network: 0.9831
Mean Accuracy Random Forest: 0.9705
Wilcoxon Statistic: 0.0000
P-value: 0.000002

✓ REZULTAT: Diferență statistică semnificativă (p ≤ 0.05)
  → Respingem ipoteza nulă H0
  → Cele două modele au performanțe diferite
  → Neural Network este superior lui Random Forest


======================================================================
                 PARTEA 2: FEATURE SELECTION (CURS 7)                 
======================================================================

======================================================================
COMPARARE METODE FEATURE SELECTION
======================================================================

============================================================
FEATURE SELECTION - Variance Threshold
============================================================
Threshold: 0.01
Features originale: 6
Features selectate: 6
Features eliminate: 0

Features păstrate: ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

============================================================
FEATURE SELECTION - SelectKBest (Mutual Information)
============================================================
K (număr features): 4

Top 4 features selectate:
  1. safety: 0.1711
  2. persons: 0.1446
  3. maint: 0.0555
  4. buying: 0.0538

============================================================
FEATURE SELECTION - Sequential (FORWARD)
============================================================
Antrenare forward selection (poate dura câteva minute)...
Features selectate (4): ['maint', 'doors', 'lug_boot', 'safety']

======================================================================
               Method  Features  Mean Accuracy  Std Accuracy
Original (6 features)         6       0.831102      0.070242
   Variance Threshold         6       0.831102      0.070242
     SelectKBest (MI)         4       0.717076      0.070754
   Sequential Forward         4       0.617487      0.033379
======================================================================


======================================================================
                 PARTEA 3: PARAMETER TUNING (CURS 7)                  
======================================================================

============================================================
PARAMETER TUNING - Grid Search (NEURAL_NETWORK)
============================================================
Parametri testați:
  - hidden_layer_sizes: [(50,), (100,), (50, 50), (100, 50)]
  - activation: ['relu', 'tanh']
  - alpha: [0.0001, 0.001, 0.01]
  - learning_rate: ['constant', 'adaptive']

Total combinații: 48
Rulare Grid Search (poate dura câteva minute)...
Fitting 5 folds for each of 48 candidates, totalling 240 fits

============================================================
REZULTATE GRID SEARCH
============================================================
Best Score: 0.8825
Best Parameters:
  - activation: relu
  - alpha: 0.01
  - hidden_layer_sizes: (100, 50)
  - learning_rate: constant

Top 5 configurații:

  Rank 1:
    Score: 0.8825 ± 0.0193
    Params: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100, 50), 'learning_rate': 'adaptive'}

  Rank 1:
    Score: 0.8825 ± 0.0193
    Params: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100, 50), 'learning_rate': 'constant'}

  Rank 3:
    Score: 0.8820 ± 0.0147
    Params: {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (100, 50), 'learning_rate': 'adaptive'}

  Rank 3:
    Score: 0.8820 ± 0.0147
    Params: {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (100, 50), 'learning_rate': 'constant'}

  Rank 5:
    Score: 0.8802 ± 0.0221
    Params: {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 50), 'learning_rate': 'constant'}

============================================================
PARAMETER TUNING - Grid Search (RANDOM_FOREST)
============================================================
Parametri testați:
  - n_estimators: [50, 100, 200]
  - max_depth: [None, 10, 20, 30]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]

Total combinații: 108
Rulare Grid Search (poate dura câteva minute)...
Fitting 5 folds for each of 108 candidates, totalling 540 fits

============================================================
REZULTATE GRID SEARCH
============================================================
Best Score: 0.8311
Best Parameters:
  - max_depth: None
  - min_samples_leaf: 1
  - min_samples_split: 2
  - n_estimators: 100

Top 5 configurații:

  Rank 1:
    Score: 0.8311 ± 0.0702
    Params: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

  Rank 1:
    Score: 0.8311 ± 0.0702
    Params: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

  Rank 1:
    Score: 0.8311 ± 0.0702
    Params: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

  Rank 4:
    Score: 0.8236 ± 0.0617
    Params: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}

  Rank 4:
    Score: 0.8236 ± 0.0617
    Params: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}

======================================================================
EVALUARE FINALĂ - MODELE OPTIMIZATE
======================================================================
Rulare 30 iterații random subsampling...

============================================================
REZULTATE PERFORMANȚĂ - Neural Network (Optimized)
============================================================
Accuracy: 0.9832 ± 0.0065
F1-Score: 0.9832
Cohen's Kappa: 0.9631
  → Interpretare: Almost perfect agreement
Rulare 30 iterații random subsampling...

============================================================
REZULTATE PERFORMANȚĂ - Random Forest (Optimized)
============================================================
Accuracy: 0.9705 ± 0.0082
F1-Score: 0.9705
Cohen's Kappa: 0.9355
  → Interpretare: Almost perfect agreement

======================================================================
                    ANALIZĂ COMPLETATĂ CU SUCCES!                     
======================================================================

Fișiere generate:
  - confusion_matrix_*.png
  - roc_curve_*.png
  - feature_selection_*.png

Process finished with exit code 0

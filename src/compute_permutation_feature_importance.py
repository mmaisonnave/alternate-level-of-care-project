import json
import joblib
import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from imblearn.ensemble import BalancedRandomForestClassifier

import sys

sys.path.append('..')

from utilities import configuration
from utilities import health_data


if __name__ == '__main__':
    SEED = 1593085724

    config = configuration.get_config()

    EXPERIMENT_CONFIGURATION_NAME = 'configuration_82'
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])
    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')
    print()
    print(f'X_test.shape= {X_test.shape}')
    print(f'y_test.shape= {y_test.shape}')



    brf = joblib.load(config['balanced_random_forest_path'])
    print(f"Trained model retrieved from disk ({config['balanced_random_forest_path'].split('/')[-1]})")


    print(f'Test AUC={roc_auc_score(y_true=y_test, y_score=brf.predict_proba(X_test)[:,1])}')

    # ##   DEBUG   ##
    # X_test = X_test[:10000, :100]
    # y_test = y_test[:10000,]
    # feature_names = feature_names[:100]
    # brf.fit(X_test, y_test)
    # ## END DEBUG ##

    print('Computing permutation feature importance ...')
    r = permutation_importance(brf,
                            X_test.toarray(),
                            y_test,
                            n_repeats=10,
                            scoring='roc_auc',
                            random_state=np.random.RandomState(seed=SEED))
    print('Permutation feature importance computation finished, formating results. ')

    del r['importances']
    results = pd.DataFrame(r)
    results['variable'] = feature_names

    columns = results.columns
    columns = list(columns[-1:]) + list(columns[:-1])
    results = results[columns]

    results = results.sort_values(by='importances_mean', ascending=False)

    results.to_csv(config['permutation_feature_importance_results'],
                   index=None)
    print('Results stored to disk.')
    print('Done')
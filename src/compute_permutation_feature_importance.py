"""
This script computes the permutation feature importance for a BalancedRandomForestModel model.
The scripts stores the results in a CSV file that contains the sorted features with the associated
importance score and std.

if IGNORING_DIAG_AND_INTERV==True, then the scripts computes the feature importance of a model that
only uses categorical and numerical variables (ignoring diagnosis and intervention codes).

If IGNORING_DIAG_AND_INTERV==False, the it uses the four types of features. However, this is super expensive
to compute.


If IGNORING_DIAG_AND_INTERV==True, the results is stored in:
    - permutation_feature_importance_only_num_and_cat.csv and 
    - brf_with_cat_and_num.csv (performance metric for the BRF model without interv and diag features)

If IGNORING_DIAG_AND_INTERV==False, then the results are stored in: 
    - permutation_feature_importance.csv 

"""
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,accuracy_score

import argparse
import sys

sys.path.append('..')

from utilities import configuration
from utilities import health_data
from utilities import metrics


# def _get_metric_evaluations(evaluated_model, X, y_true, model_config_name, description=''):
#     y_pred = evaluated_model.predict(X)
#     y_score = evaluated_model.predict_proba(X)[:,1]

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     results = {'Description': description,
#                         'Accuract': accuracy_score(y_true, y_pred),
#                         'Precision': precision_score(y_true, y_pred),
#                         'Recal': recall_score(y_true, y_pred),
#                         'F1-Score': f1_score(y_true, y_pred),
#                         'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
#                         'TN': tn,
#                         'TP': tp,
#                         'FN': fn,
#                         'FP': fp,
#                         'Model config': model_config_name
#                         }
#     results = {key: [results[key]] for key in results}
#     return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ignore-diag-and-interv',
                        dest='ignore_diag_and_interv',
                        required=True,
                        help='Wether to use only cat and num variables (ignoring diag and interv codes)',
                        type=str,
                        choices=['True', 'False']
                        )
    args = parser.parse_args()

    
    SEED = 1593085724
    IGNORING_DIAG_AND_INTERV= args.ignore_diag_and_interv == 'True'
    EXPERIMENT_CONFIGURATION_NAME = 'configuration_93' # (N)+(C)+(I)+ Combined D (CD)
    MODEL_CONFIGURATION_NAME='model_312' # BRF, balanced
    PERMUTATION_REPETITION_COUNT=10



    print(f'IGNORING_DIAG_AND_INTERV={IGNORING_DIAG_AND_INTERV}')

    config = configuration.get_config()

    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])
    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')
    print()
    print(f'X_test.shape= {X_test.shape}')
    print(f'y_test.shape= {y_test.shape}')


    if IGNORING_DIAG_AND_INTERV:
        print('Filtering intervention and diagnosis codes from X_train and X_test')
        diagnosis_mapping = health_data.Admission.get_diagnoses_mapping()
        intervention_mapping = health_data.Admission.get_intervention_mapping()
        codes = set(map(str.lower, 
                        intervention_mapping.keys())).union(map(str.lower,
                                                                set(diagnosis_mapping.keys())))

        cat_and_num_variable = list(filter(lambda feature: not feature.lower() in codes, feature_names))

        X_train = X_train[:,:len(cat_and_num_variable)]
        X_test = X_test[:,:len(cat_and_num_variable)]
        feature_names=feature_names[:len(cat_and_num_variable)]
        print(f'X_train.shape={X_train.shape}')
        print(f'y_train.shape={y_train.shape}')
        print()
        print(f'X_test.shape= {X_test.shape}')
        print(f'y_test.shape= {y_test.shape}')

        print('Training new BRF model with new X_train and X_test')
        brf = configuration.model_from_configuration_name(MODEL_CONFIGURATION_NAME)
        print(f'Training model MODEL_CONFIGURATION_NAME={MODEL_CONFIGURATION_NAME} ...')
        brf.fit(X_train, y_train)
        print('Storing performance of new BRF model')

        df = pd.concat([metrics.get_metric_evaluations(brf,
                                            X_train,
                                            y_train,
                                            MODEL_CONFIGURATION_NAME,
                                            experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                            description='Main BRF only cat and num (train)'
                                            ),
                        metrics.get_metric_evaluations(brf,
                                            X_test,
                                            y_test,
                                            MODEL_CONFIGURATION_NAME,
                                            experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                            description='Main BRF only cat and num (test)')])
        
        print(df[['Precision', 'Recal', 'F1-Score', 'AUC']])

        df.to_csv(config['permutation_brf_with_cat_and_num'],
                index=True)
    else: 
        print('Using all features')
        print(f'X_train.shape={X_train.shape}')
        print(f'y_train.shape={y_train.shape}')
        print()
        print(f'X_test.shape= {X_test.shape}')
        print(f'y_test.shape= {y_test.shape}')

        print('Training new BRF model with new X_train and X_test')
        brf = configuration.model_from_configuration_name(MODEL_CONFIGURATION_NAME)
        print(f'Training model MODEL_CONFIGURATION_NAME={MODEL_CONFIGURATION_NAME} ...')
        brf.fit(X_train, y_train)
        print('Storing performance of new BRF model')
        df = pd.concat([
            _get_metric_evaluations(brf, X_train, y_train, MODEL_CONFIGURATION_NAME, description='Main BRF only cat and num (train)'),
            _get_metric_evaluations(brf, X_test, y_test, MODEL_CONFIGURATION_NAME, description='Main BRF only cat and num (test)')
        ])

        
        print(df[['Precision', 'Recal', 'F1-Score', 'AUC']])

        df.to_csv(config['permutation_brf_all_features'],
                index=True)


    print('Computing permutation feature importance ...')
    r = permutation_importance(brf,
                            X_test.toarray(),
                            y_test,
                            n_repeats=PERMUTATION_REPETITION_COUNT,
                            scoring='roc_auc',
                            random_state=np.random.RandomState(seed=SEED))
    print('Permutation feature importance computation finished, formating results. ')

    del r['importances']
    results = pd.DataFrame(r)
    results['variable'] = feature_names

    columns = results.columns
    columns = list(columns[-1:]) + list(columns[:-1])
    results = results[columns]

    results = results.sort_values(by='importances_mean',
                                  ascending=False)

    output_filename = config['permutation_feature_importance_results']
    if IGNORING_DIAG_AND_INTERV:
        output_filename = output_filename[:-4] + "_only_num_and_cat.csv"

    results.to_csv(output_filename,
                   index=None)
    print(f'Results stored to disk (output_filename={output_filename}).')
    print('Done')
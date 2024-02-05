import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from imblearn.ensemble import BalancedRandomForestClassifier

import sys 
sys.path.append('..')

from utilities import health_data
from utilities import configuration



def _get_metric_evaluations(evaluated_model, X, y_true, model_config_name, description=''):
    y_pred = evaluated_model.predict(X)
    y_score = evaluated_model.predict_proba(X)[:,1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results = {'Description': description,
                        'Precision': precision_score(y_true, y_pred),
                        'Recal': recall_score(y_true, y_pred),
                        'F1-Score': f1_score(y_true, y_pred),
                        'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                        'TN': tn,
                        'TP': tp,
                        'FN': fn,
                        'FP': fp,
                        'Model config': model_config_name
                        }
    results = {key: [results[key]] for key in results}
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-disk',
                        dest='from_disk',
                        required=True,
                        help='Wether to get main model (BRF) from disk',
                        type=str,
                        choices=['True', 'False']
                        )
    args = parser.parse_args()

    config = configuration.get_config()
    print('configuration loaded')

    FROM_DISK = args.from_disk=='True'
    EXPERIMENT_CONFIGURATION_NAME = 'configuration_82'
    BRF_MODEL_CONFIGURATION_NAME = 'model_300'
    LR_MODEL_CONFIGURATION_NAME = 'model_299'
    DT_MODEL_CONFIGURATION_NAME = 'model_301'

    print(f'Using EXPERIMENT_CONFIGURATION_NAME={EXPERIMENT_CONFIGURATION_NAME}')
    print('Loading data ...')
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(
                                                    experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])


    if FROM_DISK:
        print(f'Loading model from disk (FROM_DISK={FROM_DISK})')
        brf = joblib.load(config['balanced_random_forest_path'])
    else:
        print(f'Creating model (FROM_DISK={FROM_DISK}), training ... ')
        brf = configuration.model_from_configuration_name(BRF_MODEL_CONFIGURATION_NAME)
        # model_configurations = json.load(open(config['models_config'], encoding='utf-8'))
        # model_dict = model_configurations[BRF_MODEL_CONFIGURATION_NAME]
        # MODEL_SEED = 1270833263
        # model_random_state=np.random.RandomState(MODEL_SEED)
        # brf = configuration.model_from_configuration(model_dict, random_state=model_random_state)
        print(f'Training model {str(brf)}')
        brf.fit(X_train, y_train)


    yhat_train = brf.predict(X_train)
    yhat_test = brf.predict(X_test)
    # ---------- ---------- ---------- ---------- ---------- #
    # Surrogate DecisionTreeClassifier Model                 #
    # ---------- ---------- ---------- ---------- ---------- #
    surrogate_model = configuration.model_from_configuration_name(DT_MODEL_CONFIGURATION_NAME)
    print('Training DecisionTreeClassifier ...')
    surrogate_model.fit(X_train, yhat_train, )

    # SAVING DecisionTree Plot
    fig, ax = plt.subplots(figsize=(10,10))
    tree.plot_tree(surrogate_model, feature_names=list(columns), class_names=['NR', 'R'])
    fig.savefig(config['surrogate_decision_tree_figure'], bbox_inches='tight')


    # ---------- ---------- ---------- ---------- ---------- #
    # Surrogate LogisticRegression Model                     #
    # ---------- ---------- ---------- ---------- ---------- #
    surrogate_model_lr = configuration.model_from_configuration_name(LR_MODEL_CONFIGURATION_NAME)
    print('Training LogisticRegression ...')
    surrogate_model_lr.fit(X_train, yhat_train, )

    df = pd.concat([
                _get_metric_evaluations(brf, X_train, y_train, description='BRF training', model_config_name=BRF_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(brf, X_test, y_test, description='BRF testing', model_config_name=BRF_MODEL_CONFIGURATION_NAME),

                _get_metric_evaluations(surrogate_model, X_train, y_train, description='DT training', model_config_name=DT_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(surrogate_model, X_test, y_test, description='DT testing', model_config_name=DT_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(surrogate_model, X_train, yhat_train, description='DT training (comp BRF)', model_config_name=DT_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(surrogate_model, X_test, yhat_test, description='DT testing (comp BRF)', model_config_name=DT_MODEL_CONFIGURATION_NAME),

                _get_metric_evaluations(surrogate_model_lr, X_train, y_train, description='LR training', model_config_name=LR_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(surrogate_model_lr, X_test, y_test, description='LR testing', model_config_name=LR_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(surrogate_model_lr, X_train, yhat_train, description='LR training (comp BRF)', model_config_name=LR_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(surrogate_model_lr, X_test, yhat_test, description='LR testing (comp BRF)', model_config_name=LR_MODEL_CONFIGURATION_NAME),
            ])
    
    df['configuration_name'] = [EXPERIMENT_CONFIGURATION_NAME]*df.shape[0]
    df.to_csv(config['surrogate_models_results'], index=False)

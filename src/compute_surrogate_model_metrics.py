"""
The purpose of this script was to create surrogate models (LogReg and DecisionTree) for the best model we have
so far (BalancedRandomForest). However, I am going to replace that to the models trained on the ground truth 
data (not trained to copy the BRF model).

So, this script would be replaced by:
    - run_explainable_dt.py
    - run_explainable_logreg.py
"""
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler
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
    X_train, y_train, X_test, y_test, features_names = health_data.Admission.get_train_test_matrices(
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
    fig, ax = plt.subplots(figsize=(20,10))
    tree.plot_tree(surrogate_model, 
                   feature_names=list(features_names), 
                   class_names=['NR', 'R'],
                   fontsize=9,
                   impurity=False,
                   label='none',
                   filled=True,
                   )
    fig.savefig(config['surrogate_decision_tree_figure'], bbox_inches='tight')


    # ---------- ---------- ---------- ---------- ---------- #
    # Surrogate LogisticRegression Model                     #
    # ---------- ---------- ---------- ---------- ---------- #
    surrogate_model_lr = configuration.model_from_configuration_name(LR_MODEL_CONFIGURATION_NAME)
    print('Training LogisticRegression ...')
    surrogate_model_lr.fit(X_train,
                           yhat_train,)


    # ---------- ---------- ---------- ---------- ---------- #
    # Surrogate NORMALIZED LogisticRegression Model          #
    # ---------- ---------- ---------- ---------- ---------- #
    scaler = StandardScaler()
    # norm_X_train = scaler.fit_transform(X_train.toarray())
    norm_X_test = scaler.fit_transform(X_test.toarray())

    surrogate_model_norm_lr = configuration.model_from_configuration_name(LR_MODEL_CONFIGURATION_NAME)
    print('Training LogisticRegression with normalize features (to analyze coefficients) ...')
    surrogate_model_norm_lr.fit(norm_X_test,
                        yhat_test, )
    
    # ---------- ---------- #
    # SAVING COEFFICIENTS   #
    # ---------- ---------- #
    print('Finished computing coefficients (normalized features), formating and saving ...')
    scored_feature_names = list(zip(list(surrogate_model_norm_lr.coef_[0,:]),
                                    features_names))

    scored_feature_names = sorted(scored_feature_names, 
                                  key=lambda x:np.abs(x[0]), reverse=True)

    coefficients_df = pd.DataFrame(scored_feature_names,
                                   columns=['Score', 'Feature Name'])
    
    coefficients_df = coefficients_df[['Feature Name', 'Score']]

    coefficients_df.to_csv(config['surrogate_logreg_coefficients'], index=False)
    print('DONE, saving metrics ...')

    # ---------- ---------- #
    # SAVING METRICS        #
    # ---------- ---------- #

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

                # _get_metric_evaluations(surrogate_model_norm_lr, norm_X_train, y_train, description='Norm LR training', model_config_name=LR_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(surrogate_model_norm_lr, norm_X_test, y_test, description='Norm LR testing', model_config_name=LR_MODEL_CONFIGURATION_NAME),
                # _get_metric_evaluations(surrogate_model_norm_lr, norm_X_train, yhat_train, description='Norm LR training (comp BRF)', model_config_name=LR_MODEL_CONFIGURATION_NAME),
                _get_metric_evaluations(surrogate_model_norm_lr, norm_X_test, yhat_test, description='Norm LR testing (comp BRF)', model_config_name=LR_MODEL_CONFIGURATION_NAME),
            ])
    
    df['configuration_name'] = [EXPERIMENT_CONFIGURATION_NAME]*df.shape[0]
    df.to_csv(config['surrogate_models_results'], index=False)
    print('DONE')


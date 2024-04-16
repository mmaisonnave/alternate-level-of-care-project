"""
This script takes configuration no 82 (configuration_82) (All dummy features min_df=2), and loads an already
trained  BRF model from disk. For each numeric var (['age', 'cmg', 'case_weight', 'acute_days', 'alc_days]),
the scripts computes the partial dependency plot (PDP) and it stores it in disk (results/figures/PDP_{var}.jpg)
"""
import sys
import json
import pickle
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
sys.path.append('..')

from utilities import configuration
from utilities import health_data
from utilities import metrics

if __name__ == '__main__':
    SHAP_SAMPLE_SIZE=5000
    EXPERIMENT_CONFIGURATION_NAME='configuration_31_under_and_over'
    MODEL_CONFIGURATION_NAME ="model_305" # BRF, not balanced
    print(f'Using SHAP_SAMPLE_SIZE={SHAP_SAMPLE_SIZE}')
    config = configuration.get_config()
    print(f'Loading data from  EXPERIMENT_CONFIGURATIO_NAME={EXPERIMENT_CONFIGURATION_NAME} ...')
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(
                                    experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])
    print(f'X_train.shape={X_train.shape}')

    # brf = joblib.load(config['balanced_random_forest_path'])

    brf = configuration.model_from_configuration_name(MODEL_CONFIGURATION_NAME)
    print('Training from scratch....')
    brf.fit(X_train, y_train)

    df = pd.concat([metrics.get_metric_evaluations(brf,
                                        X_train,
                                        y_train,
                                        MODEL_CONFIGURATION_NAME,
                                        experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                        description='TRAIN'
                                        ),
                    metrics.get_metric_evaluations(brf,
                                        X_test,
                                        y_test,
                                        MODEL_CONFIGURATION_NAME,
                                        experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                        description='TEST')])
    
    df.to_csv(config['pdp_model_performance'], index=False)


    print(f'Test AUC={roc_auc_score(y_true=y_test, y_score=brf.predict_proba(X_test)[:,1])}')


    sampled_X = shap.utils.sample(X_test,
                         SHAP_SAMPLE_SIZE).toarray()  # 5000 instances for use as the background distribution


    output_files = [config[name] for name in ["ppd_age_figure",
                                            "ppd_cmg_figure",
                                            "ppd_case_weight_figure",
                                            "ppd_acute_days_figure",
                                            "ppd_alc_days_figure"]]


    new_names = ['Age', 'CMG', 'RIW', 'Acute Days', 'ALC days']
    for var_,output_file in zip(new_names,output_files):
        fig, ax = plt.subplots(1, figsize=(6.4, 4.8))
        shap.partial_dependence_plot(
            var_,
            brf.predict,
            sampled_X,
            feature_names=new_names,
            ice=False,
            model_expected_value=True,
            feature_expected_value=True,
            ax=ax
        )

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)

        fig.savefig(output_file, 
                    bbox_inches='tight')
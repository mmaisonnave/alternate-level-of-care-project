"""
This script takes configuration no 82 (configuration_82) (All dummy features min_df=2), and loads an already
trained  BRF model from disk. For each numeric var (['age', 'cmg', 'case_weight', 'acute_days', 'alc_days]),
the scripts computes the partial dependency plot (PDP) and it stores it in disk (results/figures/PDP_{var}.jpg)
"""
import sys
import json
import pickle
from imblearn.ensemble import BalancedRandomForestClassifier
import shap
import matplotlib.pyplot as plt
import joblib

sys.path.append('..')

from utilities import configuration
from utilities import health_data
if __name__ == '__main__':
    SHAP_SAMPLE_SIZE=5000
    EXPERIMENT_CONFIGURATIO_NAME='configuration_82'
    print(f'Using SHAP_SAMPLE_SIZE={SHAP_SAMPLE_SIZE}')
    config = configuration.get_config()
    print(f'Loading data from  EXPERIMENT_CONFIGURATIO_NAME={EXPERIMENT_CONFIGURATIO_NAME} ...')
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(
                                    experiment_configurations[EXPERIMENT_CONFIGURATIO_NAME])
    print(f'X_train.shape={X_train.shape}')

    brf = joblib.load(config['balanced_random_forest_path'])

    from sklearn.metrics import roc_auc_score
    print(f'Test AUC={roc_auc_score(y_true=y_test, y_score=brf.predict_proba(X_test)[:,1])}')


    sampled_X = shap.utils.sample(X_test,
                         SHAP_SAMPLE_SIZE).toarray()  # 5000 instances for use as the background distribution


    output_files = [config[name] for name in ["ppd_age_figure",
                                            "ppd_cmg_figure",
                                            "ppd_case_weight_figure",
                                            "ppd_acute_days_figure",
                                            "ppd_alc_days_figure"]]

    for var_,output_file in zip(['age', 'cmg', 'case_weight', 'acute_days', 'alc_days'],
                               output_files):
        fig, ax = plt.subplots(1, figsize=(10,10))
        shap.partial_dependence_plot(
            var_,
            brf.predict,
            sampled_X,
            feature_names=columns,
            ice=False,
            model_expected_value=True,
            feature_expected_value=True,
            ax=ax
        )
        fig.savefig(output_file, bbox_inches='tight')
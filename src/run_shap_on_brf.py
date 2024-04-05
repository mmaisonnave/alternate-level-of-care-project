import sys
import pickle
import joblib
import pandas as pd
sys.path.append('..')

from utilities import configuration
from utilities import health_data
from utilities import metrics

import json
import shap

import matplotlib.pyplot as plt


if __name__ == "__main__":
    SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION=10000
    SAMPLE_SIZE_TO_COMPUT_SHAP_ON=100000
    EXPERIMENT_CONFIGURATION_NAME = 'configuration_91' # (N)+(C)+(D)+(I) with U(1.0) and O(0.1)
    MODEL_CONFIGURATION_NAME ="model_305" # BRF, not balanced
    FROM_DISK = False # If true, MODEL_CONFIGURATION_NAME not used.

    config = configuration.get_config()
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])


    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={X_train.shape}')
    print()

    print(f'X_test.shape= {X_test.shape}')
    print(f'y_test.shape= {y_test.shape}')
    print()



    if FROM_DISK:
        print('Loading model from disk ...')
        brf = joblib.load(config['balanced_random_forest_path'])
    else:
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

    print(df[['Precision', 'Recal', 'F1-Score', 'AUC']])

    sampled_X = shap.utils.sample(X_test.toarray(),
                                  SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION
                                  )  # No. of instances for use as the background distribution

    print(f'Sampled X (for background noise) shape={sampled_X.shape}')

    print('Creating explainer ...')
    explainer = shap.Explainer(brf.predict,
                               sampled_X,
                               feature_names=feature_names,
                               )

    print('Calculating SHAP VALUES')
    shap_values = explainer(X_test[:SAMPLE_SIZE_TO_COMPUT_SHAP_ON,:].toarray(),
                            max_evals=2*X_test.shape[1]+1,
                            )

    print(f'shap_values.shape={shap_values.shape}')
    shap.plots.beeswarm(shap_values,
                        max_display=30,
                        )

    fig, ax = plt.gcf(), plt.gca()

    figure_name = config['shap_on_brf_figures'][:-len(".jpg")]
    figure_name = figure_name + f'_BGD={SAMPLE_SIZE_FOR_BACKGROUND_DISTRIBUTION}'
    figure_name = figure_name + f'_SS={SAMPLE_SIZE_TO_COMPUT_SHAP_ON}.jpg'

    fig.savefig(figure_name,
                bbox_inches='tight')
"""
This scripts just takes 'configuration_82' (All dummy features min_df=2), and 
'model_104' (BRF, n_estimators=500, sampling_strategy="majority", replacement=True, 
class_weight=balanced_subsample) creates and trains the model.

The resulting model is stored in checkpoints/balanced_random_forest.joblib

"""
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import joblib
import pandas as pd
import json
import sys 
sys.path.append('..')

from utilities import health_data
from utilities import configuration
from utilities import metrics

if __name__ == '__main__':
    config = configuration.get_config()

    EXPERIMENT_CONFIGURATION_NAME = 'configuration_82'
    print(f'Using configuration = {EXPERIMENT_CONFIGURATION_NAME}')
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))

    print('Computing training and testing matrices ...')
    X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])



    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')

    print(f'X_test.shape= {X_train.shape}')
    print(f'y_test.shape= {y_train.shape}')
    print(f'len(columns)= {len(columns)}')

    MODEL_CONFIGURATION_NAME='model_300'
    # print(f'Using model configuration name={MODEL_NAME}')
    # with open(config['models_config'], encoding='utf-8') as reader:
    #     model_configurations = json.load(reader)

    # MODEL_SEED = 1270833263
    # print(f'MODEL_SEED={MODEL_SEED}')
    # model_random_state=np.random.RandomState(MODEL_SEED)
    # model = configuration.model_from_configuration(model_configurations[MODEL_NAME],
    #                                                random_state=model_random_state)

    model = configuration.model_from_configuration_name(MODEL_CONFIGURATION_NAME)
    print(f'model={str(model)}')

    print('Training model ...')
    model.fit(X_train, y_train)

    print(f"Model trained, saving in {config['balanced_random_forest_path']}")
    config = configuration.get_config()
    joblib.dump(model, config['balanced_random_forest_path'])

    df = pd.concat([metrics.get_metric_evaluations(model,
                                        X_train,
                                        y_train,
                                        MODEL_CONFIGURATION_NAME,
                                        experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                        description='TRAIN'
                                        ),
                    metrics.get_metric_evaluations(model,
                                            X_test,
                                            y_test,
                                            MODEL_CONFIGURATION_NAME,
                                            experiment_config_name=EXPERIMENT_CONFIGURATION_NAME,
                                            description='TEST')])
    
    df.to_csv(config['balanced_random_forest_metrics'],
              index=None)
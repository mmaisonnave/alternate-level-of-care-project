from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import joblib
import pandas as pd
import json
import sys 
sys.path.append('..')

from utilities import health_data
from utilities import configuration

config = configuration.get_config()


CONFIGURATION_NAME = 'configuration_82'
print(f'Using configuration = {CONFIGURATION_NAME}')
experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))

print('Computing training and testing matrices ...')
X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(experiment_configurations[CONFIGURATION_NAME])



print(f'X_train.shape={X_train.shape}')
print(f'y_train.shape={y_train.shape}')

print(f'X_test.shape= {X_train.shape}')
print(f'y_test.shape= {y_train.shape}')
print(f'len(columns)= {len(columns)}')

from imblearn.ensemble import BalancedRandomForestClassifier
MODEL_NAME='model_104'
print(f'Using model configuration name={MODEL_NAME}')
with open(config['models_config'], encoding='utf-8') as reader:
    model_configurations = json.load(reader)

MODEL_SEED = 1270833263
print(f'MODEL_SEED={MODEL_SEED}')
model_random_state=np.random.RandomState(MODEL_SEED)
model = configuration.model_from_configuration(model_configurations[MODEL_NAME],
                                               random_state=model_random_state)

print(f'model={str(model)}')

print('Training model ...')
model.fit(X_train, y_train)

print(f"Model trained, saving in {config['balanced_random_forest_path']}")
config = configuration.get_config()
joblib.dump(model, config['balanced_random_forest_path']) 


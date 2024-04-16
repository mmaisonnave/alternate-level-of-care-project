import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,accuracy_score

import sys
sys.path.append('..')
from utilities import configuration
from utilities import health_data

if __name__ == '__main__':
    EXPERIMENT_CONFIGURATION_NAME='configuration_108' # (N)+(C)+(I)+(CD)+U(1.0) + O(0.1)
    # MODEL_CONFIGURATION_NAME = 'model_316' # (N)+(C)+(I)+ Combined D (CD) + class balanced weights

    print(f'Using EXPERIMENT_CONFIGURATION_NAME={EXPERIMENT_CONFIGURATION_NAME}')
    # print(f'Using MODEL_CONFIGURATION_NAME=     {MODEL_CONFIGURATION_NAME}')

    config = configuration.get_config()

    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    X_train, y_train, X_test, y_test, features_names = health_data.Admission.get_train_test_matrices(experiment_configurations[EXPERIMENT_CONFIGURATION_NAME])

    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')
    print()
    print(f'X_test.shape= {X_test.shape}')
    print(f'y_test.shape= {y_test.shape}')
    print()


    print('Standarizing data ...')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.toarray())
    X_test = scaler.transform(X_test.toarray())

    print('Creating model ...')
    # logreg = configuration.model_from_configuration_name(MODEL_CONFIGURATION_NAME)
    logit_model=sm.Logit(y_train,X_train)


    print('Training data ...')
    result=logit_model.fit()

    print(result.summary())

    def _get_metric_evaluations(evaluated_model, params, X, y_true, description=''):
        y_pred = evaluated_model.predict(params, X)>0.5
        y_score = evaluated_model.predict(params, X)[:,1]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results = {'Description': description,
                            'Accuracy': accuracy_score(y_true, y_pred),
                            'Precision': precision_score(y_true, y_pred),
                            'Recal': recall_score(y_true, y_pred),
                            'F1-Score': f1_score(y_true, y_pred),
                            'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                            'TN': tn,
                            'TP': tp,
                            'FN': fn,
                            'FP': fp,
                            'Model config': "Logit statsmodel"
                            }
        results = {key: [results[key]] for key in results}
        return pd.DataFrame(results)

    print('Computing model results ...')
    df = pd.concat([_get_metric_evaluations(logit_model,result.params, X_train, y_train, description='logit train'),
                    _get_metric_evaluations(logit_model,result.params, X_test, y_test, description='logit test')])


    df.to_csv(config['explainable_logit_metrics'], index=False)


    print('Formating and storing resuts ...')
    diagnosis_mapping = health_data.Admission.get_diagnoses_mapping()
    intervention_mapping = health_data.Admission.get_intervention_mapping()


    def code2description(code):
        if code.upper() in diagnosis_mapping or code in diagnosis_mapping:
            assert not (code.upper() in intervention_mapping or code in intervention_mapping)
            return "DIAG: "+diagnosis_mapping[code.upper()]
        
        if code.upper() in intervention_mapping or code in intervention_mapping:
            assert not (code.upper() in diagnosis_mapping or code in diagnosis_mapping)
            return "INT: "+intervention_mapping[code.upper()]
        return "N/A"


    print('Finished computing coefficients (normalized features), formating and saving ...')
    scored_feature_names = list(zip(list(result.params),
                                    features_names))

    scored_feature_names = sorted(scored_feature_names, 
                                key=lambda x:np.abs(x[0]), reverse=True)

    coefficients_df = pd.DataFrame(scored_feature_names,
                                columns=['Score', 'Feature Name'])

    coefficients_df = coefficients_df[['Feature Name', 'Score']]
    coefficients_df['Code Description'] = list(map(code2description, coefficients_df['Feature Name']))
    coefficients_df['p-value'] = result.pvalues

    coefficients_df.to_csv(config['explainable_logit_coefficients'], index=False)
    print('DONE')

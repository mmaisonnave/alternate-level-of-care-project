import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn import tree
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utilities import configuration
from utilities import health_data
import json

def _capitalize_feature_name(feature_name:str)->str:
    if feature_name=='cmg':
        return 'CMG'
    elif feature_name=='case_weight':
        return 'RIW'
    else:
        aux = feature_name.replace('_', ' ').replace('-', ' ').strip()
        if aux.split(' ')=='':
            print('Error')
            print(aux)
        return ' '.join([word[0].upper()+word[1:] for word in aux.split(' ')])



if __name__ == '__main__':
    EXPERIMENT_CONFIGURATION_NAMES = [('configuration_27', '(N)'),      #  + U(1.0) + O(0.1)
                                      ('configuration_28', '(C)'),      # + U(1.0) + O(0.1)
                                      ('configuration_87','(N)_(C)'), # + U(1.0) + O(0.1)
                                      ('configuration_30', '(I)'),      #  + U(1.0) + O(0.1)
                                      ('configuration_29_combined', 'Combined D (CD)'), #  +Combined D (CD)
                                      ('configuration_93', '(N)_(C)_(D)_(I)') #(N)+(C)+(I)+ Combined D (CD)
                                      ]
    config = configuration.get_config()
    metric_dfs=[]
    for experiment_configuration_name, experiment_configuration_description in EXPERIMENT_CONFIGURATION_NAMES:
        print(f'EXPERIMENT_CONFIGURATION_NAME={experiment_configuration_name}')
        experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
        X_train, y_train, X_test, y_test, features_names = health_data.Admission.get_train_test_matrices(experiment_configurations[experiment_configuration_name])

        print(f'X_train.shape={X_train.shape}')
        print(f'y_train.shape={X_train.shape}')
        print()

        print(f'X_test.shape= {X_test.shape}')
        print(f'y_test.shape= {y_test.shape}')
        print()

        DT_MODEL_CONFIGURATION_NAME = 'model_315_depth_3'
        print(f'DT_MODEL_CONFIGURATION_NAME={DT_MODEL_CONFIGURATION_NAME}')
        dt_model = configuration.model_from_configuration_name(DT_MODEL_CONFIGURATION_NAME)
        dt_model.fit(X_train, y_train)



        def _get_metric_evaluations(evaluated_model, X, y_true, model_config_name, experiment_config_name, description='', ):
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
                                'Experiment config':experiment_config_name,
                                'Model config': model_config_name
                                }
            results = {key: [results[key]] for key in results}
            return pd.DataFrame(results)

        metric_dfs += [_get_metric_evaluations(dt_model,
                                                X_train,
                                                y_train,
                                                DT_MODEL_CONFIGURATION_NAME, 
                                                experiment_config_name=experiment_configuration_name,
                                                description='TRAIN'
                                                ),
                        _get_metric_evaluations(dt_model,
                                                X_test,
                                                y_test,
                                                DT_MODEL_CONFIGURATION_NAME, 
                                                experiment_config_name=experiment_configuration_name,
                                                description='TEST')]



        fig, ax = plt.subplots(figsize=(14,7))
        tree.plot_tree(dt_model,
                    feature_names=list(map(_capitalize_feature_name, features_names)),
                    class_names=['NR', 'R'],
                    fontsize=13,
                    impurity=False,
                    label='none',
                    filled=True,
                    node_ids=False,
                    )

        output_file = config['explainable_dt_figures'].replace('.jpg', f'_{experiment_configuration_description}.jpg')

        fig.savefig(output_file, bbox_inches='tight')

    df = pd.concat(metric_dfs)
    df.to_csv(config['explainable_dt_metrics'], index=None)

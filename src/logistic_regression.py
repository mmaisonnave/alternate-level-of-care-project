import argparse
import os
import numpy as np
import pandas as pd
import re
import sys
sys.path.append('..')
from utilities import logger
from utilities import configuration
from utilities import health_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score,recall_score,roc_auc_score,confusion_matrix
from scipy import sparse
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # LOGGING
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    config = configuration.get_config()
    logging = logger.init_logger(config['logreg_log'])
    logging.debug('Starting Logistic Regression experiments ...')

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # MANAGING ARGUMENTS
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    parser = argparse.ArgumentParser()
    parser.add_argument('--categorical-features', action=argparse.BooleanOptionalAction)
    parser.add_argument('--numerical-features', action=argparse.BooleanOptionalAction)
    parser.add_argument('--diagnosis-features', action=argparse.BooleanOptionalAction)
    parser.add_argument('--intervention-features', action=argparse.BooleanOptionalAction)

    parser.add_argument('--fix-missings', action=argparse.BooleanOptionalAction)
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction)
    parser.add_argument('--fix-skew', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-idf', action=argparse.BooleanOptionalAction)
    parser.add_argument('--class-balanced', action=argparse.BooleanOptionalAction)
    parser.add_argument('--remove-outliers', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    numerical_features = True if args.numerical_features else False
    categorical_features = True if args.categorical_features else False
    diagnosis_features = True if args.diagnosis_features else False
    intervention_features = True if args.intervention_features else False

    fix_missings = True if args.fix_missings else False
    normalize = True if args.normalize else False
    fix_skew = True if args.fix_skew else False
    use_idf = True if args.use_idf else False
    class_balanced = True if args.class_balanced else False
    remove_outliers = True if args.remove_outliers else False

    params = {'fix_skew': fix_skew,
              'normalize': normalize,
              'fix_missing_in_testing': fix_missings,
              'numerical_features': numerical_features,
              'categorical_features': categorical_features,
              'diagnosis_features': diagnosis_features,
              'intervention_features':intervention_features,
              'use_idf':use_idf,
              'class_balanced':class_balanced,
              'remove_outliers': remove_outliers,
          }
    
    for key, value in params.items():
        logging.debug(f'{key:30}={value}')
    
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # RETRIEVING TRAIN AND TEST
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    training ,testing = health_data.Admission.get_training_testing_data()
    if params['fix_missing_in_testing']:
        for admission in testing:
            admission.fix_missings(training)
            
    logging.debug(f'Training size={len(training):,}')
    logging.debug(f'Testing  size={len(testing):,}')

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # TRAINING MATRIX
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    features = []
    if params['numerical_features']:
        numerical_df = health_data.Admission.numerical_features(training,)
        
        if params['remove_outliers']:
            stds = np.std(numerical_df)
            mean = np.mean(numerical_df, axis=0)
            is_outlier=np.sum(numerical_df.values > (mean+4*stds).values, axis=1)>0
        
        if params['fix_skew']:
            numerical_df['case_weight'] = np.log10(numerical_df['case_weight']+1)
            numerical_df['acute_days'] = np.log10(numerical_df['acute_days']+1)
            numerical_df['alc_days'] = np.log10(numerical_df['alc_days']+1)

        if params['normalize']:
            scaler = StandardScaler()
            if params['remove_outliers']:
                scaler.fit(numerical_df.values[~is_outlier,:])
            else:
                scaler.fit(numerical_df.values)
            numerical_df = pd.DataFrame(scaler.transform(numerical_df.values), columns=numerical_df.columns)

        features.append(sparse.csr_matrix(numerical_df.values))

    if params['categorical_features']:
        categorical_df, main_pt_services_list = health_data.Admission.categorical_features(training)
        features.append(sparse.csr_matrix(categorical_df.values))

    if params['diagnosis_features']:
        vocab_diagnosis, diagnosis_matrix = health_data.Admission.diagnosis_codes_features(training, 
                                                                                        use_idf=params['use_idf'])
        features.append(diagnosis_matrix)

    if params['intervention_features']:
        vocab_interventions, intervention_matrix = health_data.Admission.intervention_codes_features(training, 
                                                                                                    use_idf=params['use_idf'])
        features.append(intervention_matrix)

    if params['remove_outliers']:
        mask=~is_outlier
    else:
        mask = np.ones(shape=(len(training)))==1

    X_train = sparse.hstack([matrix[mask,:] for matrix in features])
    y_train = health_data.Admission.get_y(training)[mask]


    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # TESTING MATRIX
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    features = []
    if params['numerical_features']:
        numerical_df = health_data.Admission.numerical_features(testing,)
        
        if params['fix_skew']:
            numerical_df['case_weight'] = np.log10(numerical_df['case_weight']+1)
            numerical_df['acute_days'] = np.log10(numerical_df['acute_days']+1)
            numerical_df['alc_days'] = np.log10(numerical_df['alc_days']+1)

        if params['normalize']:
            numerical_df = pd.DataFrame(scaler.transform(numerical_df.values), columns=numerical_df.columns)
        features.append(sparse.csr_matrix(numerical_df.values))

    if params['categorical_features']:
        categorical_df,_ = health_data.Admission.categorical_features(testing, main_pt_services_list=main_pt_services_list)
        features.append(sparse.csr_matrix(categorical_df.values))

    if params['diagnosis_features']:
        vocab_diagnosis, diagnosis_matrix = health_data.Admission.diagnosis_codes_features(testing, 
                                                                                        vocabulary=vocab_diagnosis, 
                                                                                        use_idf=params['use_idf'])
        features.append(diagnosis_matrix)

    if params['intervention_features']:
        vocab_interventions, intervention_matrix = health_data.Admission.intervention_codes_features(testing, 
                                                                                                    vocabulary=vocab_interventions, 
                                                                                                    use_idf=params['use_idf']
                                                                                                    )
        features.append(intervention_matrix)

    X_test = sparse.hstack(features)
    y_test = health_data.Admission.get_y(testing)


    logging.debug(f'X_train.shape = ({X_train.shape[0]:,} x {X_train.shape[1]:,})')
    logging.debug(f'y_train.shape = ({y_train.shape[0]:,} x )')
    # print()
    logging.debug(f'X_test.shape =  ({X_test.shape[0]:,} x {X_test.shape[1]:,})')
    logging.debug(f'y_test.shape =  ({y_test.shape[0]:,} x )')


    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # LOGISTIC REGRESSION MODEL 
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    class_weight = 'balanced' if params['class_balanced'] else None
    clf = LogisticRegression(class_weight=class_weight, max_iter=7000,).fit(X_train, y_train,)
    
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # RESULTS (METRICS)
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    y_true = y_train
    y_pred = clf.predict(X_train)
    y_score= clf.predict_proba(X_train)

    model_name = str(clf)
    columns = ['Model','split','TN','FP','FN','TP','Precision','Recall','F1-Score','AUC']
    param_names  = [key for key in params.keys()]
    columns = columns[0:1] + param_names + columns[1:]
    str_ = ';'.join(columns)
    logging.debug(str_)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    str_ = f'{model_name};{str(params)};TRAIN;{tn};{fp};{fn};{tp};{precision_score(y_true, y_pred,)};{recall_score(y_true, y_pred,)};'\
        f'{f1_score(y_true, y_pred,)};{roc_auc_score(y_true=y_true, y_score=y_pred)}'
    logging.debug(str_)


    vec1 = [model_name,
            'TRAIN',
            tn,
            fp,
            fn,
            tp,
            precision_score(y_true, y_pred,),
            recall_score(y_true, y_pred,),
            f1_score(y_true, y_pred,),
            roc_auc_score(y_true=y_true, y_score=y_pred),
            ]
    vec1 = vec1[0:1] + [params[param_name] for param_name in param_names] + vec1[1:]

    y_true = y_test
    y_pred = clf.predict(X_test)
    y_score= clf.predict_proba(X_test)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    str_ = f'{model_name};{str(params)};TEST;{tn};{fp};{fn};{tp};{precision_score(y_true, y_pred,)};{recall_score(y_true, y_pred,)};'\
        f'{f1_score(y_true, y_pred,):};{roc_auc_score(y_true=y_true, y_score=y_pred)}'
    logging.debug(str_)                         

    vec2 = [model_name,
            'TEST',
            tn,
            fp,
            fn,
            tp,
            precision_score(y_true, y_pred,),
            recall_score(y_true, y_pred,),
            f1_score(y_true, y_pred,),
            roc_auc_score(y_true=y_true, y_score=y_pred),
            ]
    vec2 = vec2[0:1] + [params[param_name] for param_name in param_names] + vec2[1:]
    m = np.vstack([vec1, vec2])
    df = pd.DataFrame(m, columns=columns)

    if os.path.isfile(config['logreg_results']):
        old_df = pd.read_csv(config['logreg_results'], sep=';')
        df = pd.concat([old_df,df])

    df.to_csv(config['logreg_results'], index=False, sep=';')

    # Using the last version from disk (I read everythin again because the 
    # last row it is displayed diferently if I don't)
    df = pd.read_csv(config['logreg_results'], sep=';')

    latex_df = df.drop(columns=['fix_missing_in_testing', 'class_balanced','Model', ])
    latex_df = latex_df[latex_df['split']=='TEST']
    latex_df = latex_df.drop(columns=['split', ])

    latex_df = latex_df.rename(columns={"fix_skew": "Fix Skew", 
                                        "normalize": "Normalize",
                                        "numerical_features": "Numerical Features",
                                        "categorical_features": "Categorical Features",
                                        "diagnosis_features": "Diagnoses",
                                        "intervention_features": "Interventions",
                                        "use_idf": "Use IDF",
                                        "remove_outliers": "Remove Outliers",
                                        })
    # latex_df.columns = [elem.replace('_','\\_') for elem in latex_df.columns]
    # latex_df['Model'] = [re.sub('\(.*\)','',model_name) for model_name in latex_df['Model'] ]
    
    latex_df.to_latex(config['logreg_latex'],float_format=f"{{:0.3f}}".format, index=False)
    logging.debug('Finishing Logistic Regression execution\n')

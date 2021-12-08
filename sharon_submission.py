#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import pandas as pd
import numpy as np
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import dill as pickle
from datetime import datetime
from numbers import Number
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import torch # For building the networks 
import torchtuples as tt # Some useful functions
from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis


# In[85]:


import dill
dill.__version__


# In[82]:


cancer_types = ['BLCA', 'BRCA', 'HNSC', 'LAML', 'LGG', 'LUAD']
omics_types = ['exp', 'methy', 'mirna']
task_ids = [1,2,3]
path_prefix = 'C:/Users/sharony/SurvivalAnalysis/'
if os.getlogin() =='meiry':
    path_prefix = 'D:/sharon/medical_genomics_data/'
if os.getlogin() =='yedid':
    path_prefix = "C:/Users/yedid/Sharon/sharon from laptop/medical_genomics_data/"

min_feature_std = 0.00 #0.01

#script_name <task_id> <cancer type> <input_file/dir_paths> <output_file_path>

model_path = "C:/Users/yedid/Sharon/Original/"
output_file_path = "C:/Users/yedid/Sharon/sharon from laptop/submissions/"
train_data_path_prefix = "C:/Users/yedid/Sharon/Original/"
held_out_cancer_dir = "C:/Users/yedid/Sharon/Original/"


# In[5]:



def calc_time_to_event(x, verbose = False):
    #print(type(x) , x)
    assert x.vital_status in ['Dead', 'Alive']
    if x.vital_status=='Dead':
        try:
            assert isinstance(int(x.death_days_to), int)
        except:
            if verbose:
                print(type(x.death_days_to), 'non int entry in x.death_days_to' , x.death_days_to, 'removing row',x.vital_status, x.death_days_to)        
            return None
        assert float(x.death_days_to) >= 0
        return int(x.death_days_to)
    try:
        assert isinstance(int(x.last_contact_days_to), int) 
    except :
        if verbose:
            print(type(x.last_contact_days_to), 'non int entry in x.last_contact_days_to' , x.last_contact_days_to, 'removing row',x.vital_status, x.death_days_to)        
        return None
    if int(x.last_contact_days_to) < 0:
        if verbose:
            print('negative entry in x.last_contact_days_to' , x.last_contact_days_to, 'fixing it')
    return abs(int(x.last_contact_days_to))


# In[6]:


def uncorrelated_features(ct, comment, omics, min_correlation_threshold = 0.9, max_features_to_correlate = None  ):
    if max_features_to_correlate is None: 
        max_features_to_correlate = omics.shape[1]
    assert(isinstance(max_features_to_correlate, int))
    corr_matrix = omics.iloc[:,0:max_features_to_correlate-1].corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > min_correlation_threshold)]
    print(datetime.now(), ct, comment,  'dropping highly correlated features', len(to_drop_corr))
    cols_to_keep = [col not in to_drop_corr for col in omics.columns]
    return cols_to_keep


# In[7]:


# in training
def process_omics(cancer_type, omics_type, omics_raw, clinical_data, comment=''):
    print('process_omics', 'cancer_type:', cancer_type, omics_type, 'shape:', omics_raw.shape)
    dup_keep_mask = ~omics_raw.duplicated()
    std_keep_mask = omics_raw.std(axis=1, skipna=True) > min_feature_std   
    scaler = MinMaxScaler()
    omics_raw_t_scaled = pd.DataFrame(scaler.fit_transform(omics_raw.T), columns=omics_raw.index)
    T = clinical_data.apply(calc_time_to_event, axis = 1)
    E = (clinical_data["vital_status"] == 'Dead').astype(bool)
    print('clinical:', clinical_data.shape, 'T: len=', len(T), 'notnull:', sum(T.notnull()),
          omics_raw_t_scaled.shape)
    print('T null:', T.isnull().sum(),' E null:', E.isnull().sum(), omics_raw_t_scaled.isnull().sum().sum() )
    importances_E = mutual_info_classif(omics_raw_t_scaled.loc[T.notnull().values],
                                        E.loc[T.notnull().values])
    importances_T = mutual_info_classif(omics_raw_t_scaled.loc[T.notnull().values],
                                        T.loc[T.notnull().values])
    feature_importances_E = pd.Series(importances_E, omics_raw.index)
    feature_importances_T = pd.Series(importances_T, omics_raw.index)
    drop_low_ig_E = list(feature_importances_E[feature_importances_E==0].keys())
    drop_low_ig_T = list(feature_importances_T[feature_importances_T==0].keys())
    ig_keep_mask = [ (col not in drop_low_ig_T) and (col not in drop_low_ig_E) for col in omics_raw_t_scaled.columns]
    uncor_keep_mask = uncorrelated_features(cancer_type, comment, omics_raw_t_scaled)
    print('uncor', len(uncor_keep_mask), sum(uncor_keep_mask))
    final_mask = np.all([uncor_keep_mask, ig_keep_mask, dup_keep_mask, std_keep_mask],  0)
    np.save(model_path+f'sharon_{cancer_type}_{omics_type}_feature_selector',final_mask)
    with open(model_path+ f'sharon_{cancer_type}_{omics_type}_scaler.pkl', 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(cancer_type, omics_type, comment, len(final_mask), sum(final_mask)) 
    return omics_raw_t_scaled.loc[:,final_mask]


# In[32]:


# feature preprocessing
# def calc_survival_rate(clinical_data):
#         T = clinical_data.apply(calc_time_to_event, axis = 1)
#         E = (clinical_data['vital_status'].values == 'Dead')[T.notnull()].astype(int)
#         return (1 - sum(E)/len(E))
def preprocessing():    
    for cancer_type in cancer_types:
        clinical_data = pd.read_table(f'{train_data_path_prefix}{cancer_type}'+'/clinical')
    #     np.save(model_path+f'sharon_{cancer_type}_survival_plateau', calc_survival_rate(clinical_data))
    #     continue
        process_omics(cancer_type, 'mirna',
                      np.log2(1+pd.read_table(f'{train_data_path_prefix}{cancer_type}'+'/mirna')),
                      clinical_data)
        process_omics(cancer_type, 'exp',
                      np.log2(1+pd.read_table(f'{train_data_path_prefix}{cancer_type}'+'/exp')),
                      clinical_data)
        process_omics(cancer_type, 'methy',
                      pd.read_table(f'{train_data_path_prefix}{cancer_type}'+'/methy'),
                      clinical_data)

                                 


# model building


#submission test


# In[73]:


def preprocessing_feature_count():
    for cancer_type in cancer_types:
        for omics_type in omics_types:        
            _, _ = get_omics(cancer_type, f'{train_data_path_prefix}', omics_type, printStats = True)


# In[81]:


def model_building():
    trainall = False
    for cancer_type in cancer_types:
        #build separate models
        clinical_data = pd.read_table(f'{train_data_path_prefix}{cancer_type}'+'/clinical')
        omicses = []
        predictions = []
        for omics_type in omics_types:        
            omics, _ = get_omics(cancer_type, f'{train_data_path_prefix}', omics_type)
            omicses.append(omics)
            patients, X, E, T, prediction = task1_model(cancer_type, omics_type, clinical_data,omics, trainall = trainall)
            predictions.append(prediction)
        joint = np.concatenate(omicses, axis=1)
        patients,  X, E, T, _ = task1_model(cancer_type, 'joint', clinical_data, joint, trainall=trainall)

        X = (pd.DataFrame(np.array(predictions))).transpose()
        y = np.asarray(list(zip(E,T)),dtype=[('e.tdm', '?'), ('t.tdm', '<f8')])
        cox = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=1e-16, n_alphas = 1, alphas = [1e-16])
        cox_train_idx = np.ones(X.shape[0], dtype=np.bool)
        if trainall == False:
            cox_test_idx = cox_train_idx.copy()
            cox_test_idx[np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)] = False    
            cox_train_idx = ~cox_test_idx


        cox.fit(X[cox_train_idx], y[cox_train_idx])
        print(cancer_type, 'Cox Joint CI: =', cox.score(X[cox_test_idx],y[cox_test_idx]))
        cox.fit(X,y)

        pickle.dump(cox, open(model_path+ f'sharon_{cancer_type}_task1_joint_cox.pkl', 'wb'))

        # Task 2
        task2_predictions = []
        exp_omics, _ = get_omics(cancer_type, f'{train_data_path_prefix}', 'exp')            
        for omics_type in omics_types:
            omics, _ = get_omics(cancer_type, f'{train_data_path_prefix}', omics_type) 
            T = clinical_data.apply(calc_time_to_event, axis = 1)
            if omics_type=='exp':
                with open(model_path+ f'sharon_{cancer_type}_{omics_type}_task1_model_boosting.pkl', 'rb') as file:
                    est = pickle.load(file)  
                    task2_prediction = est.predict(omics[T.notnull()])
    #                 print('task2', cancer_type, omics_type, len(task2_prediction))
                    task2_predictions.append(task2_prediction)
                    continue
            X, T, E, task2_prediction = task2_model(cancer_type, omics_type, clinical_data,exp_omics, trainall = trainall)
    #         print(cancer_type, omics_type, len(task2_prediction))
            task2_predictions.append(task2_prediction)

        X = (pd.DataFrame(np.array(task2_predictions))).transpose()
        y = np.asarray(list(zip(E,T)),dtype=[('e.tdm', '?'), ('t.tdm', '<f8')])
        with open(model_path+ f'sharon_{cancer_type}_task1_joint_cox.pkl', 'rb') as file:
            cox = pickle.load(file) 
        print(cancer_type, 'Cox2 Joint CI: =', cox.score(X,y))

        # task 3
        if cancer_type == 'LAML':
            continue
        task3_predictions = [] 
        stime = time.time()
        T = clinical_data.apply(calc_time_to_event, axis = 1)
        E = (clinical_data.loc[:, 'vital_status'].values == 'Dead')[T.notnull()].astype(int)

        for ct in cancer_types:
            for ot in omics_types:
                omics, _ = get_omics(cancer_type, f'{train_data_path_prefix}', ot, featureHandlingAs = ct) 
    #             print('1', omics.shape, len(T), len(T[T.notnull()]))
                with open(model_path+ f'sharon_{ct}_{ot}_task1_model_boosting.pkl', 'rb') as file:
                    est = pickle.load(file)  
                task3_prediction = est.predict(omics[T.notnull()])
                task3_predictions.append(task3_prediction)
        X = (pd.DataFrame(np.array(task3_predictions))).transpose()
        print('task 3 X.shape', X.shape)
        y = np.asarray(list(zip(E,T[T.notnull()])),dtype=[('e.tdm', '?'), ('t.tdm', '<f8')])

        cox = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.1)#, alphas = [1e-8])#, n_alphas = 3)#, alphas = [1e-16])
        cox_train_idx = np.ones(X.shape[0], dtype=np.bool)
        if trainall == False:
            cox_test_idx = cox_train_idx.copy()
            cox_test_idx[np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)] = False    
            cox_train_idx = ~cox_test_idx

        cox.fit(X[cox_train_idx], y[cox_train_idx])
        print(cancer_type, 'Task 3 Cox Joint CI: =', cox.score(X[cox_test_idx],y[cox_test_idx]))
        cox.fit(X,y)

        pickle.dump(cox, open(model_path+ f'sharon_{cancer_type}_task3_cox.pkl', 'wb'))
        print('task3 time', cancer_type,  time.time()- stime)

    


# In[79]:


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def task2_model(cancer_type, omics_type, clinical_data, 
                omics, create = True, trainall = True,
                comment=''):
    stime = time.time()
    T = clinical_data.apply(calc_time_to_event, axis = 1)
    X = omics.copy()
    X = X[T.notnull()].astype(np.float32)
    E = (clinical_data.loc[:, 'vital_status'].values == 'Dead')[T.notnull()].astype(int)
#     T = T[T.notnull()].astype(int)
    
    train_idx = np.ones(X.shape[0], dtype=np.bool)
    test_idx = np.zeros(X.shape[0], dtype=np.bool)
    if not trainall:
        test_idx = train_idx.copy()
        test_idx[np.random.choice(X.shape[0], int(0.8*X.shape[0]), replace=False)] = False  
        train_idx = ~test_idx
    with open(model_path+ f'sharon_{cancer_type}_{omics_type}_task1_model_boosting.pkl', 'rb') as file:
        est = pickle.load(file) 
    target_omics, _ = get_omics(cancer_type, f'{train_data_path_prefix}', omics_type)    
#     print('test_idx', len(test_idx), sum(test_idx))
#     print('train_idx', len(train_idx), sum(train_idx), X.shape, omics.shape, target_omics.shape)

    y = est.predict(target_omics)[T.notnull()]    
#     print('y', len(y), '[T.notnull()]', len(T[T.notnull()]))
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train = omics[T.notnull()][train_idx].astype(np.float32)
    X_test = omics[T.notnull()][test_idx].astype(np.float32)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    print(cancer_type, omics_type, 'Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    regr.fit(X, y)
    pickle.dump(regr, open(model_path+ f'sharon_{cancer_type}_{omics_type}_task2_from_exp_model_regr.pkl', 'wb'))
#     print(X.shape, len(T[T.notnull()]), len(E))
    print('task2 time', cancer_type, omics_type, time.time()- stime)

    return X, T[T.notnull()], E, regr.predict(X)

    


# In[80]:


num_durations = 50
get_target = lambda df: (df['time_to_event'].values, df['event'].values)
labtrans = LogisticHazard.label_transform(num_durations)

def task1_model(cancer_type, omics_type, clinical_data, 
                omics, create = True, trainall = True,
                comment='', num_durations = 50):
#     print('task1_model:', cancer_type, omics_type, str(trainall))
    stime = time.time()
    train_idx = np.ones(omics.shape[0], dtype=np.bool)
    if not trainall:
        test_idx = train_idx.copy()
        test_idx[np.random.choice(omics.shape[0], int(0.8*omics.shape[0]), replace=False)] = False  
        train_idx = ~test_idx

    X_train = omics[train_idx].copy()
#     in_features = X_train.shape[1]
#     num_nodes = [32, 32]
#     batch_norm = True
#     dropout = 0.1
#     batch_size = 256
#     epochs = 100
#     callbacks = [tt.cb.EarlyStopping()]

    df_train = pd.DataFrame()
    T = clinical_data[train_idx].apply(calc_time_to_event, axis = 1)
    df_train.loc[:,'time_to_event'] = T[T.notnull()].astype(np.int64).values
    df_train.loc[:,'event'] = (clinical_data.loc[train_idx, 'vital_status'].values == 'Dead')[T.notnull()].astype(int)
    y_train = labtrans.fit_transform(*get_target(df_train))
    X_train = X_train[T.notnull()].astype(np.float32)
#     out_features = labtrans.out_features
    T_train = y_train[0]
    E_train = y_train[1]
#     net = torch.nn.Sequential(
#         torch.nn.Linear(in_features, 32),
#         torch.nn.ReLU(),
#         torch.nn.BatchNorm1d(32),
#         torch.nn.Dropout(0.1),
#         torch.nn.Linear(32, 32),
#         torch.nn.ReLU(),
#         torch.nn.BatchNorm1d(32),
#         torch.nn.Dropout(dropout),
#         torch.nn.Linear(32, out_features)
#     )
    
#     print(x_train.shape, y_train[0].shape, y_train[1].shape)
#     model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

#     log = model.fit(torch.tensor(X_train.astype(np.float32)),
#                     (torch.tensor(T_train.astype(np.int64)),
#                      torch.tensor(E_train.astype(np.float32))),
#                     batch_size, epochs, callbacks,  verbose = False)
#     if trainall:
#         pickle.dump(model, 
#                 open(model_path+ f'sharon_{cancer_type}_{omics_type}_task1_model_dill.pt', 'wb'))
    
#         model.save_net(model_path+ f'sharon_{cancer_type}_{omics_type}_task1_model_net.pt')
    

    est = GradientBoostingSurvivalAnalysis(learning_rate=0.8, subsample=0.3, dropout_rate=0.1, random_state=0)
    est.set_params(n_estimators=80)
    try:
        est.fit(X_train, np.asarray(list(zip(E_train,T_train)),dtype=[('e.tdm', '?'), ('t.tdm', '<f8')]))
    except:
        pass


    
    if trainall:
        print('Task1:', model_path+ f'sharon_{cancer_type}_{omics_type}_task1_model_boosting.pkl', X_train.shape )
        pickle.dump(est, open(model_path+ f'sharon_{cancer_type}_{omics_type}_task1_model_boosting.pkl', 'wb'))
        return T_train.index, X_train, E_train, T_train, est.predict(X_train)

    X_test = omics[test_idx].copy()
    T_test = clinical_data[test_idx].apply(calc_time_to_event, axis = 1)
    E_test = (clinical_data.loc[test_idx,'vital_status'].values == 'Dead')[T_test.notnull()].astype(int)
    X_test = X_test[T_test.notnull()].astype(np.float32)
    T_test = T_test[T_test.notnull()].astype(np.int64)
#     ev_test = EvalSurv(model.predict_surv_df(X_test), np.asarray(T_test), np.asarray(E_test), censor_surv='km')
#     print(cancer_type, omics_type, 'NN Test Set CI: =', ev_test.concordance_td('antolini'))
    est = GradientBoostingSurvivalAnalysis(learning_rate=1.0, random_state=0)
    est.set_params(n_estimators=80)
    try:
        est.fit(X_train, np.asarray(list(zip(E_train,T_train)),dtype=[('e.tdm', '?'), ('t.tdm', '<f8')]))
        cindex = est.score(X_test, np.asarray(list(zip(E_test,T_test)),dtype=[('e.tdm', '?'), ('t.tdm', '<f8')]))
        print(cancer_type, omics_type, 'Boosting Test Set CI: =', cindex)
    except:
        pass
    T = clinical_data.apply(calc_time_to_event, axis = 1)
    X = omics.copy()
    X = X[T.notnull()].astype(np.float32)
    E = (clinical_data.loc[:, 'vital_status'].values == 'Dead')[T.notnull()].astype(int)
    T = T[T.notnull()].astype(int)
    print('task1 time', cancer_type, omics_type, time.time()- stime)
    return T.index, X, E, T, est.predict(X)

    


# In[10]:


# if TEST:
#     submission(task_id, cancer_type, held_out_cancer_dir, output_file_path)


# In[88]:


import random, time, joblib
#https://techoverflow.net/2019/11/12/simulating-survival-data-for-kaplan-meier-plots-in-python/
def simulate_T_E(cancer_type, pred):
    survival_plateau = np.load(model_path+f'sharon_{cancer_type}_survival_plateau.npy')
    print(f'Survival rate for {cancer_type} is {survival_plateau}')
    N = len(pred.columns)
    T = np.zeros(N)
    E = np.zeros(N)
    for i in range(N):
        r = random.random()
        if r <= survival_plateau:
            # Event is censoring at the end of the time period
            T[i] = pred.index[-1]
            E[i] = 0
        else: # Event occurs
            # Normalize where we are between 100% and the survival plateau
            p = (r - survival_plateau) / (1 - survival_plateau)
            # Linear model: Time of event linearly depends on uniformly & randomly chosen position
            #  in range (0...tplateau)
            T[i] = p * pred.index[-1]
            E[i] = 1
    return T, E


def get_omics(cancer_type, cancer_dir, omics_type,
              only_exp = False, featureHandlingAs = None, printStats = False):
    try:
        assert os.path.exists(os.path.join(model_path,f'sharon_{cancer_type}_{omics_type}_feature_selector.npy'))
    except:
        print('get_omics: FAILED to find data file', os.path.join(model_path,f'sharon_{cancer_type}_{omics_type}_feature_selector.npy'))
        print('model_path =', model_path)
    try:
        assert os.path.exists(os.path.join(model_path,f'sharon_{cancer_type}_{omics_type}_scaler.pkl'))
    except:
        print('get_omics: FAILED to find data file', os.path.join(model_path,f'sharon_{cancer_type}_{omics_type}_scaler.pkl'))
        print('model_path =', model_path)
        
    if only_exp:
        target_path = cancer_dir
    else:
        target_path = os.path.join(os.path.join(cancer_dir,f'{cancer_type}'), f'{omics_type}')
#     print('get_omics', target_path, only_exp)
    if omics_type in ['exp', 'mirna']:
        data_raw = np.log2(1+pd.read_table(target_path))
    else:
        data_raw = pd.read_table(target_path) 
    ctForFeatureHandling = cancer_type
    if featureHandlingAs is not None:
        ctForFeatureHandling = featureHandlingAs
    feature_selector = np.load(os.path.join(model_path,f'sharon_{ctForFeatureHandling}_{omics_type}_feature_selector.npy'))
    with open( os.path.join(model_path,f'sharon_{ctForFeatureHandling}_{omics_type}_scaler.pkl'), 'rb') as handle:
        scaler = pickle.load(handle)
    assert len(feature_selector)== data_raw.shape[0]
    scaled  = scaler.transform(data_raw.T)
    result = scaled[:,feature_selector.astype(bool)]
    if printStats:
        print('Stats', cancer_type, omics_type, 'raw:', data_raw.shape, 'result:', result.shape  )
    return result.astype(np.float32), data_raw.columns



# will write sharon_task_{task_id}_{cancer_type}.csv
def submission(task_id, cancer_type, held_out_cancer_dir, output_file_path):
    
    assert os.path.isdir(output_file_path)
    if task_id == 2:
        try:
            assert os.path.isfile(held_out_cancer_dir) 
        except:
            print('for task 2 I was expecting a file with exp for the third argument')
            raise
    else:
        try:
            assert os.path.isdir(held_out_cancer_dir)   
        except:
            print('I was expecting a directory where to put the submission as the last argument')
            raise

            
    print(f'Task #: {task_id}  cancer_type:{cancer_type}')
    starttime = time.time()
    print('starting to predict', starttime)
    if task_id == 1:
        predictions = []
        omicses = []
        for omics_type in omics_types:
            print(cancer_type, held_out_cancer_dir, 'working on', omics_type)
            omics, patients = get_omics(cancer_type, held_out_cancer_dir, omics_type)
            print('omics shape', type(omics), omics.shape)
            # load boosting model
            with open(os.path.join(model_path, f'sharon_{cancer_type}_{omics_type}_task1_model_boosting.pkl'), 'rb') as file:
                est = pickle.load(file) 
            predictions.append(est.predict(omics))  
            omicses.append(omics)
        X = (pd.DataFrame(np.array(predictions))).transpose()
        with open(os.path.join(model_path, f'sharon_{cancer_type}_task1_joint_cox.pkl'), 'rb') as file:
            cox = pickle.load(file)        
        cox_hazard = cox.predict(X)
    if task_id == 2:
        exp_file_path = held_out_cancer_dir+'exp'
        if __name__ == "__main__":
            exp_file_path = held_out_cancer_dir
        omics, patients = get_omics(cancer_type, exp_file_path, 'exp', only_exp = True)
        with open(os.path.join(model_path, f'sharon_{cancer_type}_task1_joint_cox.pkl'), 'rb') as file:
            cox = pickle.load(file) 
        task2_predictions = []
        for omics_type in omics_types:
            if omics_type=='exp':
                with open(os.path.join(model_path, f'sharon_{cancer_type}_{omics_type}_task1_model_boosting.pkl'), 'rb') as file:
                    est = pickle.load(file)     
                task2_predictions.append(est.predict(omics))
                continue
            with open(os.path.join(model_path, f'sharon_{cancer_type}_{omics_type}_task2_from_exp_model_regr.pkl'), 'rb') as file:
                regr = pickle.load(file) 
            task2_predictions.append(regr.predict(omics))

        X = (pd.DataFrame(np.array(task2_predictions))).transpose()
        cox_hazard = cox.predict(X) 
        
        
        
    if task_id == 3:
        if cancer_type == 'LAML':
            return None, None

        task3_predictions = []
        for ct in cancer_types:
            for ot in omics_types:
                omics, patients = get_omics(cancer_type, f'{held_out_cancer_dir}', ot , featureHandlingAs = ct) 
                with open(os.path.join(model_path, f'sharon_{ct}_{ot}_task1_model_boosting.pkl'), 'rb') as file:
                    est = pickle.load(file)  
                task3_predictions.append(est.predict(omics))
        X = (pd.DataFrame(np.array(task3_predictions))).transpose()
        with open(os.path.join(model_path, f'sharon_{cancer_type}_task3_cox.pkl'), 'rb') as file:
            cox = pickle.load(file)        
        cox_hazard = cox.predict(X)
    
    
    result_df = pd.DataFrame(columns=['sample_id', 'rank'])
    result_df['sample_id'] = patients
    result_df['rank'] = pd.Series(cox_hazard).rank(method='min').astype(int)
    endtime = time.time()
    print(f'{task_id} {cancer_type}prediction time was', endtime-starttime)
    submission_file = os.path.join(output_file_path, f'sharon_submission_task_{task_id}_{cancer_type}.csv')
    result_df.to_csv(submission_file, index=False, header=True, sep='\t')
    return result_df, cox_hazard 


# In[94]:


# # if TEST:
# task_id = 2
# for cancer_type in cancer_types:
#     result_df, hazard = submission(task_id, cancer_type, held_out_cancer_dir+'BRCA/exp',output_file_path)


# In[77]:


import sys
from typing import List, Any

USAGE = f'Usage: python {sys.argv[0]}  <task_id> <cancer type> <input_file/dir_paths> <output_file_path>'


def execute_submission(args: List[str]):
    # If passed to the command line, need to convert
    # the optional 3rd argument from string to int
    model_path = ""
    assert os.path.exists(model_path+f'sharon_BLCA_exp_feature_selector.npy')

    if len(args) != 4:
        print(USAGE)
        return
    if not args[0].isdigit():
        print(USAGE)
        return
    task_id = int(args[0])
    if task_id == 1 or task_id==2 or task_id == 3:
        cancer_type = args[1]
        if cancer_type in cancer_types:
            if not (task_id == 3 and cancer_type == 'LAML'):
                if (task_id == 2 and os.path.isfile(args[2])) or (task_id != 2 and os.path.isdir(args[2])):
                    if os.path.isdir(args[3]):
                        submission(task_id, cancer_type, args[2], args[3])
                    else:
                        print(f'cannot find folder for submissions {args[3]}')
                        return
                else:
                    if task_id == 2:
                        print(f'cannot find {args[2]}, was expecting a file name with exp')
                    else:
                        print(f'cannot find {args[2]}, was expecting a direcotry with omics')
                    return    
            else:
                print(f' {cancer_type} not legit for task {task_id}')
                return
        else:
            print(f'{cancer_type} must be in {cancer_types}')
            return
                    
                    
def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(USAGE)

    if args[0] == "--help":
        print(USAGE)
    else:
        execute_submission(args)

if __name__ == "__main__":
    model_path = os.getcwd() +'/'
    print('model_path:', model_path)
    main()


# In[ ]:





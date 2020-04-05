
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn import preprocessing
from scipy import stats
from scipy.stats import mstats
from sklearn.preprocessing import FunctionTransformer
from math import ceil
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

import os
cwd = os.path.dirname(os.getcwd()) 

df = pd.read_parquet(r'path')

metric_columns = ['Accuracy-F1','worst_precision_score', 'roc_auc_score',
            'Worst_Class_f1_score','average_precision_score','worst_recall_score','scaling_no',
            'Classifier','Train_Score','Test_Score','Macro_Avg_F1',
            
            'Accuracy-F1_Val','worst_precision_score_Val', 'roc_auc_score_Val',
            'Worst_Class_f1_score_Val','average_precision_score_Val','worst_recall_score_Val','scaling_no_Val',
            'Classifier_Val','Train_Score_Val','Test_Score_Val','Macro_Avg_F1_Val']

df_metrics_lgbm = pd.DataFrame(columns=metric_columns)
df_metrics_xg = pd.DataFrame(columns=metric_columns)
df_metrics_cat = pd.DataFrame(columns=metric_columns)
df_metrics_rf = pd.DataFrame(columns=metric_columns)
df_metrics_DecisionTree = pd.DataFrame(columns=metric_columns)
df_metrcis_knn = pd.DataFrame(columns=metric_columns)
df_metrics_lda = pd.DataFrame(columns=metric_columns)
df_metrics_gnb = pd.DataFrame(columns=metric_columns)
df_metrcis_svm = pd.DataFrame(columns=metric_columns)
df_metrcis_svm_linear = pd.DataFrame(columns=metric_columns)
df_metrcis_svm_polynomial = pd.DataFrame(columns=metric_columns)
df_metrcis_svm_rbf = pd.DataFrame(columns=metric_columns)
df_metrcis_svm_sigmoid = pd.DataFrame(columns=metric_columns)



def drop_cols_including_with_text(df_in: pd.DataFrame, text_list: list):
    '''
    dropColsStartingWithText: drop cols starting with text in text_list
    df: data frame to drop columns
    text_list: potential text list including texts to look for on df
    '''
    for text in text_list:
        df_in = df_in[df_in.columns.drop(list(df_in.filter(regex=text)))]
    return df_in
dropped_columns = ["Min", "Max"]
df = drop_cols_including_with_text(df, dropped_columns)



df = df.select_dtypes(exclude=[object])

target = 'target'
targetlabel = 'target_label'

#Labeling for Classification
df.loc[0.03 < df[target] , targetlabel] = 0
df.loc[(0 <= df[target]) & (df[target] <= 0.03) , targetlabel] = 1

df_nio = df.loc[df[targetlabel] == 0]
df_io_class = df.loc[df[targetlabel] == 1]

#!Validation Frame
frames_validation = [df_nio, df_io_class]
df_validation = pd.concat(frames_validation)
df_validation = df_validation.select_dtypes(exclude=[object])
df_validation = df_validation.fillna(df_validation.mean())
#nunique_val = df_validation.apply(pd.Series.nunique)
#cols_to_drop_val = nunique_val[nunique_val == 1].index

#df_validation=df_validation.drop(cols_to_drop_val, axis=1)

df_validation = df_validation.drop(columns=[ 'Feature1',
                                                            'Feature2'])
df_validation_for_transformation = df_validation.drop(columns=[targetlabel])
df_target_cols_val = df_validation.iloc[:,-1:]
df_target_cols_val = df_target_cols_val.reset_index(drop=True)



#! Take All Without label
df_io_class = df_io_class.drop(columns=[targetlabel])

#!Replace Outlier 
df_io_class = df_io_class.mask(df_io_class.sub(df_io_class.mean()).div(df_io_class.std()).abs().gt(3))
df_io_class = df_io_class.fillna(df.median())

df_io_class[targetlabel] = 1

#! Take All Without label NIO
df_nio = df_nio.drop(columns=[targetlabel])

#!Replace Outlier 
df_nio = df_nio.mask(df_nio.sub(df_nio.mean()).div(df_nio.std()).abs().gt(3))
df_nio = df_nio.fillna(df.median())

df_nio[targetlabel] = 0

#! Class ınbalance must be reduced...
np.random.seed(10)
remove_n = len(df_io_class)-2*len(df_nio)
drop_indices = np.random.choice(df_io_class.index,remove_n,replace=False)
df_io_class = df_io_class.drop(drop_indices)
df_io_class = df_io_class.reset_index(drop=True)


frames = [df_nio, df_io_class]
df_reduced = pd.concat(frames)
df_reduced = df_reduced.sample(frac=1)
df_reduced = df_reduced.reset_index(drop=True)
df = df_reduced


print("Classes:")
print(df[targetlabel].value_counts())
df = df.select_dtypes(exclude=[object])




df_target_cols = df.iloc[:, -2:]
df_for_transformation = df.iloc[:, :-2]



print("Transformation")
print(df_for_transformation)
df_for_transformation = df_for_transformation.fillna(df_for_transformation.mean())



nunique = df_for_transformation.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
df_for_transformation=df_for_transformation.drop(cols_to_drop, axis=1)

df_for_transformation = df_for_transformation.drop(columns=[ 'feature1',
                                                            'feature2'])


df_for_transformation_meta = df_for_transformation.copy()
df_validation_for_transformation_meta = df_validation_for_transformation.copy()


columns_for_transformation = list(df_for_transformation)
columns_validation_for_transformation = list(df_validation_for_transformation)


Scalers1 = {0:MinMaxScaler(),1:Normalizer(),2:QuantileTransformer()}

Scalers2 = {0:FunctionTransformer(np.log1p), 1:StandardScaler(), 2:Normalizer(),
                3:MaxAbsScaler(),4:RobustScaler(),5:QuantileTransformer(),
                6:PowerTransformer(),1:MinMaxScaler()}



metrics_row = 0

for i in range(len(Scalers1)):

    for a in range(len(Scalers2)):
        
        #a = i+1 + a
        if a < len(Scalers2):
            
            first_scaler = Scalers1[i]
            second_scaler = Scalers2[a]
            print("Selected Scalers:")
            print(first_scaler,second_scaler)
            print("Selected Items")
            print(i,a)
           

            scaler1 = first_scaler
            df_for_transformation = scaler1.fit_transform(df_for_transformation)
            
            scaler2 = second_scaler
            df_for_transformation = scaler2.fit_transform(df_for_transformation)

            df_validation_for_transformation = scaler1.fit_transform(df_validation_for_transformation)
            df_validation_for_transformation = scaler2.fit_transform(df_validation_for_transformation)
                

            df_for_transformation = pd.DataFrame(df_for_transformation)
            df_for_transformation.columns = columns_for_transformation
            df_for_transformation = pd.concat([df_for_transformation, df_target_cols], axis=1, sort=False)

            df_validation_for_transformation = pd.DataFrame(df_validation_for_transformation)
            df_validation_for_transformation.columns = columns_validation_for_transformation
            df_validation_for_transformation = pd.concat([df_validation_for_transformation,df_target_cols_val], axis=1, sort=False)





            
            correlation_value =0

            corr = df_for_transformation.corr()
            corr = corr[target].to_frame()
            corr = corr.loc[(corr[target] >= correlation_value) | (corr[target] <= -correlation_value)]
            corr = corr.reset_index()
            corr.columns = ['Correlation_Label', 'Correlation_Value']
            corr['Correlation_Value'] = abs(corr['Correlation_Value'])
            corr = corr.sort_values(by='Correlation_Value',ascending=False)
            corr = corr[:]
            corr = corr.reset_index(drop=True)
            corr_number = 50
            corr = corr.iloc[:corr_number]

            col_for_gauss = corr['Correlation_Label']
            
            df_validation_transformed = df_validation_for_transformation[col_for_gauss]
            df_pair_transformed = df_for_transformation[col_for_gauss]
            df_pair_untransformed = df[col_for_gauss]
            
            #print(df_pair_transformed)
            df_pair_transformed = df_pair_transformed.drop(columns=['PN_155301001_Result_MeasResults_S2SDispersal_RangeS2STabTiCurve2'])
            df_validation_transformed = df_validation_transformed.drop(columns=['PN_155301001_Result_MeasResults_S2SDispersal_RangeS2STabTiCurve2'])
            
            
            df_pair_transformed[:100].to_excel(r'C:\Users\tay5bu\Desktop\df_pair_transformed.xlsx')
            df_validation_transformed[:100].to_excel(r'C:\Users\tay5bu\Desktop\df_validation_transformed.xlsx')

        
            target = 'CvoC_Label'
            #target_labels = np.array(['NOK1','NOK2','OK1','OK2'])
            target_labels = np.array(['NOK','OK'])

            y = df_pair_transformed[target]
            X = df_pair_transformed.drop(columns=[target])
            #y.to_excel(r'C:\Users\tay5bu\Desktop\y.xlsx')
            #X.to_excel(r'C:\Users\tay5bu\Desktop\X.xlsx')

            #y = df_pair_transformed[target]
            #X = df_pair_transformed.drop(columns=[target])
            #poly = PolynomialFeatures(degree=6).fit(X)
            #X = poly.transform(X)
            #print("Get Polynomial Features: ",poly.get_feature_names())

            

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=109)

            #! fit a Randomforest model to the data
            model = RandomForestClassifier(n_jobs=2, random_state=0)
            model.fit(X_train, y_train)
            print(); print(model)
            expected_y  = y_test
            predicted_y = model.predict(X_test)

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)

            print(); print('RandomForest_Validation: ')
            print(); print(metrics.classification_report(expected_y_validation, predicted_y_validation,
                           target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y_validation, predicted_y_validation))



            print(); print('RandomForest: ')
            print(); print(metrics.classification_report(expected_y, predicted_y,
                           target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))

            df_metrics_rf.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrics_rf.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)


            df_metrics_rf.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrics_rf.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrics_rf.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrics_rf.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrics_rf.at[metrics_row,'Classifier'] = 'RandomForest'
            df_metrics_rf.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrics_rf.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrics_rf.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)




            df_metrics_rf.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrics_rf.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_rf.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrics_rf.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrics_rf.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrics_rf.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_rf.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrics_rf.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)



            scaling_no = i*10+a
            df_metrics_rf.at[metrics_row,'scaling_no'] = scaling_no
            df_metrics_rf.scaling_no = df_metrics_rf.scaling_no.astype(float)


            #!LGBM
            params_default_LGBM ={'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':1.0,
                       'importance_type':'split', 'learning_rate':0.05, 'max_depth':-1,
                       'min_child_samples':20, 'min_child_weight':0.001, 'min_split_gain':0.0,
                       'n_estimators':100, 'n_jobs':-1, 'num_leaves':50, 'objective':None,
                       'random_state':None, 'reg_alpha':0.0, 'reg_lambda':0.0, 'silent':True,
                       'subsample':1.0, 'subsample_for_bin':200000, 'subsample_freq':0}

            params_default_LGBM_cv ={'boosting_type':['gbdt'], 'class_weight':[None], 'colsample_bytree':[1.0],
                       'importance_type':['split'], 'learning_rate':[0.1,0.2,0.5,0.8,1], 'max_depth':[-1],
                       'min_child_samples':[20], 'min_child_weight':[0.001], 'min_split_gain':[0.0],
                       'n_estimators':[10,20,60,80], 'n_jobs':[-1], 'num_leaves':[12,30,50], 'objective':[None],
                       'random_state':[None], 'reg_alpha':[0.0], 'reg_lambda':[0.0],
                       'subsample':[1.0], 'subsample_for_bin':[200000], 'subsample_freq':[0],'silent':[True]}

            def lgbm_classifier(X_train,y_train, bIsGridSearch):
            # fit a LightGBM model to the data
                model = LGBMClassifier(**params_default_LGBM)
                model_name = 'alpha_lgbm_class_s2s.h5'
                if bIsGridSearch:
                    gcv = GridSearchCV(model,
                    params_default_LGBM_cv,
                    )
                    gcv.fit(X_train,y_train)
                    model = gcv.best_estimator_
                    joblib.dump(model, model_name, compress = 1)
                    print("Best Params for LGBM")
                    print(model)
                    #model.save_model(model_name)
                else:
                    model = joblib.load(model_name)
                    model.fit(X_train,y_train)

                return model
            bIsGridSearch = True
            model = lgbm_classifier(X_train,y_train, bIsGridSearch)


            # make predictions

            expected_y  = y_test
            predicted_y = model.predict(X_test)


            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)


            print(); print('LightGBM_Validation: ')
            print(); print(metrics.classification_report(expected_y_validation, predicted_y_validation,
                           target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y_validation, predicted_y_validation))



            # summarize the fit of the model
            print(); print('LightGBM: ')
            print(); print(metrics.classification_report(expected_y, predicted_y,
                           target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))
            print("Accuracy-F1:")
            print(metrics.accuracy_score(expected_y,predicted_y))
            print("worst_precision_score:")
            print(metrics.precision_score(expected_y,predicted_y))
            print("roc_auc_score:")
            print(metrics.roc_auc_score(expected_y,predicted_y))
            print("Worst_Class_f1_score:")
            print(metrics.f1_score(expected_y,predicted_y))

            print("average_precision_score:")
            print(metrics.average_precision_score(expected_y,predicted_y))
            print("worst_recall_score:")
            print(metrics.recall_score(expected_y,predicted_y))
            print("Macro_Avg_F1")
            print(metrics.balanced_accuracy_score(expected_y,predicted_y))


            
            df_metrics_lgbm.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrics_lgbm.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)

            df_metrics_lgbm.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrics_lgbm.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrics_lgbm.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrics_lgbm.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrics_lgbm.at[metrics_row,'Classifier'] = 'LGBM'
            df_metrics_lgbm.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrics_lgbm.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrics_lgbm.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)


            df_metrics_lgbm.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrics_lgbm.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_lgbm.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrics_lgbm.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrics_lgbm.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrics_lgbm.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_lgbm.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrics_lgbm.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)







            scaling_no = i*10+a
            df_metrics_lgbm.at[metrics_row,'scaling_no'] = scaling_no
            df_metrics_lgbm.scaling_no = df_metrics_lgbm.scaling_no.astype(float)



            
            params_default_XGB = {'base_score':0.5, 'booster':'gbtree', 'colsample_bylevel':1,
                    'colsample_bynode':1, 'colsample_bytree':1, 'gamma':0,
                    'learning_rate':0.1, 'max_delta_step':0, 'max_depth':3,
                    'min_child_weight':1, 'missing':None, 'n_estimators':100, 'n_jobs':1,
                    'nthread':None, 'objective':'multi:softprob', 'random_state':0,
                    'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'seed':None,
                    'silent':True, 'subsample':1, 'verbosity':1,'num_class' : 2}
            
            #! fit a XGBoost model to the data
            model = XGBClassifier(**params_default_XGB)
            model.fit(X_train, y_train)
            print(); print(model)

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)

            # make predictions
            expected_y  = y_test
            predicted_y = model.predict(X_test)
            # summarize the fit of the model
            print(); print('XGBoost: ')
            print(); print(metrics.classification_report(expected_y, predicted_y,
                           target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))
            

            df_metrics_xg.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrics_xg.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)


            df_metrics_xg.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrics_xg.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrics_xg.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrics_xg.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrics_xg.at[metrics_row,'Classifier'] = 'XGBOOST'
            df_metrics_xg.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrics_xg.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrics_xg.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)

            df_metrics_xg.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrics_xg.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_xg.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrics_xg.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrics_xg.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrics_xg.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_xg.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrics_xg.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)


            scaling_no = i*10+a
            df_metrics_xg.at[metrics_row,'scaling_no'] = scaling_no
            df_metrics_xg.scaling_no = df_metrics_xg.scaling_no.astype(float)




            #! fit a CatBoost model to the data
            model = CatBoostClassifier()
            model.fit(X_train, y_train,silent=True)
            print(); print(model)
            # make predictions
            expected_y  = y_test
            predicted_y = model.predict(X_test)

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)
            # summarize the fit of the model
            print(); print('CatBoost: ')
            print(); print(metrics.classification_report(expected_y, predicted_y,
                           target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))


            df_metrics_cat.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrics_cat.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)


            df_metrics_cat.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrics_cat.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrics_cat.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrics_cat.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrics_cat.at[metrics_row,'Classifier'] = 'CATBOOST'
            df_metrics_cat.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrics_cat.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrics_cat.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)



            df_metrics_cat.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrics_cat.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_cat.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrics_cat.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrics_cat.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrics_cat.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_cat.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrics_cat.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)



            scaling_no = i*10+a
            df_metrics_cat.at[metrics_row,'scaling_no'] = scaling_no
            df_metrics_cat.scaling_no = df_metrics_cat.scaling_no.astype(float)




            #! fit a DecisionTree
            model = DecisionTreeClassifier().fit(X_train, y_train)
            expected_y  = y_test
            predicted_y = model.predict(X_test)
            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))
            print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
            
            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)

            df_metrics_DecisionTree.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrics_DecisionTree.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)

            df_metrics_DecisionTree.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrics_DecisionTree.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrics_DecisionTree.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrics_DecisionTree.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrics_DecisionTree.at[metrics_row,'Classifier'] = 'DecisionTree'
            df_metrics_DecisionTree.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrics_DecisionTree.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrics_DecisionTree.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)



            df_metrics_DecisionTree.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrics_DecisionTree.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_DecisionTree.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrics_DecisionTree.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrics_DecisionTree.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrics_DecisionTree.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_DecisionTree.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrics_DecisionTree.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)


            scaling_no = i*10+a
            df_metrics_DecisionTree.at[metrics_row,'scaling_no'] = scaling_no
            df_metrics_DecisionTree.scaling_no = df_metrics_DecisionTree.scaling_no.astype(float)



            #! K-Nearest Neighbors
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)

            expected_y  = y_test
            predicted_y = model.predict(X_test)


            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)

            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))

            print('Accuracy of K-NN classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of K-NN classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

            df_metrcis_knn.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrcis_knn.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)


            df_metrcis_knn.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrcis_knn.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrcis_knn.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrcis_knn.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrcis_knn.at[metrics_row,'Classifier'] = 'KNN'
            df_metrcis_knn.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrcis_knn.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrcis_knn.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)




            df_metrcis_knn.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrcis_knn.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_knn.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrcis_knn.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrcis_knn.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrcis_knn.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_knn.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrcis_knn.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)



            scaling_no = i*10+a
            df_metrcis_knn.at[metrics_row,'scaling_no'] = scaling_no
            df_metrcis_knn.scaling_no = df_metrcis_knn.scaling_no.astype(float)


            #!Linear Discriminant Analysis
            model = LinearDiscriminantAnalysis()
            model.fit(X_train, y_train)

            expected_y  = y_test
            predicted_y = model.predict(X_test)
            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)

            print('Accuracy of LDA classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of LDA classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
            

            df_metrics_lda.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrics_lda.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)


            df_metrics_lda.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrics_lda.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrics_lda.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrics_lda.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrics_lda.at[metrics_row,'Classifier'] = 'LDA'
            df_metrics_lda.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrics_lda.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrics_lda.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)



            df_metrics_lda.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrics_lda.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_lda.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrics_lda.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrics_lda.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrics_lda.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_lda.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrics_lda.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)



            scaling_no = i*10+a
            df_metrics_lda.at[metrics_row,'scaling_no'] = scaling_no
            df_metrics_lda.scaling_no = df_metrics_lda.scaling_no.astype(float)


            
            #!Gaussian Naive Bayes
            model = GaussianNB()
            model.fit(X_train, y_train)

            expected_y  = y_test
            predicted_y = model.predict(X_test)
            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))

            print('Accuracy of GNB classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of GNB classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)


            df_metrics_gnb.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrics_gnb.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)

            df_metrics_gnb.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrics_gnb.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrics_gnb.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrics_gnb.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrics_gnb.at[metrics_row,'Classifier'] = 'GNB'
            df_metrics_gnb.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrics_gnb.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrics_gnb.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)
            


            df_metrics_gnb.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrics_gnb.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_gnb.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrics_gnb.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrics_gnb.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrics_gnb.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrics_gnb.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrics_gnb.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)


            scaling_no = i*10+a
            df_metrics_gnb.at[metrics_row,'scaling_no'] = scaling_no
            df_metrics_gnb.scaling_no = df_metrics_gnb.scaling_no.astype(float)


            #!Support Vector Machine
            model = SVC()
            model.fit(X_train, y_train)

            expected_y  = y_test
            predicted_y = model.predict(X_test)
            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)

            print('Accuracy of SVM classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of SVM classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

            

            df_metrcis_svm.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrcis_svm.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)

            df_metrcis_svm.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrcis_svm.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrcis_svm.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrcis_svm.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrcis_svm.at[metrics_row,'Classifier'] = 'SVM'
            df_metrcis_svm.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrcis_svm.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrcis_svm.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)


            df_metrcis_svm.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrcis_svm.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)



            scaling_no = i*10+a
            df_metrcis_svm.at[metrics_row,'scaling_no'] = scaling_no
            df_metrcis_svm.scaling_no = df_metrcis_svm.scaling_no.astype(float)




            #! Support Vector Machine Linear
            model = SVC(kernel='linear')
            model.fit(X_train,y_train)

            expected_y = y_test
            predicted_y = model.predict(X_test)

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)

            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))


            print('Accuracy of SVM_Linear classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of SVM_Linear classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))


            df_metrcis_svm_linear.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrcis_svm_linear.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)

            df_metrcis_svm_linear.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrcis_svm_linear.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrcis_svm_linear.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrcis_svm_linear.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrcis_svm_linear.at[metrics_row,'Classifier'] = 'SVM_Linear'
            df_metrcis_svm_linear.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrcis_svm_linear.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrcis_svm_linear.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)


            df_metrcis_svm_linear.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_linear.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_linear.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_linear.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_linear.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrcis_svm_linear.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_linear.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_linear.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)




            scaling_no = i*10+a
            df_metrcis_svm_linear.at[metrics_row,'scaling_no'] = scaling_no
            df_metrcis_svm_linear.scaling_no = df_metrcis_svm_linear.scaling_no.astype(float)
            
            #! Support Vector Machine Polynomial

            model = SVC(kernel='poly',degree=3) #degree default 3
            model.fit(X_train,y_train)

            expected_y = y_test
            predicted_y = model.predict(X_test)

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)

            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))


            print('Accuracy of SVM_Polynomial classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of SVM_Polynomial classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))


            df_metrcis_svm_polynomial.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrcis_svm_polynomial.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)

            df_metrcis_svm_polynomial.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrcis_svm_polynomial.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrcis_svm_polynomial.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrcis_svm_polynomial.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrcis_svm_polynomial.at[metrics_row,'Classifier'] = 'SVM_Polynomial'
            df_metrcis_svm_polynomial.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrcis_svm_polynomial.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrcis_svm_polynomial.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)


            df_metrcis_svm_polynomial.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_polynomial.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_polynomial.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_polynomial.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_polynomial.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrcis_svm_polynomial.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_polynomial.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_polynomial.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)



            scaling_no = i*10+a
            df_metrcis_svm_polynomial.at[metrics_row,'scaling_no'] = scaling_no
            df_metrcis_svm_polynomial.scaling_no = df_metrcis_svm_polynomial.scaling_no.astype(float)

            
            #! Support Vector Machine Radial Basis

            model = SVC(kernel='rbf',random_state=0, gamma=10, C=1)
            model.fit(X_train,y_train)

            expected_y = y_test
            predicted_y = model.predict(X_test)


            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)


            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))


            print('Accuracy of SVM_Radial classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of SVM_Radial classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))


            df_metrcis_svm_rbf.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrcis_svm_rbf.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)

            df_metrcis_svm_rbf.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrcis_svm_rbf.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrcis_svm_rbf.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrcis_svm_rbf.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrcis_svm_rbf.at[metrics_row,'Classifier'] = 'SVM_Radial'
            df_metrcis_svm_rbf.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrcis_svm_rbf.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrcis_svm_rbf.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)


            df_metrcis_svm_rbf.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_rbf.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_rbf.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_rbf.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_rbf.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrcis_svm_rbf.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_rbf.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_rbf.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)





            scaling_no = i*10+a
            df_metrcis_svm_rbf.at[metrics_row,'scaling_no'] = scaling_no
            df_metrcis_svm_rbf.scaling_no = df_metrcis_svm_rbf.scaling_no.astype(float)


            #! Support Vector Machine Sigmoid

            model = SVC(kernel='sigmoid')
            model.fit(X_train,y_train)

            expected_y = y_test
            predicted_y = model.predict(X_test)

            expected_y_validation =  df_validation_transformed[target]

            df_validation_transformed_inputs = df_validation_transformed.drop(columns=[target])
            predicted_y_validation = model.predict(df_validation_transformed_inputs)


            print(); print(metrics.classification_report(expected_y, predicted_y,target_names=target_labels))
            print(); print(metrics.confusion_matrix(expected_y, predicted_y))


            print('Accuracy of SVM_Sigmoid classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
            print('Accuracy of SVM_Sigmoid classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))


            df_metrcis_svm_sigmoid.at[metrics_row,'Train_Score'] = model.score(X_train, y_train)
            df_metrcis_svm_sigmoid.at[metrics_row,'Test_Score'] = model.score(X_test, y_test)

            df_metrcis_svm_sigmoid.at[metrics_row,'Accuracy-F1'] = metrics.accuracy_score(expected_y,predicted_y)
            df_metrcis_svm_sigmoid.at[metrics_row,'worst_precision_score'] = metrics.precision_score(expected_y,predicted_y)
            df_metrcis_svm_sigmoid.at[metrics_row,'roc_auc_score'] = metrics.roc_auc_score(expected_y,predicted_y)
            df_metrcis_svm_sigmoid.at[metrics_row,'Worst_Class_f1_score'] = metrics.f1_score(expected_y,predicted_y)
            df_metrcis_svm_sigmoid.at[metrics_row,'Classifier'] = 'SVM_Sigmoid'
            df_metrcis_svm_sigmoid.at[metrics_row,'average_precision_score'] = metrics.average_precision_score(expected_y,predicted_y)
            df_metrcis_svm_sigmoid.at[metrics_row,'worst_recall_score'] = metrics.recall_score(expected_y,predicted_y)
            df_metrcis_svm_sigmoid.at[metrics_row,'Macro_Avg_F1'] = metrics.balanced_accuracy_score(expected_y,predicted_y)



            df_metrcis_svm_sigmoid.at[metrics_row,'Accuracy-F1_Val'] = metrics.accuracy_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_sigmoid.at[metrics_row,'worst_precision_score_Val'] = metrics.precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_sigmoid.at[metrics_row,'roc_auc_score_Val'] = metrics.roc_auc_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_sigmoid.at[metrics_row,'Worst_Class_f1_score_Val'] = metrics.f1_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_sigmoid.at[metrics_row,'Classifier_Val'] = 'LGBM'
            df_metrcis_svm_sigmoid.at[metrics_row,'average_precision_score_Val'] = metrics.average_precision_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_sigmoid.at[metrics_row,'worst_recall_score_Val'] = metrics.recall_score(expected_y_validation,predicted_y_validation)
            df_metrcis_svm_sigmoid.at[metrics_row,'Macro_Avg_F1_Val'] = metrics.balanced_accuracy_score(expected_y_validation,predicted_y_validation)



            scaling_no = i*10+a
            df_metrcis_svm_rbf.at[metrics_row,'scaling_no'] = scaling_no
            df_metrcis_svm_rbf.scaling_no = df_metrcis_svm_rbf.scaling_no.astype(float)





            df_for_transformation = df_for_transformation_meta.copy()
            df_validation_for_transformation = df_validation_for_transformation_meta.copy()
            metrics_row = metrics_row + 1
            

metric_frames = [df_metrics_lgbm, df_metrics_xg,
                 df_metrics_cat,df_metrics_rf,df_metrics_DecisionTree,df_metrcis_knn,
                 df_metrics_lda,df_metrics_gnb,df_metrcis_svm,df_metrcis_svm_linear,
                 df_metrcis_svm_polynomial,df_metrcis_svm_rbf,df_metrcis_svm_sigmoid]
print(df_metrics_lgbm)
print(df_metrics_xg)
print(df_metrics_cat)
print(df_metrics_rf)

df_metrics = pd.concat(metric_frames)
df_metrics_classifier = df_metrics['Classifier']
df_metrics = df_metrics.drop(columns='Classifier')
df_metrics = df_metrics.apply(pd.to_numeric)
df_metrics = pd.concat([df_metrics,df_metrics_classifier],axis=1,sort=False)
df_metrics.to_excel(r'C:\Users\tay5bu\Desktop\Df_metrics.xlsx')
print(df_metrics)



sns.lineplot(x="scaling_no", y="Accuracy-F1",data=df_metrics,hue='Classifier')
plt.show()
sns.lineplot(x="scaling_no", y="Worst_Class_f1_score",data=df_metrics,hue='Classifier')
plt.show()
sns.lineplot(x="scaling_no", y="Macro_Avg_F1", data=df_metrics, hue="Classifier")
plt.show()

#sns.lineplot(x="scaling_no", y='Test_Score',data=df_metrics,hue='Classifier')
#sns.lineplot(x="scaling_no", y='Train_Score',data=df_metrics,hue='Classifier')


f, axes = plt.subplots(1, 2,sharey=True)

sns.lineplot(x="scaling_no", y='Train_Score',data=df_metrics,hue='Classifier', ax=axes[0])
sns.lineplot(x="scaling_no", y='Test_Score',data=df_metrics,hue='Classifier', ax=axes[1])
plt.show()


print("Accuracy-F1 Max: ",df_metrics['Accuracy-F1'].max())
print("Accuracy-F1 Mean: ",df_metrics['Accuracy-F1'].mean())
print("Accuracy-F1 Min:  ",df_metrics['Accuracy-F1'].min())


print("Worst_Class_f1_score Max: ",df_metrics['Worst_Class_f1_score'].max())
print("Worst_Class_f1_score Mean: ",df_metrics['Worst_Class_f1_score'].mean())
print("Worst_Class_f1_score Min:  ",df_metrics['Worst_Class_f1_score'].min())

print("Macro_Avg_F1 Max: ",df_metrics['Macro_Avg_F1'].max())
print("Macro_Avg_F1 Mean: ",df_metrics['Macro_Avg_F1'].mean())
print("Macro_Avg_F1 Min:  ",df_metrics['Macro_Avg_F1'].min())


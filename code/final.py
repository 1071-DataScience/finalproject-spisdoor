import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np
import math
import datetime
import gc

### load dataset ###
dtypes = {
  'order_id': 'object',
  'group_id': 'object'
}

df_airline = pd.read_csv('../data/dataset/airline.csv', dtype=dtypes)
df_cache_map = pd.read_csv('../data/dataset/cache_map.csv', dtype=dtypes)
df_day_schedule = pd.read_csv('../data/dataset/day_schedule.csv', dtype=dtypes)
df_group = pd.read_csv('../data/dataset/group.csv', dtype=dtypes)
df_order = pd.read_csv('../data/dataset/order.csv', dtype=dtypes)
df_test = pd.read_csv('../data/testing-set.csv', dtype=dtypes)
df_train = pd.read_csv('../data/training-set.csv', dtype=dtypes)

### merge dataset except train and test ###
df_total = df_order.merge(df_group, on='group_id')

go_back = df_airline.groupby("go_back")
df_go = go_back.get_group("去程")
df_back = go_back.get_group("回程")
df_go.columns = ['go_' + str(col) for col in df_go.columns]
df_back.columns = ['back_' + str(col) for col in df_back.columns]
df_go = df_go.drop_duplicates(subset=['go_group_id', 'go_go_back'], keep='first')
df_back = df_back.drop_duplicates(subset=['back_group_id', 'back_go_back'], keep='last')
df_new_airline = pd.merge(df_go, df_back, left_on=['go_group_id'], right_on=['back_group_id'])
df_total = df_total.merge(df_new_airline, how='left', left_on='group_id', right_on='go_group_id')

# df_total = df_total.dropna(how='any')
# print(df_total.info())
# print(df_total.isnull().values.any())
# print(df_total.columns)
# exit()

### data engineering ###
month_mapping = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

def convert_date_formate(x):
  if (str(x) == 'nan'):
    return pd.to_datetime('1970-1-1')
  else:
    year = x[-2: ]
    month = month_mapping[x[-6: -3]]
    day = x[: -7]
    return pd.to_datetime('20' + year + '-' + month + '-'+ day)

def convert_date_formate_airline(x):
  if (str(x) == 'nan'):
    return pd.to_datetime('1970-1-1')
  else:
    date = str(x).split(' ')
    date_info = date[0].split('/')
    return pd.to_datetime(str(date_info[0]) + '-' + str(date_info[1]) + '-'+ str(date_info[2]))

def convert_datetime_to_sec(x):
  date_str = str(x).split(' ')[0]
  time_str = str(x).split(' ')[1]
  date_info = date_str.split('-')
  time_info = time_str.split(':')
  t1 = datetime.datetime(int(date_info[0]), int(date_info[1]), int(date_info[2]))
  t2 = int(time_info[0]) * 60 * 60 + int(time_info[1]) * 60 + int(time_info[2])
  return (t1 - datetime.datetime(1970, 1, 1)).total_seconds()  + t2

# airline data
# group_id,go_back,fly_time,src_airport,arrive_time,dst_airport

# group data
# group_id,sub_line,area,days,begin_date,price,product_name,promotion_prog
# 'group_id', 'sub_line', 'area', 'days', 'begin_date', 'price', 'product_name', 'promotion_prog'
df_total['begin_date'] = df_total['begin_date'].apply(lambda x: convert_date_formate(x))
df_total['begin_date_year'] = pd.to_datetime(df_total['begin_date']).dt.year
df_total['begin_date_month'] = pd.to_datetime(df_total['begin_date']).dt.month
df_total['begin_date_day'] = pd.to_datetime(df_total['begin_date']).dt.day
df_total = df_total.merge(df_total[['area', 'begin_date_month']][['area', 'begin_date_month']].groupby(['area', 'begin_date_month']).size().rename('area_begin_date_month').to_frame().reset_index(),
                          on=['area', 'begin_date_month'], how='left')
df_total = df_total.join(pd.get_dummies(df_total['sub_line']))
df_total = df_total.join(pd.get_dummies(df_total['area']))
df_total['price_per_day'] = df_total['price'] / df_total['days']
df_total = df_total.merge(df_total[['group_id']][['group_id']].groupby(['group_id']).size().rename('group_id_count').to_frame().reset_index(),
                          on=['group_id'], how='left')

# order data
# order_id,group_id,order_date,source_1,source_2,unit,people_amount
# 'order_id', 'group_id', 'order_date', 'source_1', 'source_2', 'unit', 'people_amount'
df_total['order_date'] = df_total['order_date'].apply(lambda x: convert_date_formate(x))
df_total['order_date_year'] = pd.to_datetime(df_total['order_date']).dt.year
df_total['order_date_month'] = pd.to_datetime(df_total['order_date']).dt.month
df_total['order_date_day'] = pd.to_datetime(df_total['order_date']).dt.day

df_total = df_total.merge(df_total[['source_1', 'source_2']][['source_1', 'source_2']].groupby(['source_1', 'source_2']).size().rename('source_1_source_2').to_frame().reset_index(),
                          on=['source_1', 'source_2'], how='left')
df_total = df_total.merge(df_total[['source_1', 'source_2', 'unit']][['source_1', 'source_2', 'unit']].groupby(['source_1', 'source_2', 'unit']).size().rename('product_name_source_1_source_2_unit').to_frame().reset_index(),
                          on=['source_1', 'source_2', 'unit'], how='left')

def is_four_people(x):
  if (int(x) == 4):
    return 0
  else:
    return 1

df_total['people_is_4'] = df_total['people_amount'].apply(lambda x: is_four_people(x))
df_total = df_total.join(pd.get_dummies(df_total['source_1']))
df_total = df_total.join(pd.get_dummies(df_total['source_2']))
df_total = df_total.join(pd.get_dummies(df_total['unit']))
# exit()
# merge data
df_total['pre_days']=(df_total['begin_date'] - df_total['order_date']).dt.days
df_total['begin_date_weekday']= df_total['begin_date'].dt.dayofweek
df_total['order_date_weekday']= df_total['order_date'].dt.dayofweek
df_total['return_date_weekday']= (df_total['begin_date'].dt.dayofweek + df_total['days']) % 7

# df_total['begin_date'].fillna(value=pd.to_datetime('1970-1-1'), inplace=True)
# df_total['order_date'].fillna(value=pd.to_datetime('1970-1-1'), inplace=True)

df_total['begin_date'] = df_total['begin_date'].apply(lambda x: convert_datetime_to_sec(x))
df_total['order_date'] = df_total['order_date'].apply(lambda x: convert_datetime_to_sec(x))

df_total['go_src_airport'].fillna(value='none', inplace=True)
df_total['go_dst_airport'].fillna(value='none', inplace=True)
df_total['back_src_airport'].fillna(value='none', inplace=True)
df_total['back_dst_airport'].fillna(value='none', inplace=True)

df_total = df_total.join(pd.get_dummies(df_total['go_src_airport'], prefix='go_src'))
df_total = df_total.join(pd.get_dummies(df_total['go_dst_airport'], prefix='go_dst'))
df_total = df_total.join(pd.get_dummies(df_total['back_src_airport'], prefix='back_src'))
df_total = df_total.join(pd.get_dummies(df_total['back_dst_airport'], prefix='back_dst'))

df_total['go_fly_time'].fillna(value='1970/1/1', inplace=True)
df_total['go_arrive_time'].fillna(value='1970/1/1', inplace=True)
df_total['back_fly_time'].fillna(value='1970/1/1', inplace=True)
df_total['back_arrive_time'].fillna(value='1970/1/1', inplace=True)

df_total['go_fly_time'] = df_total['go_fly_time'].apply(lambda x: convert_date_formate_airline(x))
df_total['go_arrive_time'] = df_total['go_arrive_time'].apply(lambda x: convert_date_formate_airline(x))
df_total['back_fly_time'] = df_total['back_fly_time'].apply(lambda x: convert_date_formate_airline(x))
df_total['back_arrive_time'] = df_total['back_arrive_time'].apply(lambda x: convert_date_formate_airline(x))

df_total['go_arrive_time_weekday']= df_total['go_arrive_time'].dt.dayofweek
df_total['go_fly_time_weekday']= df_total['go_fly_time'].dt.dayofweek
df_total['back_arrive_time_weekday']= df_total['back_arrive_time'].dt.dayofweek
df_total['back_fly_time_weekday']= df_total['back_fly_time'].dt.dayofweek

df_total['go_days']=(df_total['go_arrive_time'] - df_total['go_fly_time']).dt.days
df_total['back_days']=(df_total['back_arrive_time'] - df_total['back_fly_time']).dt.days
df_total['airline_days']=(df_total['back_arrive_time'] - df_total['go_fly_time']).dt.days

df_total['go_fly_time'] = df_total['go_fly_time'].apply(lambda x: convert_datetime_to_sec(x))
df_total['go_arrive_time'] = df_total['go_arrive_time'].apply(lambda x: convert_datetime_to_sec(x))
df_total['back_fly_time'] = df_total['back_fly_time'].apply(lambda x: convert_datetime_to_sec(x))
df_total['back_arrive_time'] = df_total['back_arrive_time'].apply(lambda x: convert_datetime_to_sec(x))
# df_total['begin_date'] = df_total['begin_date'].apply(lambda x: convert_datetime_to_sec(x))
# df_total['order_date'] = df_total['order_date'].apply(lambda x: convert_datetime_to_sec(x))

def get_name_length(x):
  if (str(x) == 'nan'):
    return 0
  else:
    return len(x)

df_total['product_name'] = df_total['product_name'].apply(lambda x: get_name_length(x))

def has_picture(x):
  if (str(x) == 'nan'):
    return 0
  else:
    if (x.find('src') != -1):
      return 1
    else:
      return 0

df_total['promotion_prog'] = df_total['promotion_prog'].apply(lambda x: has_picture(x))

# drop columns
df_total.drop('source_1', axis=1, inplace=True)
df_total.drop('source_2', axis=1, inplace=True)
df_total.drop('unit', axis=1, inplace=True)
df_total.drop('sub_line', axis=1, inplace=True)
df_total.drop('area', axis=1, inplace=True)

df_total.drop('go_group_id', axis=1, inplace=True)
df_total.drop('back_group_id', axis=1, inplace=True)
df_total.drop('go_go_back', axis=1, inplace=True)
df_total.drop('back_go_back', axis=1, inplace=True)

df_total.drop('go_src_airport', axis=1, inplace=True)
df_total.drop('go_dst_airport', axis=1, inplace=True)
df_total.drop('back_src_airport', axis=1, inplace=True)
df_total.drop('back_dst_airport', axis=1, inplace=True)

# print(df_total.columns)
# exit()

### merge data to train and test ###
# print(df_train.shape)
df_train_1 = df_train.merge(df_total, on='order_id')
# print(df_train_1.shape)

# print(df_test.shape)
df_test_1 = df_test.merge(df_total, how='left', on='order_id')
# print(df_test_1.shape)

# columns = df_train_1.columns
# percent_missing = df_train_1.isnull().sum() * 100 / len(df_train_1)
# missing_value_df = pd.DataFrame({'column_name': columns, 'percent_missing': percent_missing})
# print(missing_value_df)

# columns = df_test_1.columns
# percent_missing = df_test_1.isnull().sum() * 100 / len(df_test_1)
# missing_value_df = pd.DataFrame({'column_name': columns, 'percent_missing': percent_missing})
# print(missing_value_df)

# df_test_1.to_csv('test.csv', index=False)
# exit()

### model building ###
model_tune_final = None
# random guess
# from sklearn import metrics
# preds = np.random.randn(*df_train_1['deal_or_not'].shape)
# print(metrics.roc_auc_score(df_train_1['deal_or_not'], preds))
# exit()

# cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def n_fold_cross_validation(X, y, model, N):
  global model_tune_final
  mean_auc = 0
  for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i*42)
    param_grid = {
      # 'n_estimators': [100],
      'n_estimators': [50, 100, 150],
    }
    model_tune = GridSearchCV(model, param_grid=param_grid, n_jobs=1, cv=2)
    model_tune.fit(X_train, y_train)
    preds = model_tune.predict_proba(X_test)[:,1]
    auc = metrics.roc_auc_score(y_test, preds)
    print('AUC (fold %d/%d): %f' % (i + 1, N, auc))
    mean_auc += auc
    model_tune_final = model_tune
  return mean_auc / N

feats = [f for f in df_train_1.columns if f not in ['order_id','deal_or_not','group_id']]

# model
import xgboost as xgb
model = xgb.XGBClassifier(
    learning_rate=0.03,
    max_depth=9,
    nthread=50,
    seed=1,
    n_estimators=100,
    silent=1
)
param = {
  'learning_rate': 0.03,
  'max_depth': 9,
  'nthread': 50,
  'seed': 1,
  'n_estimators': 100,
  'silent': 1
}
print(n_fold_cross_validation(df_train_1[feats], df_train_1['deal_or_not'], model, 5))

# from sklearn.ensemble import GradientBoostingClassifier
# model = GradientBoostingClassifier()
# param = {
#   'n_estimators': 10,
# }
# print(n_fold_cross_validation(df_train_1[feats], df_train_1['deal_or_not'], model, 5))

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# param = {
#   'n_estimators': 10,
# }
# print(n_fold_cross_validation(df_train_1[feats], df_train_1['deal_or_not'], model, 5))

# exit()
def n_fold_cross_validation_lgbm(X, y, model, params, N):
  global model_tune_final
  mean_auc = 0
  for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i*42)
    # model_tune = GridSearchCV(model, n_jobs=1, cv=2)
    lgb_train = lgb.Dataset(X_train, y_train, feature_name = "auto")
    lgb_eval = lgb.Dataset(X_test, y_test, feature_name = "auto")
    model_tune = model.train(
      params=params,
      train_set=lgb_train,
      num_boost_round=10000,
      valid_sets=lgb_eval,
      verbose_eval=50,
      early_stopping_rounds=20
    )
    # model.fit(lgb_train, eval_set=[(lgb_eval)], eval_metric='l1', early_stopping_rounds=20)
    preds = model_tune.predict(X_test, num_iteration=model_tune.best_iteration)
    auc = metrics.roc_auc_score(y_test, preds)
    print('AUC (fold %d/%d): %f' % (i + 1, N, auc))
    mean_auc += auc
    model_tune_final = model_tune
  return mean_auc / N

# import lightgbm as lgb
# model = lgb.LGBMRegressor(
#   nthread=32,
#   boosting_type='dart',
#   objective='binary',
#   metric='auc',
#   learning_rate=0.01,
#   num_leaves=70,
#   max_depth=9,
#   subsample=1,
#   feature_fraction=0.9,
#   colsample_bytree=0.08,
#   min_split_gain=0.09,
#   min_child_weight=9.5,
#   #'reg_alpha': 1,
#   #'reg_lambda': 50,
#   verbose=1,
#   # parameters for dart
#   drop_rate=0.7,
#   skip_drop=0.7,
#   max_drop=5,
#   uniform_drop=False,
#   xgboost_dart_mode=True,
#   drop_seed=4
# )
# params = {
#   'nthread': 32,
#   'boosting_type': 'dart',
#   'objective': 'binary',
#   'metric': 'auc',
#   'learning_rate': 0.01,
#   'num_leaves': 70,
#   'max_depth': 9,
#   'subsample': 1,
#   'feature_fraction': 0.9,
#   'colsample_bytree': 0.08,
#   'min_split_gain': 0.09,
#   'min_child_weight': 9.5,
#   #'reg_alpha': 1,
#   #'reg_lambda': 50,
#   'verbose': 1,
#   # parameters for dart
#   'drop_rate':0.7,
#   'skip_drop':0.7,
#   'max_drop':5,
#   'uniform_drop':False,
#   'xgboost_dart_mode':True,
#   'drop_seed':4
#  }

# print(n_fold_cross_validation_lgbm(df_train_1[feats], df_train_1['deal_or_not'], lgb, params, 5))

# exit()

### output file ###
# target = model.predict(df_test_1[feats], num_iteration=model.best_iteration)
target = model_tune_final.predict_proba(df_test_1[feats])[:,1]
final = pd.DataFrame({'order_id': df_test_1['order_id'], 'deal_or_not': target})
final.to_csv('../results/submit_file.csv', index=False)

import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

param = {
    'bagging_freq': 5,  #handling overfitting
    'bagging_fraction': 0.5,  #handling overfitting - adding some noise
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05, #handling overfitting
    'learning_rate': 0.01,  #the changes between one auc and a better one gets really small thus a small learning rate performs better
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 50,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 10,
    'num_threads': 5,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}

def create_model(train, test, target):
  folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=17)

  oof = np.zeros(len(train))

  #for predictions
  predictions = np.zeros(len(test))
  
  for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    X_train, y_train = train.iloc[trn_idx], target.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx], target.iloc[val_idx]
    
    print("Computing Fold {}".format(fold_))
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    
    num_round = 5000 
    verbose=1000 
    stop=500 
    
    #TRAIN THE MODEL
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=verbose, early_stopping_rounds = stop)
    
    #CALCULATE PREDICTION FOR VALIDATION SET
    oof[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
    
    #CALCULATE PREDICTIONS FOR TEST DATA, using best_iteration on the fold
    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
    
  #print overall cross-validation score
  print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
  
  return predictions
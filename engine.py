import numpy as np
import lightgbm as lightgbm
from metrics import amex_metric, lightgbm_amex_metric
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import logging 

logger = logging.getLogger(__name__)

def train_test(train, test,cfg):
    cat_cols = cfg.cat_features
    cat_cols = [f"{col}_last" for col in cat_cols]
    for col in cat_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    float_cols = train.select_dtypes(include=['float']).columns
    float_cols = [col for col in float_cols if 'last' in col]
    train[float_cols] = train[float_cols].round(2)
    test[float_cols] = test[float_cols].round(2)
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    for col in num_cols:
        train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
        test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
    float_cols = train.select_dtypes(include=['float']).columns
    train[float_cols] = train[float_cols].astype(np.float16)
    test[float_cols] = test[float_cols].astype(np.float16)
    features = [col for col in train.columns if col not in ['customer_ID', cfg.target]]
    params = cfg.params 
    test_predictions = np.zeros(len(test))
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[cfg.target])):
        logger.info(f'Fold:{fold} ...')
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[cfg.target].iloc[trn_ind], train[cfg.target].iloc[val_ind]
        lightgbm_train = lightgbm.Dataset(x_train, y_train, categorical_feature=cat_cols)
        lightgbm_val = lightgbm.Dataset(x_val, y_val, categorical_feature=cat_cols)
        model = lightgbm.train(params, lightgbm_train, valid_sets=[lightgbm_train, lightgbm_val],
                          valid_names=['train', 'val'], num_boost_round=1000,
                          early_stopping_rounds=50, verbose_eval=50,
                          feval=lightgbm_amex_metric)
        oof_predictions[val_ind] = model.predict(x_val)
        test_predictions += model.predict(test[features]) / cfg.n_folds
        score = amex_metric(y_val,model.predict(x_val))
    score = amex_metric(train[cfg.target], oof_predictions)
    logger.info(f"OOF score: {score}")
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
    return test_df
    
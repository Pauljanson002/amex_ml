import numpy as np
from tqdm import tqdm
import pandas as pd
import logging 

logger = logging.getLogger(__name__)

def get_difference(data, num_features):
    dataframe = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(["customer_ID"]), total=data["customer_ID"].nunique(), desc="Calculating differences"):
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        dataframe.append(diff_df1)
        customer_ids.append(customer_id)
    dataframe = np.concatenate(dataframe, axis=0)
    dataframe = pd.DataFrame(dataframe, columns=[col + "_diff1" for col in df[num_features].columns])
    dataframe["customer_ID"] = customer_ids
    return dataframe

def read_preprocess_data(cfg):
    logger.info("Reading the data")
    train = pd.read_csv(f"{cfg.input_dir}amex-default-prediction/train_data.csv")
    features = train.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
    cat_features = cfg.cat_features 
    num_features = [col for col in features if col not in cat_features]
    logger.info("Train feature engineering")
    train_num_agg = train.groupby("customer_ID")[num_features].agg(["mean", "std", "min", "max", "last"])
    train_num_agg.columns = ["_".join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace=True)
    train_cat_agg = train.groupby("customer_ID")[cat_features].agg(["count", "last", "nunique"])
    train_cat_agg.columns = ["_".join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace=True)
    train_labels = pd.read_csv(f"{cfg.input_dir}amex-default-prediction/train_labels.csv")
    cols = list(train_num_agg.dtypes[train_num_agg.dtypes == "float64"].index)
    for col in tqdm(cols):
        train_num_agg[col] = train_num_agg[col].astype(np.float32)
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == "int64"].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)
    train_diff = get_difference(train, num_features)
    train = train_num_agg.merge(
        train_cat_agg, how="inner", on="customer_ID"
    ).merge(train_diff, how="inner", on="customer_ID").merge(
        train_labels, how="inner", on="customer_ID"
    )
    test = pd.read_csv(f"{cfg.input_dir}amex-default-prediction/test_data.csv")
    logger.info("Test feature engineering...")
    test_num_agg = test.groupby("customer_ID")[num_features].agg(["mean", "std", "min", "max", "last"])
    test_num_agg.columns = ["_".join(x) for x in test_num_agg.columns]
    test_num_agg.reset_index(inplace=True)
    test_cat_agg = test.groupby("customer_ID")[cat_features].agg(["count", "last", "nunique"])
    test_cat_agg.columns = ["_".join(x) for x in test_cat_agg.columns]
    test_cat_agg.reset_index(inplace=True)
    test_diff = get_difference(test, num_features)
    test = test_num_agg.merge(test_cat_agg, how="inner", on="customer_ID").merge(
        test_diff, how="inner", on="customer_ID"
    )
    return train, test
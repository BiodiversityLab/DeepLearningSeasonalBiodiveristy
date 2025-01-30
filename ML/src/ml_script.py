import sys
sys.path.append(".")
sys.path.append("..")
import pandas as pd
from datetime import datetime
from sklearn import linear_model, ensemble
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import xgboost as xgb

from config import settings, all_features
from src.dataset import GeoDiversityData

def prepare_df_from_geo(data: GeoDiversityData, use_avgs: bool=False, pred_data=False):
    if pred_data:
        dfs = [
            data.get_numerical_features(),
        ]
        if use_avgs:
            dfs.append(data.get_averaged_spatial_features())
        return pd.concat(dfs, axis=1, join='inner')
    else:
        dfs = [
            data.get_labels(),
            data.get_numerical_features(),
        ]
        if use_avgs:
            dfs.append(data.get_averaged_spatial_features())
        return pd.concat(dfs, axis=1, join='inner')


def experiment(ModelClass,
               data: GeoDiversityData,
               model_config: dict[str] = {},
               use_avgs: bool = False,
               random_state: int = 42,
               **kwargs):
    settings.init_seeds()  # Just in case we miss some random states to be set

    kf = KFold(random_state=random_state, **kwargs)

    df = prepare_df_from_geo(data, use_avgs=use_avgs)

    trap_ids = {}

    regressions = []

    all_trap_ids = data.get_trapids()
    folds = [(all_trap_ids[train], all_trap_ids[test]) for train, test in kf.split(all_trap_ids)]

    results = []

    for i, (trap_ids["train"], trap_ids["test"]) in enumerate(folds):
        indices: dict[str, GeoDiversityData] = {}
        dfs: dict[str, pd.DataFrame] = {}
        X: dict[str, pd.DataFrame] = {}
        y: dict[str, pd.Series] = {}
        for split in splits:
            indices[split] = data.get_indices_from_trapids(trap_ids=trap_ids[split])
            dfs[split] = df.loc[indices[split]]
            X[split], y[split] = dfs[split].drop(columns=[settings.label_col]), dfs[split][settings.label_col]
            y[split] /= settings.max_label

        if ModelClass == linear_model.LinearRegression:
            model = ModelClass(**model_config)
        else:
            model = ModelClass(random_state=random_state, **model_config)
        model.fit(X["train"], y["train"])
        y["pred"] = model.predict(X["test"])

        # print("Coefficients: \n", regr.coef_)
        results.append({
            "RMSE": mean_squared_error(y['test'], y['pred'], squared=False),
            "MAE": mean_absolute_error(y['test'], y['pred']),
            "R2": r2_score(y['test'], y['pred'])
        })
        regressions.append(model)

    results = pd.DataFrame(results)
    scores = pd.concat([results.mean(), results.std()], axis=1)
    scores.columns = ["average", "std"]

    return scores, results, regressions


def train_and_predict(ModelClass, train_data: GeoDiversityData, predict_data: GeoDiversityData, model_config: dict = {}, use_avgs: bool=False, random_state: int=42):
    settings.init_seeds() # Just in case we miss some random states to be set

    # Prepare the training and prediction dataframes
    train_df = prepare_df_from_geo(train_data, use_avgs=use_avgs)
    predict_df = prepare_df_from_geo(predict_data, use_avgs=use_avgs, pred_data=True)

    # Separate features and target for the training dataset
    X_train = train_df.drop(columns=[settings.label_col])
    y_train = train_df[settings.label_col]
    y_train /= settings.max_label

    # Initialize and train the model
    if ModelClass == linear_model.LinearRegression:
        model = ModelClass(**model_config)
    else:
        model = ModelClass(random_state=random_state, **model_config)
    model.fit(X_train, y_train)

    # Make predictions on the prediction dataset
    y_predict = model.predict(predict_df)
    y_predict *= settings.max_label

    save_predictions_to_csv(predict_data, y_predict, ModelClass.__name__)


def save_predictions_to_csv(data, predictions, model_name):
    unique_ids = data.get_sampleids()
    predictions_df = pd.DataFrame({
        'unique_id': unique_ids.astype(int),
        'prediction': predictions
    })

    date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
    file_path = settings.prediction_dir / f"{model_name}_predictions_{date_time_str}.csv"

    predictions_df.to_csv(file_path, index=False)
    print(f"Predictions for {model_name} saved to {file_path}")


splits = ["train", "test"]
include_features = ['NDVI', 'dem', 'slope', 'aspect', 'bio01', 'bio02', 'bio03', 'bio04', 'bio07', 'bio12', 'bio15', 'dewpoint', 'max_temp', 'mean_temp', 'min_temp', 'precip', 'hii', 'ph', 'volume', 'average_height', 'ground_area', 'average_diameter', 'biomass', 'vegetation_quota','soil_moisture']
features = GeoDiversityData.get_feature_list(all_features, include_features=include_features)
# train data
diversity_data = GeoDiversityData(pd.read_csv(settings.samples_file), features_list=features)
# prediction data
preds_data = GeoDiversityData(pd.read_csv(settings.prediction_file), features_list=features)

linreg_scores, linreg_results, linreg_regressions = experiment(linear_model.LinearRegression, diversity_data, n_splits=7, shuffle=True, random_state=42)
rf_scores, rf_results, rf_regressions = experiment(ensemble.RandomForestRegressor, diversity_data, n_splits=7, shuffle=True, random_state=42)
xgb_scores, xgb_results, xgb_regressions = experiment(xgb.XGBRegressor, model_config={"objective":"reg:squarederror"},data=diversity_data, n_splits=7, shuffle=True, random_state=42)

date_time = datetime.now().strftime("%y%m%d-%H%M%S")
with open(settings.prediction_dir / f'results_scores_{date_time}.txt', 'w') as f:
    f.write(f'Linear Regression\nFolds:\n{linreg_results}\nScores:\n{linreg_scores}\n{"-"*40}\n')
    f.write(f'Random Forest\nFolds:\n{rf_results}\nScores:\n{rf_scores}\n{"-"*40}\n')
    f.write(f'XGBoost\nFolds:\n{xgb_results}\nScores:\n{xgb_scores}\n{"-"*40}\n')

train_and_predict(linear_model.LinearRegression, diversity_data, preds_data)
train_and_predict(ensemble.RandomForestRegressor, diversity_data, preds_data, random_state=42)
train_and_predict(xgb.XGBRegressor, diversity_data, preds_data, model_config={"objective":"reg:squarederror"}, random_state=42)

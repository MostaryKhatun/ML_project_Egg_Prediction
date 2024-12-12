import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

def train_model():
    try:
        dataset_path = r'D:\EggProduction_PredictApp\eggprediction\mlapp\data\data.csv'

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        dataset = pd.read_csv(dataset_path)

        if 'Egg_production_percentage' not in dataset.columns:
            raise KeyError("The column 'Egg_production_percentage' is missing from the dataset.")

        X = dataset.drop(columns=['Egg_production_percentage'])
        y = dataset['Egg_production_percentage']

        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].astype('category').cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestRegressor()
        rf_model.fit(X_train_scaled, y_train)
        rf_y_pred_rf = rf_model.predict(X_test_scaled)
        rf_mse = mean_squared_error(y_test, rf_y_pred_rf)
        rf_r2 = r2_score(y_test, rf_y_pred_rf)

        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_y_pred_lr = lr_model.predict(X_test_scaled)
        lr_mse = mean_squared_error(y_test, lr_y_pred_lr)
        lr_r2 = r2_score(y_test, lr_y_pred_lr)

        svr_model = SVR()
        svr_model.fit(X_train_scaled, y_train)
        svr_y_pred_svr = svr_model.predict(X_test_scaled)
        svr_mse = mean_squared_error(y_test, svr_y_pred_svr)
        svr_r2 = r2_score(y_test, svr_y_pred_svr)

        models = {
            'RandomForest': {'model': rf_model, 'mse': rf_mse, 'r2': rf_r2},
            'LinearRegression': {'model': lr_model, 'mse': lr_mse, 'r2': lr_r2},
            'SVR': {'model': svr_model, 'mse': svr_mse, 'r2': svr_r2},
        }

        best_model_name = min(models, key=lambda x: models[x]['mse'])
        best_model = models[best_model_name]['model']

        model_path = r'D:\EggProduction_PredictApp\eggprediction\mlapp\models\egg_model.sav'
        scaler_path = r'D:\EggProduction_PredictApp\eggprediction\mlapp\models\scaler.sav'

        with open(model_path, 'wb') as model_file:
            pickle.dump(best_model, model_file)
        
        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)

    except Exception as e:
        raise

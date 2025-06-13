import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset

class SalesModel:

    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset with DateTimeIndex, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {} # dict of dicts with model results

    def score(self, truth, preds):
        return {
            'MAPE': MAPE(truth, preds),
            'RMSE': RMSE(truth, preds),
            'MSE': MSE(truth, preds)
        }

    def run_models(self, n_splits=4, test_size=365):
        '''Run the models and store results for cross validated splits in
        self.results.
        '''
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # keep track of splits
        
        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]
            # Base model - please leave this here
            preds = self._base_model(X_train, X_test)
            
            if 'Base' not in self.results:
                self.results['Base'] = {}
            self.results['Base'][cnt] = self.score(X_test[self.target_col],preds)
            
            self.plot(preds, 'Base')
            # Other models...
            
            # SARIMA model
            preds_sarima = self._sarima_model(X_train, X_test)
            if 'SARIMA' not in self.results:
                self.results['SARIMA'] = {}
            self.results['SARIMA'][cnt] = self.score(X_test[self.target_col], preds_sarima)
            self.plot(preds_sarima, 'SARIMA')

            ## XGBoost model
            preds_xgb = self._xgb_model(X_train, X_test)
            if 'XGBoost' not in self.results:
                self.results['XGBoost'] = {}
            self.results['XGBoost'][cnt] = self.score(X_test[self.target_col], preds_xgb)
            self.plot(preds_xgb, 'XGBoost')

            ## XGBoost model with more features; lags and rolling means 
            preds_xgb2 = self._xgb_model2(X_train, X_test)
            if 'XGBoost2' not in self.results:
                self.results['XGBoost2'] = {}
            self.results['XGBoost2'][cnt] = self.score(X_test[self.target_col], preds_xgb2)
            self.plot(preds_xgb2, 'XGBoost2')
            
            # Prophet model
            preds_prophet = self._prophet_model(X_train, X_test)
            if 'Prophet' not in self.results:
                self.results['Prophet'] = {}
            self.results['Prophet'][cnt] = self.score(X_test[self.target_col], preds_prophet)
            self.plot(preds_prophet, 'Prophet')

            # DeepAR model
            preds_deepar = self._deepar_model(X_train, X_test)
            if 'DeepAR' not in self.results:
                self.results['DeepAR'] = {}
            self.results['DeepAR'][cnt] = self.score(X_test[self.target_col], preds_deepar)
            self.plot(preds_deepar, 'DeepAR')
            
            cnt += 1

    def _base_model(self, train, test):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col],
                                        period=365)
        res_clip = res.seasonal.apply(lambda x: max(0,x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index = test.index, 
                         data = map(lambda x: res_dict[x], test.index.dayofyear))

    def _sarima_model(self, train, test):
        '''SARIMA model using statsmodels'''
        model = SARIMAX(train[self.target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 365))
        results = model.fit(disp=False)
        preds = results.predict(start=test.index[0], end=test.index[-1])
        return preds

    def _xgb_model(self, train, test):
        '''
        XGBoost model using engineered features like month and quarter.
    
        Returns:
            pd.Series: Predictions for the test set
        '''
        # Drop target and extras from features
        features = train.drop(columns=[self.target_col,'lag_1', 'lag_7', 'lag_30','roll_mean_7', 'roll_std_7','roll_mean_30', 
                                       'roll_std_30','dayofweek','_id'])
        features_test = test.drop(columns=[self.target_col,'lag_1', 'lag_7', 'lag_30','roll_mean_7', 'roll_std_7','roll_mean_30', 
                                       'roll_std_30','dayofweek','_id'])
    
        # Ensure month and quarter exist and are numeric
        for col in ['monthly', 'quarter']:
            if col not in features.columns:
                raise ValueError(f'Missing expected feature column: {col}')
    
        X_train = features
        y_train = train[self.target_col]
        X_test = features_test
    
        # Initialize and train model
        model = XGBRegressor(n_estimators=100, learning_rate=0.1,
                             max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        preds_xg = model.predict(X_test)
        return pd.Series(preds_xg, index=test.index)

    def _xgb_model2(self, train, test):
        '''
        XGBoost model using more features.
    
        Returns:
            pd.Series: Predictions for the test set
        '''
        features_to_use = train.drop(columns=[self.target_col,'_id'])
        features_test = test.drop(columns=[self.target_col,'_id'])
    
        X_train = features_to_use
        y_train = train[self.target_col]
        X_test = features_test
    
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
        # Get feature importances
        importances = model.feature_importances_
        features = X_train.columns
        
        # Convert to a DataFrame for easier viewing
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print(importance_df)
        preds = model.predict(X_test)
        return pd.Series(preds, index=test.index)

    def _prophet_model(self, train, test):
        '''Prophet model using fbprophet'''
        #df_train = train[[train.index, self.target_col]].rename(columns={train.index: 'ds', self.target_col: 'y'})
        df_train = train.reset_index()[['Timestamp', self.target_col]]
        df_train.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df_train)
        future = model.make_future_dataframe(periods=len(test), freq='D')
        forecast = model.predict(future)
        preds = forecast['yhat'].iloc[-len(test):].values
        return pd.Series(preds, index=test.index)

    def _deepar_model(self, train, test):
            '''DeepAR model using GluonTS'''
            df_train = train.reset_index()[['Timestamp', self.target_col]]
            df_train.columns = ['Timestamp', 'target']
            train_ds = ListDataset([{'start': df_train['Timestamp'].iloc[0], 'target': df_train['target'].values}], freq='D')
            estimator = DeepAREstimator(freq='D', prediction_length=len(test))
            predictor = estimator.train(train_ds)
            forecast = list(predictor.predict(train_ds))
            preds = forecast[0].mean
            return pd.Series(preds, index=test.index)
    
    def plot(self, preds, label):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds, label = label, color='red')
        plt.title(f'{label} Forecast')
        plt.legend()
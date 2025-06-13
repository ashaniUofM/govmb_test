import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor



class RedemptionModel:

    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {} # dict of dicts with model results

    def score(self, truth, preds):
        # Score our predictions - modify this method as you like
        return {
            'MAPE': MAPE(truth, preds),'RMSE': RMSE(truth, preds),'MSE': MSE(truth, preds)
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
            self.results['Base'][cnt] = self.score(X_test[self.target_col],
                                preds)
            self.plot(preds, 'Base')
            # Other models...
            preds2 = self._updated_model(X_train, X_test)
            if 'Updated' not in self.results:
                self.results['Updated'] = {}
            self.results['Updated'][cnt] = self.score(X_test[self.target_col],
                                preds2['mean'])
            self.plot_new(preds2, 'Updated')

            ## XGBoost model with more features; month, dayofweek, lags and rolling means 
            preds_xgb = self._xgb_model(X_train, X_test)
            if 'XGBoost' not in self.results:
                self.results['XGBoost'] = {}
            self.results['XGBoost'][cnt] = self.score(X_test[self.target_col], preds_xgb)
            self.plot(preds_xgb, 'XGBoost')
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

    def _updated_model(self, train, test):
        '''
        Updated model using the features; monthly, quarter
        Base model limits the seasonal signal to an average per day of the year
        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=365, model='additive')
        res_clip = res.seasonal.apply(lambda x: max(0,x))
        
        # Re-align index in order to merge with calendar features
        seasonal = res_clip.to_frame('seasonal')
        seasonal['date'] = seasonal.index
        seasonal['monthly'] = seasonal['date'].dt.month
        seasonal['quarter'] = seasonal['date'].dt.quarter.values
        seasonal['dayofweek'] = seasonal['date'].dt.dayofweek
        
        # Group by both month and dayofweek
        grouped = seasonal.groupby(['monthly', 'quarter','dayofweek'])['seasonal'].agg(['mean', 'std'])

        # Prepare test data with same keys
        test_keys = pd.DataFrame({
            'monthly': test.monthly,
            'quarter': test.quarter,
            'dayofweek' : test.dayofweek
        }, index=test.index)

        #preds = test_keys.apply(lambda row: grouped.get((row['monthly'], row['quarter'], row['dayofweek'] ), grouped.mean()), axis=1)
        def get_forecast_with_uncertainty(row):
            key = (row['monthly'], row['quarter'], row['dayofweek'])
            if key in grouped.index:
                mu = grouped.loc[key, 'mean']
                sigma = grouped.loc[key, 'std'] if not pd.isna(grouped.loc[key, 'std']) else 0
            else:
                mu = grouped['mean'].mean()
                sigma = grouped['std'].mean()
            return pd.Series({'mean': mu, 'lower_95': mu - 1.96*sigma, 'upper_95': mu + 1.96*sigma})

        preds_df = test_keys.apply(get_forecast_with_uncertainty, axis=1)
        #return pd.Series(preds.values, index=test.index)
        return preds_df

    def _xgb_model(self, train, test):
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
    
    def plot(self, preds, label):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds, label = label, color='red')
        plt.legend()

    def plot_new(self, preds_df, label):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], label='Observed', color='grey', s=10, alpha=0.6)
    
        # Scatter for forecast mean
        ax.plot(preds_df.index, preds_df['mean'], label=f'{label} Forecast', color='blue', linewidth=2)
    
        # Uncertainty: fill between lower and upper
        ax.fill_between(preds_df.index, preds_df['lower_95'], preds_df['upper_95'], 
                        color='blue', alpha=0.2, label='Uncertainty')
    
        ax.legend()
        plt.title(f"Forecast with Uncertainty (Scatter) - {label}")
        plt.xlabel("Date")
        plt.ylabel(self.target_col)
        plt.tight_layout()
        plt.show()


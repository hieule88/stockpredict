import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

class PreProcessor():
    def __init__(self, max_params = 14, replace_nan = 'interpolate'):
        self.max_params = max_params
        self.replace_nan = replace_nan

    def preprocess(self, data: pd.DataFrame):
        # drop na do tinh TA
        #data.drop(data.index[[i for i in range(-(self.max_params + 1),0)]], inplace= True)
        # add more fill nan 


        if self.replace_nan == 'mean':
            value = data.mean()
            data.fillna(value= value, inplace= True)
        elif self.replace_nan == 'interpolate':
            data.interpolate(method='ffill', order=2, inplace=True)
        data.fillna(value= 0, inplace= True)        
        # Norm 
        data, max_close, min_close = self.normalize(data)

        return data, max_close, min_close
    
    def normalize(self, df):
        result = df.copy()
        max_close = 0
        min_close = 0
        for feature_name in df.columns:
            if feature_name not in ['TXDATE','name']:
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            if feature_name in ['close']:
                max_close = max_value
                min_close = min_value
        return result, max_close, min_close
    
    def test_stationarity(self, ts_data, column='', signif=0.05, series=False, type = 'adf'):
        if type == 'adf':
            if series:
                adf_test = adfuller(ts_data, autolag='AIC')
            else:
                adf_test = adfuller(ts_data[column], autolag='AIC')
            p_value = adf_test[1]
            if p_value <= signif:
                test_result = True
            else:
                test_result = False
        elif type == 'kpss':
            if series:
                kpsstest = kpss(ts_data, regression='c', nlags="auto")
            else:
                kpsstest = kpss(ts_data[column], regression='c', nlags="auto")
            p_value = kpsstest[1]
            if p_value < signif:
                test_result = False
            else:
                test_result = True
        elif type == 'adf_kpss':
            if self.test_stationarity(ts_data, column, signif, series, type= 'adf') \
                and self.test_stationarity(ts_data, column, signif, series, type= 'kpss'):
                return True
            else:
                return False
        elif type == 'zivot_andrews':
            if series:
                zatest = zivot_andrews(ts_data, regression='c', nlags="auto")
            else:
                zatest = zivot_andrews(ts_data[column], regression='c', nlags="auto")
            p_value = zatest[1]
            if p_value > signif:
                test_result = False
            else:
                test_result = True
        return test_result

    def convert(self, data, column, order):
        differenced_data = ''
        seed = 0
        endseed = 0

        if self.difftest == 'diffty':
            seed = data[column].iloc[0]
            endseed = data[column].iloc[-1]
            differenced_data = data[column].diff(order)
            differenced_data.fillna(differenced_data.mean(), inplace=True)
        elif self.difftest == 'cbrt':
            differenced_data = data[column].apply(np.cbrt)
            differenced_data.interpolate(method='ffill', order=2, inplace=True)
        elif self.difftest == 'cbrt&diffty':
            differenced_data = data[column].apply(np.cbrt)
            seed = differenced_data.iloc[0]
            endseed = differenced_data.iloc[-1]
            differenced_data = differenced_data.diff(order)
            # differenced_data.fillna(0, inplace=True)
        differenced_data.fillna(0, inplace=True)
        return differenced_data, seed, endseed     

    def reconvert(need_retransform = False, difftest='', data=0, \
                bonus_data=0, scaler=False, min_close = 0, max_close = 0):
        if need_retransform:
            if difftest == 'diffty':
                reverse_data = bonus_data + data.cumsum()
            elif difftest == 'cbrt':
                reverse_data = data.pow(3)
            elif difftest == 'cbrt&diffty':
                reverse_data = bonus_data + data.cumsum()
                reverse_data = reverse_data.pow(3)
        else:
            reverse_data = data
            
        if scaler:
            reverse_data = (reverse_data * (max_close - min_close)) + min_close

        return reverse_data
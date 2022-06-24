import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import pandas_ta as ta
from tqdm import tqdm
import os 

list_features = [
    'adi', 'obv', 'ema', 'sma', 'wma', 'mfi', 'cmf', 'rsi', 'cci', 'roc', 'atr', 
    'fibonacci', 'bollinger', 'wR', 'srsi', 'macd' 
]
default_dict = {
    i: 10 for i in list_features
}

class Extractor():
    def __init__(self, tickerframe, name, max_len):
        self.name = name
        tickerframe = tickerframe.sort_values(by=['TXDATE'], ascending=True)
        tkdate = tickerframe['TXDATE'] 
        tickerframe = tickerframe.drop(columns=['TICKER', 'TXDATE'])
        self.tickerframe = tickerframe
        self.close = self.tickerframe["close"]
        self.open = self.tickerframe["open"]
        self.low = self.tickerframe["low"]
        self.high = self.tickerframe["high"]
        self.volume = self.tickerframe["volume"]
        self.hyperparams = default_dict
        self.max_len = max_len
        # self.popular = pd.concat([self.countTA_popular(),tkdate], axis=1).dropna().iloc[:max_len]
        self.popular = pd.concat([self.countTA_popular(),tkdate], axis=1)

    def countTA_popular(self):
        TApopular = {}      

        TApopular['obv'] = ta.obv(close=self.close, volume=self.volume)

        TApopular['ema'] = ta.ema(close=self.close)        
        
        TApopular['wma'] = ta.wma(close=self.close)
        
        TApopular['mfi'] = ta.mfi(open=self.open, high=self.high,\
                                low=self.low, close=self.close, \
                                volume=self.volume)       
           
        TApopular['rsi'] = ta.rsi(close=self.close)             

        bollinger = ta.bbands(close=self.close) 
        for boll_index in bollinger.keys():
            TApopular[boll_index] = bollinger[boll_index] 
        
        
        srsi = ta.stochrsi(close=self.close, high=self.high, \
                                low=self.low)
        for srsi_index in srsi.keys():
            TApopular[srsi_index] = srsi[srsi_index] 

        so = ta.stoch(close=self.close, high=self.high, \
                                low=self.low)  
        # stoch and slow in result                                         
        for so_index in so.keys():
            TApopular[so_index] = so[so_index]

        macd = ta.macd(close=self.close)
        for macd_index in macd.keys():
            TApopular[macd_index] = macd[macd_index]
        
        ichimoku = ta.ichimoku(close=self.close, high=self.high, \
                                    low=self.low)[0]
        TApopular['ichimokuISA'] = ichimoku['ISA_9']
        TApopular['ichimokuISB'] = ichimoku['ISB_26']
        TApopular['ichimokuITS'] = ichimoku['ITS_9']
        TApopular['ichimokuIKS'] = ichimoku['IKS_26']
        TApopular['ichimokuICS'] = ichimoku['ICS_26']
        
        TApopular['price'] = self.close
        #TApopular['high'] = self.high
        #TApopular['low'] = self.low

        import numpy as np
        # if val is 0
        val_added = 0.001        
        TApopular['vol'] = np.log(self.volume+val_added)

        TApopular = pd.concat([TApopular[key_index] for key_index in TApopular.keys()], axis=1)

        return TApopular

    def process(self):
        # replace inf 
        large_num = 999999999999999999.0
        output=self.popular.replace('inf',large_num)
        output=output.replace('-inf',-1.0*large_num)
        output=output.replace(np.inf,large_num)
        output=output.replace(-np.inf,-1.0*large_num)
        # output=output.replace(np.nan,0)
        output.reset_index(inplace=True, drop=True)
        
        return output.iloc[-self.max_len:]
        # return output.dropna()

    def to_dataframe(self, dict_data):
        return pd.DataFrame.from_dict(dict_data)


# if __name__ =='__main__':
#     dataset = pd.DataFrame(pd.read_csv('./dataset/TradingHistory.csv'))
#     dataset.drop(columns=['Unnamed: 0'], inplace=True)
#     tickers = pd.DataFrame(pd.read_csv('./dataset/Ticker.csv')['TICKER'])
#     preprocessor = PreProcessor()

#     for row in tqdm(tickers.iterrows(), desc= "Saving Ticker's Feature: ", total= len(tickers.index)):
#     # Load ticker infor
#         p_ticker = row[1]['TICKER']
        
#         ticker_infor = dataset.loc[dataset['TICKER']==p_ticker]
#         if len(ticker_infor.index) < 60 : # Bo sung tham so toi thieu ngay giao dich
#             continue
#         ticker_infor = FeatureTicker(ticker_infor, name = p_ticker)
#         ticker_infor.popular.reset_index(inplace=True, drop=True)
#         ticker_infor = ticker_infor.popular

#         ticker_infor = preprocessor.preprocess(ticker_infor)

#         ticker_infor['name'] = [p_ticker for i in range(len(ticker_infor.index))]
#         # Save to csv
#         filepath = './dataset/Features.csv'
#         header = 1 - os.path.exists(filepath)
#         ticker_infor.to_csv(filepath, mode='a', header=header, index=False)
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages
plt.style.use('fivethirtyeight')

def define_all_combinations():
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)
 
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    print(pdq)
 
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print(seasonal_pdq)
    return pdq, seasonal_pdq

class Best:
    def __init__(self, order, seasonal_pdq, aic):
        self.Order = order
        self.SeasonalOrder = seasonal_pdq
        self.Aic = aic
def get_best_result(pdq, seasonal_pdq, data):
    best = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(data, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit()
                if best == None:
                    best = Best(param, param_seasonal, results.aic)
                else:
                    refer = Best(param, param_seasonal, results.aic)
                    if best.Aic > refer.Aic:
                        best = refer
                print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    return best
def init(data, best):
    print('ARIMA{}x{} - AIC:{}'.format(best.Order, best.SeasonalOrder, best.Aic))
    #model = sm.tsa.statespace.SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    model = sm.tsa.statespace.SARIMAX(data, order=best.Order, seasonal_order=best.SeasonalOrder, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    return results
    #print(results.summary().tables[1])
    #results.plot_diagnostics(figsize=(12, 12))
def get_prediction(results, data):
    pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
    pred_ci = pred.conf_int()

    ax = data['1990':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('CO2 Levels')
    plt.legend()
    plt.show()

def load(): 
    data = pd.read_csv('co2.csv', parse_dates=['date'], index_col='date')
    data = data['co2'].resample('MS').mean()
    data = data.fillna(data.bfill())

    print(data.head())
    #data.plot(figsize=(12, 6))

    #plt.show()
    return data
data = load()
pdq, seasonal_pdq = define_all_combinations()
best = get_best_result(pdq, seasonal_pdq, data)
results = init(data, best)
get_prediction(results, data)
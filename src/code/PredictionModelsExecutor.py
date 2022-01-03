import PredictionModel as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

from statsmodels.tsa.seasonal import STL

class PredictionModelExecutor:
    
    def __init__(self, sales_path):
        print(f'leyendo fichero {sales_path}')
        self.sales = pd.read_csv(sales_path, sep=';')
        print(f'Existen {self.sales.shape[0]} registros de venta desde {self.sales.date.min()} hasta {self.sales.date.max()}')
    
    def __format_sales(self, sales):
        result = sales
        result['date'] = pd.to_datetime(result.date.astype(str), format='%Y-%m-%d')
        return result

    def __group_sales(self, sales, time_aggregator = '1d'):
        aggregator = {
            'is_prom':'sum', 
            'is_holiday':'max', 
            'year': 'first', 
            'quarter': 'first',
            'season': 'first', 
            'month': 'first', 
            'week': 'first', 
            'day': 'first', 
            'weekday': 'first',   
            'units_sold': 'sum', 
            'units_sold_last_year': 'sum', 'units_sold_last_quarter': 'sum', 'units_sold_last_month': 'sum', 'units_sold_last_week': 'sum', 'units_sold_last_day': 'sum', 
            'units_sold_last_52_weeks': 'sum', 'units_sold_last_12_weeks': 'sum', 'units_sold_last_8_weeks': 'sum', 'units_sold_last_4_weeks': 'sum'}
        result = sales.groupby('date').agg(aggregator).reset_index()
        result = result.resample(on='date', rule=time_aggregator).agg(aggregator).reset_index()
        return result

    def __group_sales_by_sku(self, sales, time_aggregator = '1d'):
        aggregator = {
            'is_prom':'sum', 
            'is_holiday':'max', 
            'year': 'first', 
            'quarter': 'first',
            'season': 'first', 
            'month': 'first', 
            'week': 'first', 
            'day': 'first', 
            'weekday': 'first',   
            'units_sold': 'sum', 
            'units_sold_last_year': 'sum', 'units_sold_last_quarter': 'sum', 'units_sold_last_month': 'sum', 'units_sold_last_week': 'sum', 'units_sold_last_day': 'sum', 
            'units_sold_last_52_weeks': 'sum', 'units_sold_last_12_weeks': 'sum', 'units_sold_last_8_weeks': 'sum', 'units_sold_last_4_weeks': 'sum'}
        result = sales.groupby(['date', 'sku']).agg(aggregator).reset_index()
        result = result.groupby(by='sku').resample(on='date', rule=time_aggregator).agg(aggregator).reset_index()
        return result

    def __as_time_series(self, grouped_sales):
        return grouped_sales[['date', 'is_holiday', 'is_prom', 'year', 'quarter', 'season', 'month', 'day', 'weekday', 
        'units_sold', 'units_sold_last_day', 
        'units_sold_last_week', 'units_sold_last_4_weeks', 'units_sold_last_8_weeks', 'units_sold_last_12_weeks']].set_index("date")

    def __as_time_series_with_sku(self, grouped_sales):
        return grouped_sales[['date', 'sku', 'is_holiday', 'is_prom', 'year', 'quarter', 'season', 'month', 'day', 'weekday', 
        'units_sold', 'units_sold_last_day', 
        'units_sold_last_week', 'units_sold_last_4_weeks', 'units_sold_last_8_weeks', 'units_sold_last_12_weeks', 
        'units_soldNextWeek', 'units_soldNext4Weeks']].set_index("date")

    def __mark_training_and_test_data(self, time_series, date):
        result = time_series
        result['training'] = 0
        result.loc[result.index < date, 'training'] = 1
        return result

    def __get_data(self, min_test_data, time_aggregator):
        sales = self.sales
        formated_sales = self.__format_sales(sales)
        grouped_sales = self.__group_sales(formated_sales, time_aggregator)
        data = self.__as_time_series(grouped_sales)
        data = self.__mark_training_and_test_data(data, min_test_data).copy()
        return data

    def __get_data_with_sku(self, min_test_data, time_aggregator):
        sales = self.load_sales()
        formated_sales = self.__format_sales(sales)
        grouped_sales = self.__group_sales_by_sku(formated_sales, time_aggregator)
        data = self.__as_time_series_with_sku(grouped_sales)
        data = self.__mark_training_and_test_data(data, min_test_data).copy()
        return data

    def __plot_series(self, series, names, colors, min_date = '2000-01-01'):
        plt.figure(figsize=(20,10))
        for s, n, c in zip(series, names, colors):
            s = s[s.index > min_date]
            sns.lineplot(x = s.index, y = s, lw=1.2, label=n, color=c,)
        plt.show()
    
    def __calculate_error(self, prediction, actual, verbose=True):
        mae = mean_absolute_error(prediction, actual)
        r2 = r2_score(prediction, actual)
        if (verbose):
            print(f'R2: {r2:0.5f} - MAE: {mae:0.5f}')
        
        return {'r2': r2, 'mae': mae}

    def __verify_predictions(self, data_with_predictions, time_aggregator = '1d'):
        plt.figure(figsize=(20,10))

        training_data_with_predictions = data_with_predictions[data_with_predictions.training == 1]
        test_data_with_predictions = data_with_predictions[data_with_predictions.training == 0]

        data_with_predictions = data_with_predictions.resample(time_aggregator).sum()
        training_data_with_predictions = training_data_with_predictions.resample(time_aggregator).sum()
        test_data_with_predictions = test_data_with_predictions.resample(time_aggregator).sum()


        self.__plot_series([data_with_predictions.units_sold, training_data_with_predictions.sarimax_pred, test_data_with_predictions.sarimax_pred], ['Units Sold', 'Sarimax Pred Training', 'Sarimax Pred Test'], ['lightblue', 'orange', 'black'])
        self.__plot_series([data_with_predictions.units_sold, training_data_with_predictions.lstm_pred, test_data_with_predictions.lstm_pred], ['Units Sold', 'LSTM Pred Training', 'LSTM Pred Test'], ['lightblue', 'pink', 'black'])
        self.__plot_series([data_with_predictions.units_sold, training_data_with_predictions.units_sold_last_week, test_data_with_predictions.units_sold_last_week], ['Units Sold', 'Last Week Training', 'Last Week Pred Test'], ['lightblue', 'green', 'black'])
        self.__plot_series([data_with_predictions.units_sold, training_data_with_predictions.units_sold_last_4_weeks, test_data_with_predictions.units_sold_last_4_weeks], ['Units Sold', 'Last 4 Weeks Training', 'Last 4 Weeks Pred Test'], ['lightblue', 'red', 'black'])
        self.__plot_series([data_with_predictions.units_sold, training_data_with_predictions.units_sold_last_8_weeks, test_data_with_predictions.units_sold_last_8_weeks], ['Units Sold', 'Last 8 Weeks Training', 'Last 8 Weeks Pred Test'], ['lightblue', 'brown', 'black'])
        self.__plot_series([data_with_predictions.units_sold, training_data_with_predictions.prophet_pred, test_data_with_predictions.prophet_pred], ['Units Sold', 'Prophet Training', 'Prophet Pred Test'], ['lightblue', 'yellow', 'black'])

        data_test = data_with_predictions[(data_with_predictions.training == 0) & (data_with_predictions.lstm_pred.notna())]
        sarimax_error = self.__calculate_error(data_test.sarimax_pred, data_test.units_sold, verbose=False)
        lstm_error =  self.__calculate_error(data_test.lstm_pred, data_test.units_sold, verbose=False)
        last_week_error = self.__calculate_error(data_test.units_sold_last_week, data_test.units_sold, verbose=False)
        last4_weeks_error = self.__calculate_error(data_test.units_sold_last_4_weeks, data_test.units_sold, verbose=False)
        last8_weeks_error = self.__calculate_error(data_test.units_sold_last_8_weeks, data_test.units_sold, verbose=False)
        prophet_pred_error = self.__calculate_error(data_test.prophet_pred, data_test.units_sold, verbose=False)

        print (f'Sarimax error {sarimax_error}')
        print (f'LSTM error {lstm_error}')
        print (f'Last week error {last_week_error}')
        print (f'Last 4 weeks error {last4_weeks_error}')
        print (f'Last 8 weeks error {last8_weeks_error}')
        print (f'Prophet pred error {prophet_pred_error}')

    def __save_transactions(self, transactions, output_file):
        transactions = transactions.reset_index()
        transactions.to_csv(output_file, index = False, sep=';')
        print(f'{transactions.shape[0]} transacciones guardadas en: {output_file}')

    def execute_models(self, output_file):
        MIN_TEST_DATE = '2021-06-01'
        MIN_VALIDATION_DATE = '2021-01-01'

        data = self.__get_data(MIN_TEST_DATE, '1d')
        train = data[data.training == 1].copy()
        test = data[data.training == 0].copy()

        prediction_model = pm.PredictionModel()
        sarimax_pred = prediction_model.predict_sarimax(train.units_sold, data.index)
        lstm_pred = prediction_model.predict_lstm(train, MIN_VALIDATION_DATE, test)
        result_prophet = prediction_model.predict_prophet(train, test[['is_prom']], ['is_prom'])
        prophet_pred = result_prophet[1]

        data['sarimax_pred'] = sarimax_pred
        data['lstm_pred'] = lstm_pred
        data['prophet_pred'] = prophet_pred.yhat.values
        
        self.__save_transactions(data, output_file)

        print('-------------- -------------- -------------- --------------')
        print('-------------- Day predictions verification --------------')
        print('-------------- -------------- -------------- --------------')
        self.__verify_predictions(data[data.index > '2021-01-04'], '1d')
        print('-------------- -------------- --------------')
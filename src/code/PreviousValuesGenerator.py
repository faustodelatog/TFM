import pandas as pd
import numpy as np

class PreviousValuesGenerator:
    transactions = None
    
    def __init__(self, transactions_path):
        print(f'leyendo fichero {transactions_path}')
        self.transactions = pd.read_csv(transactions_path, sep=';')
        print(f'Existen {self.transactions.shape[0]} registros de venta desde {self.transactions.date.min()} hasta {self.transactions.date.max()}')

    def __transactions_with_previous_dates(self):
        result = self.transactions.copy()
        result['last_year'] = result.date - pd.offsets.DateOffset(years=1)
        result['last_quarter'] = result.date - pd.offsets.DateOffset(months=3)
        result['last_month'] = result.date - pd.offsets.DateOffset(months=1)
        result['last_week'] = result.date - pd.offsets.DateOffset(weeks=1)
        result['last_day'] = result.date - pd.offsets.DateOffset(days=1)
        result['last_52_weeks'] = result.date - pd.offsets.DateOffset(weeks=52)
        result['last_12_weeks'] = result.date - pd.offsets.DateOffset(weeks=12)
        result['last_8_weeks'] = result.date - pd.offsets.DateOffset(weeks=8)
        result['last_4_weeks'] = result.date - pd.offsets.DateOffset(weeks=4)

        return result
        
    def __set_previous_values(self, t):
        transactions = self.transactions
        y = transactions[(transactions.date == t.last_year) & (transactions.sku == t.sku)].units_sold.max()
        q = transactions[(transactions.date == t.last_quarter) & (transactions.sku == t.sku)].units_sold.max()
        m = transactions[(transactions.date == t.last_month) & (transactions.sku == t.sku)].units_sold.max()
        w = transactions[(transactions.date == t.last_week) & (transactions.sku == t.sku)].units_sold.max()
        w52 = transactions[(transactions.date == t.last_52_weeks) & (transactions.sku == t.sku)].units_sold.max()
        w12 = transactions[(transactions.date == t.last_12_weeks) & (transactions.sku == t.sku)].units_sold.max()
        w8 = transactions[(transactions.date == t.last_8_weeks) & (transactions.sku == t.sku)].units_sold.max()
        w4 = transactions[(transactions.date == t.last_4_weeks) & (transactions.sku == t.sku)].units_sold.max()
        d = transactions[(transactions.date == t.last_day) & (transactions.sku == t.sku)].units_sold.max()
        
        if not pd.isna(y): 
            y = np.int64(y)
        if not pd.isna(q): 
            q = np.int64(q)
        if not pd.isna(m): 
            m = np.int64(m)
        if not pd.isna(w): 
            w = np.int64(w)
        if not pd.isna(w52): 
            w52 = np.int64(w52)
        if not pd.isna(w12): 
            w12 = np.int64(w12)
        if not pd.isna(w8): 
            w8 = np.int64(w8)
        if not pd.isna(w4): 
            w4 = np.int64(w4)
        if not pd.isna(d): 
            d = np.int64(d)

        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_year'] = y
        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_quarter'] = q
        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_month'] = m
        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_week'] = w
        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_52_weeks'] = w52
        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_12_weeks'] = w12
        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_8_weeks'] = w8
        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_4_weeks'] = w4
        transactions.loc[(transactions.date == t.date) & (transactions.sku == t.sku), 'units_sold_last_day'] = d

    def add_previous_values(self):
        self.transactions['date'] = pd.to_datetime(self.transactions.date.astype(str), format='%Y-%m-%d')
        temp = self.__transactions_with_previous_dates()
        temp.apply(lambda t: self.__set_previous_values(t), axis=1)
    
    def save_transactions(self, output_file):
        self.transactions.to_csv(output_file, index = False, sep=';')
        print(f'{self.transactions.shape[0]} transacciones guardadas en: {output_file}')

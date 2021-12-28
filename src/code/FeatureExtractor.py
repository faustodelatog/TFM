import pandas as pd

class FeatureExtractor:
    transactions = None
    
    def __init__(self, transactions_path):
        print(f'leyendo fichero {transactions_path}')
        self.transactions = pd.read_csv(transactions_path, sep=';')
        print(f'Existen {self.transactions.shape[0]} registros de venta desde {self.transactions.date.min()} hasta {self.transactions.date.max()}')

    def add_sold_out(self):
        self.transactions['has_stock'] = 1
        self.transactions.loc[(self.transactions.stock == 0), 'has_stock'] = 0
        self.transactions['sold_out'] = 0
        self.transactions.loc[(self.transactions.has_stock == 0) & (self.transactions.units_sold > 0), 'sold_out'] = 1

    def add_has_stock(self):
        self.transactions['has_stock'] = 1
        self.transactions.loc[(self.transactions.stock == 0), 'has_stock'] = 0
        self.transactions['sold_out'] = 0
        self.transactions.loc[(self.transactions.has_stock == 0) & (self.transactions.units_sold > 0), 'sold_out'] = 1
    
    def __week_of_the_month(self, d):
        day = d.day
        if day <= 7: return 1
        if day <= 14: return 2
        if day <= 21: return 3
        return 4

    def add_season(self):
        transactions = self.transactions
        transactions['season'] = 'winter'
        spring = [80, 172]
        summer = [172, 264]
        fall = [264, 355]
        transactions.loc[(transactions.date.dt.dayofyear >= spring[0]) & (transactions.date.dt.dayofyear < spring[1]), 'season'] = 'spring'
        transactions.loc[(transactions.date.dt.dayofyear >= summer[0]) & (transactions.date.dt.dayofyear < summer[1]), 'season'] = 'summer'
        transactions.loc[(transactions.date.dt.dayofyear >= fall[0]) & (transactions.date.dt.dayofyear < fall[1]), 'season'] = 'fall'

        self.transactions = transactions
    
    def add_date_features(self):
        transactions = self.transactions
        transactions['date'] = pd.to_datetime(transactions.date.astype(str), format='%Y-%m-%d')
        transactions['year'] = transactions.date.dt.year
        transactions['quarter'] = transactions.date.dt.quarter
        transactions['month'] = transactions.date.dt.month
        transactions['week'] = transactions.date.dt.isocalendar().week
        transactions['week_of_month'] = transactions.date.apply(self.__week_of_the_month)
        transactions['day'] = transactions.date.dt.day
        transactions['weekday'] = transactions.date.dt.weekday
        transactions['month_name'] = transactions.date.dt.month_name()
        transactions['day_name'] = transactions.date.dt.day_name()
        
        self.transactions = transactions
    
    def format_sku(self):
        self.transactions.sku = ["%02d" % s for s in self.transactions.sku]
    
    def save_transactions(self, output_file):
        self.transactions.to_csv(output_file, index = False, sep=';')
        print(f'{self.transactions.shape[0]} transacciones guardadas en: {output_file}')

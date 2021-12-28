import pandas as pd

class TransactionsFiller:
    transactions = None
    
    def __init__(self, transactions_path):
        print(f'leyendo fichero {transactions_path}')
        self.transactions = pd.read_csv(transactions_path, sep=';')
        print(f'Existen {self.transactions.shape[0]} registros de venta desde {self.transactions.date.min()} hasta {self.transactions.date.max()}')
    
    def fill_empty_and_not_valid_values(self):
        # ventas menores a 0 las consideramos como error, colocamos 0
        self.transactions.loc[self.transactions.units_sold < 0, 'units_sold'] = 0
        # valores de ventas inexistentes, colocamos 0
        self.transactions.loc[self.transactions.units_sold.isna(), 'units_sold'] = 0
        # valores de stock es porque el stock quedó vacío, así que colocamos 0
        self.transactions.loc[self.transactions.stock.isna(), 'stock'] = 0
        # stock menores a 0 las consideramos como error, colocamos 0
        self.transactions.loc[self.transactions.stock < 0, 'stock'] = 0
    
    def fill_values_for_lockdown_period(self):
        transactions = self.transactions
        transactions['date'] = pd.to_datetime(transactions.date.astype(str), format='%Y-%m-%d')
        transactions['units_sold_original'] = transactions.units_sold
        lockdown_period = pd.to_datetime(['2020-03-15', '2020-05-09'], format='%Y-%m-%d')
        post_lockdown_period = pd.to_datetime(['2020-05-10', '2020-05-31'], format='%Y-%m-%d')

        tx = transactions[(transactions.date >= lockdown_period[0]) & (transactions.date <= lockdown_period[1])]
        transactions.loc[(transactions.date >= lockdown_period[0]) & (transactions.date <= lockdown_period[1]), 'units_sold'] = ((tx.units_sold_last_52_weeks + tx.units_sold_last_12_weeks + tx.units_sold_last_8_weeks) / 3)

        tx = transactions[(transactions.date >= post_lockdown_period[0]) & (transactions.date <= post_lockdown_period[1])]
        transactions.loc[(transactions.date >= post_lockdown_period[0]) & (transactions.date <= post_lockdown_period[1]), 'units_sold'] = ((tx.units_sold_last_52_weeks + tx.units_sold_last_12_weeks + tx.units_sold_original) / 3)

        self.transactions = transactions
    
    def save_transactions(self, output_file):
        self.transactions.to_csv(output_file, index = False, sep=';')
        print(f'{self.transactions.shape[0]} transacciones guardadas en: {output_file}')

    

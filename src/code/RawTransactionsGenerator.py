import os
import pandas as pd

class RawTransactionsGenerator:
    data_path = None
    sales = None
    calendar = None
    stock = None
    promotions = None
    transactions = None

    def __init__(self, data_path):
        self.data_path = data_path
        self.verify_data_dir()
        self.load_data()
    
    def verify_data_dir(self):
        # Verificando que los datos se encuentran en el directorio
        print(f'leyendo ficheros de {self.data_path}')
        print(os.listdir(self.data_path))

    def load_data(self):
        self.sales = pd.read_csv(self.data_path + '/Venta.csv', sep=';')
        self.calendar = pd.read_csv(self.data_path + '/Calendarios.csv', sep=';')
        self.stock = pd.read_csv(self.data_path + '/Stock.csv', sep=';')
        self.promotions = pd.read_csv(self.data_path + '/Promos.csv', sep=';')
        self.print_files_info()
    
    def print_files_info(self):
        print(f'Existen {self.sales.shape[0]} registros de venta desde {self.sales.fecha.min()}')
        print(f'Existen {self.calendar.shape[0]} registros de calendario desde {self.calendar.fecha.min()}')
        print(f'Existen {self.stock.shape[0]} registros de stock desde {self.stock.fecha.min()}')
        print(f'Existen {self.promotions.shape[0]} registros de promociones desde {self.promotions.fechaIni.min()}')
    
    def group_rows(self):
        self.sales = self.sales.groupby(by=['fecha', 'sku']).sum().reset_index()
        self.stock = self.stock.groupby(by=['fecha', 'sku']).sum().reset_index()
        self.calendar = self.calendar.groupby(by=['fecha', 'sku']).max().reset_index()
    
    def generate_transactions(self):
        self.group_rows()
        txs = pd.merge(self.sales, self.stock, how='left', on=['fecha', 'sku'])
        txs = pd.merge(txs, self.calendar, how='left', on=['fecha', 'sku'])
        self.transactions = txs
    
    def isProm(self, t):
        return any((self.promotions.sku == t.sku) & (self.promotions.fechaIni <= t.fecha) & (self.promotions.fechaFin >= t.fecha))
    
    def add_promotions_to_transactions(self):
        self.transactions['bolProm'] = self.transactions.apply(lambda t: self.isProm(t), axis=1)
        self.transactions.bolProm = self.transactions.bolProm.astype(int)
        print("transacciones en promoción: " + str(self.transactions[(self.transactions.bolProm == 1)].shape[0]))
    
    def save_transactions(self, output_file):
        self.transactions['date'] = pd.to_datetime(self.transactions.date.astype(str), format='%Y%m%d')
        self.transactions.columns = ['date', 'sku', 'units_sold', 'stock', 'is_open', 'is_holiday', 'is_prom']
        self.transactions.to_csv(output_file, index = False, sep=';')
        print(f'{self.transactions.shape[0]} transacciones guardadas en: {output_file}')
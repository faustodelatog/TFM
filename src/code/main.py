import RawTransactionsCreator as rtc

class Main:
    data_path = None
    raw_transactions_path = None

    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_transactions_path = f'{self.data_path}/output/_1_transactions_raw.csv'

    def create_raw_transactions(self):
        x = rtc.RawTransactionsCreator(self.data_path)
        x.create_transactions()
        x.add_promotions_to_transactions()
        x.save_transactions(self.raw_transactions_path)
    
    def otra(self):
        print('otra')
        print(self.raw_transactions_path)

DATA_PATH = '../data'
main = Main(DATA_PATH)
main.create_raw_transactions()
main.otra()


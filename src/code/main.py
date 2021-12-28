import RawTransactionsGenerator as rtg
import PreviousValuesGenerator as pvg
import TransactionsFiller as tf
import FeatureExtractor as fe

class Main:
    data_path = None
    raw_transactions_path = None
    transactions_with_past_values = None
    transactions_with_empty_values_filled = None
    transactions_with_empty_values_filled_and_past_values = None
    transactions_with_features = None

    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_transactions_path = f'{self.data_path}/output/_1_transactions_raw.csv'
        self.transactions_with_past_values = f'{self.data_path}/output/_2_transactions_with_past_values.csv'
        self.transactions_with_empty_values_filled = f'{self.data_path}/output/_3_transactions_with_empty_values_filled.csv'
        self.transactions_with_empty_values_filled_and_past_values = f'{self.data_path}/output/_4_transactions_with_empty_values_filled_and_past_values.csv'
        self.transactions_with_features = f'{self.data_path}/output/_5_transactions_with_features.csv'

    def generate_raw_transactions(self):
        raw_transactions_generator = rtg.RawTransactionsCreator(self.data_path)
        raw_transactions_generator.generate_transactions()
        raw_transactions_generator.add_promotions_to_transactions()
        raw_transactions_generator.save_transactions(self.raw_transactions_path)
    
    def generate_previous_values(self):
        previous_values_generator = pvg.PreviousValuesGenerator(self.raw_transactions_path)
        previous_values_generator.add_previous_values()
        previous_values_generator.save_transactions(self.transactions_with_past_values)

    def fill_values(self):
        transactions_filler = tf.TransactionsFiller(self.transactions_with_past_values)
        transactions_filler.fill_empty_and_not_valid_values()
        transactions_filler.fill_values_for_lockdown_period()
        transactions_filler.save_transactions(self.transactions_with_empty_values_filled)

    def generate_previous_values_for_non_empty_values(self):
        previous_values_generator = pvg.PreviousValuesGenerator(self.transactions_with_empty_values_filled)
        previous_values_generator.add_previous_values()
        previous_values_generator.save_transactions(self.transactions_with_empty_values_filled_and_past_values)

    def extract_features(self):
        feature_extractor = fe.FeatureExtractor(self.transactions_with_empty_values_filled_and_past_values)
        feature_extractor.add_date_features()
        feature_extractor.add_season()
        feature_extractor.add_has_stock()
        feature_extractor.add_sold_out()
        feature_extractor.format_sku()
        feature_extractor.save_transactions(self.transactions_with_features)

DATA_PATH = '../data'
main = Main(DATA_PATH)
#main.generate_raw_transactions()
#main.generate_previous_values()
#main.fill_values()
#main.generate_previous_values_for_non_empty_values()
main.extract_features()

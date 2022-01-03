import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras import activations
from keras.optimizers import Adam
from statsmodels.tsa.seasonal import STL

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

class PredictionModel:

    # SARIMAX
    def __remove_heteroscedasticity(self, time_series):
        result = np.log(time_series)
        result[np.isinf(result)] = result[~np.isinf(result)].min()
        result[np.isnan(result)] = result[~np.isnan(result)].min()
        return result

    def __add_heteroscedasticity(self, time_series):
        result = np.exp(time_series)
        return result

    def __extract_trend(self, time_series):
        x = np.array(range(time_series.shape[0])).reshape((-1,1))
        reg_model = LinearRegression(normalize=True, fit_intercept=True)
        reg_model.fit(x, time_series)
        trend = reg_model.predict(x)
        return trend

    def predict_sarimax(self, training_time_serie, dates_to_predict):
        no_het = self.__remove_heteroscedasticity(training_time_serie)
        no_het_trend = self.__extract_trend(no_het)
        no_het_no_trend = no_het - no_het_trend

        order = (4,0,3)
        seasonal_order = (1,0,1,7)
        model = SARIMAX(no_het_no_trend, order=order, seasonal_order=seasonal_order)
        fit = model.fit()
        pred = fit.predict(start=1, end=len(dates_to_predict))

        result = pred + np.mean(no_het_trend)
        result = self.__add_heteroscedasticity(result)
        return result
    
    # NEURAL NETWORK
    LOOK_BACK = 14
    FUTURE = 1
    EPOCHS = 25
    LEARNING_RATE = 0.005

    def split_data_by_date(self, data, date):
        train = data[data.index < date].copy()
        validation = data[data.index >= date].copy()
        return [train, validation]

    def create_dataset(self, series, lookback, future):
            features_set = [] 
            labels = []  
            for i in range(lookback, series.shape[0] - future):  
                features_set.append(series[i - lookback:i])
                labels.append(series[i + future - 1])

            features_set, labels = np.array(features_set), np.array(labels)
            features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

            return features_set, labels

    def add_predictions(self, df, preds, lookback, future, column='lstm_pred'):
            predictions = np.ones(len(df)) * np.nan
            predictions[lookback+future-1:-1] = preds[:, 0]
            df.loc[:, column] = predictions

            return df

    def scale(self, series, scaler, lookback, future):
        norm = pd.DataFrame(scaler.transform(np.array(series).reshape(-1,1)), columns=['units_sold'])
        dataset = self.create_dataset(norm.units_sold, lookback, future)
        return dataset

    def lstm_model(self, X_train, y_train, X_validation, y_validation):
        learning_rate = self.LEARNING_RATE
        epochs = self.EPOCHS

        model = Sequential()
        model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
        model.add(Dense(units=20, activation=activations.relu))
        model.add(Dense(units=1))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
        history = model.fit(X_train, y_train, epochs = epochs, validation_data = (X_validation, y_validation), shuffle=False)

        # Loss evolution
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        return model

    def predict_lstm(self, training_data, min_validation_date, test_data):
        # Variables
        lookback = self.LOOK_BACK
        future = self.FUTURE

        # Train validation and test sets
        train_and_validation = self.split_data_by_date(training_data, min_validation_date)
        train = train_and_validation[0]
        validation = train_and_validation[1]
        test = test_data

        #Scale sets
        scaler = StandardScaler()
        scaler.fit(np.array(train.units_sold).reshape(-1,1))
        train_set = self.scale(train.units_sold, scaler, lookback, future)
        validation_set = self.scale(validation.units_sold, scaler, lookback, future)
        test_set = self.scale(test.units_sold, scaler, lookback, future)
        print("units sold (mean): ", scaler.mean_)

        X_train = train_set[0]
        y_train = train_set[1]
        X_validation = validation_set[0]
        y_validation = validation_set[1]
        X_test = test_set[0]

        # Model 
        model = self.lstm_model(X_train, y_train, X_validation, y_validation)

        train_predict = scaler.inverse_transform(model.predict(X_train))
        validation_predict = scaler.inverse_transform(model.predict(X_validation))
        test_predict = scaler.inverse_transform(model.predict(X_test))

        self.add_predictions(train, train_predict, lookback, future)
        self.add_predictions(validation, validation_predict, lookback, future)
        self.add_predictions(test, test_predict, lookback, future)

        result = pd.concat([train, validation, test])
        
        return result.lstm_pred
    
    # PROPHET
    def train_model(self, training_data, regressors):
        columns = regressors.copy()
        columns.append('units_sold')
        df = training_data[columns].reset_index()
        df = df.rename(columns={'date':'ds', 'units_sold':'y'})
        m = Prophet()
        for regressor in regressors:
            m.add_regressor(regressor)
        m.fit(df)
        print('fitted')
        return m

    def predict_future_prophet(self, model, time_serie):
        future = time_serie.reset_index().rename(columns={'date':'ds'})
        forecast = model.predict(future)
        return forecast

    def predict_prophet(self, training_data, test_data, regressors = []):
        model = self.train_model(training_data, regressors)
        forecast = self.predict_future_prophet(model, training_data.append(test_data))
        return model, forecast
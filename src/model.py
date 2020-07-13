# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:15:11 2019

@author: Ignacio
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import datetime as dt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_gen import datagen, dataset

kraken_tickers = ['BTC', 'ETH', 'XMR', 'XRP', 'XLM', 'LTC', 'EOS',
                  'BCH', 'DASH', 'ETC', 'GNO', 'QTUM', 'REP', 'USDT']

class LSTM_model:
    '''LSTM model'''
    def __init__(self, config, datasource):
        self.config = config
        self.opt = None
        self.checkpoint = None
        self.tensorboard = None
        self.early_stopping = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.weights = None
        self.train_data_gen = None
        self.test_data_gen = None
        self.datasource = datasource
        self.set_params()

    def set_params(self):
        '''prepares the hypermarameters to be used by the model'''

        if self.config == 1:
            self.model_params = {
                'optimizer': Adam,
                'epochs': 150,
                'learning_rate': 0.001,
                'decay': 1e-6,
                'dropout': 0.5,
                'SEQ_LEN': 90,
                'NAME': f"LSTM-{self.config}-\
                {time.strftime('%Y-%m-%d %H-%M-%S')}",
                'patience': 100,
                'lstm_neurons': [256, 256, 128],
                'shuffle': True,
                'batch_size': 32,
                'steps_per_epoch': 50
                }
        elif self.config == 2:
            self.model_params = {
                'optimizer': Adam,
                'epochs': 150,
                'learning_rate': 0.001,
                'decay': 1e-6,
                'dropout': 0.5,
                'SEQ_LEN': 10,
                'NAME': f"LSTM-{self.config}-\
                {time.strftime('%Y-%m-%d %H-%M-%S')}",
                'patience': 100,
                'lstm_neurons': [256, 256, 128],
                'shuffle': True,
                'batch_size': 32,
                'steps_per_epoch': 50
                }
        else:
            assert 0, "Bad Config creation: " + self.config.name

    @staticmethod
    def get_weights(dependent_var):
        '''calculate classification weights'''
        classes, cnt = np.unique(dependent_var, return_counts=True, axis=0)
        weights = 1/(cnt/cnt.sum())
        weights = weights/weights.sum()
        return dict(zip(classes, weights))

    def add_optimizer(self):
        '''add optimiser to be used by LSTM'''
        self.opt = self.model_params['optimizer'](
            lr=self.model_params['learning_rate'],
            decay=self.model_params['decay'], clipnorm=1.)

    def create_model(self):
        '''create LSTM model'''
        self.model = Sequential()
        self.model.add(LSTM(units=self.model_params['lstm_neurons'][0],
                            return_sequences=True,
                            input_shape=(self.model_params['SEQ_LEN'],
                                         self.train_generator.df.shape[1]-3)))
        self.model.add(Dropout(self.model_params['dropout']))
        self.model.add(BatchNormalization())

        if len(self.model_params['lstm_neurons']) > 1:
            for i in self.model_params['lstm_neurons'][1:]:
                self.model.add(LSTM(units=i, return_sequences=True if i !=
                                    self.model_params['lstm_neurons'][-1]
                                    else False))
                self.model.add(Dropout(self.model_params['dropout']))
                self.model.add(BatchNormalization())

        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(self.model_params['dropout']))

        self.model.add(Flatten())
        self.model.add(Dense(3, activation='softmax'))

        self.add_optimizer()

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=self.opt, metrics=['accuracy'])
        self.model.summary()
        '''
        self.tensorboard = TensorBoard(log_dir=f"./LSTM_Models/LSTM_logs/\
                                       {self.model_params['NAME']}")'''
        # run tensorboard from the console with next comment to follow training
        # tensorboard --logdir=LSTM_Models/LSTM_logs/

        self.checkpoint = ModelCheckpoint('./LSTM_Models/Models/LSTM_T1-Best-'
                                          + self.datasource,
                                          monitor='val_accuracy', verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True, mode='max')
        self.early_stopping =\
            EarlyStopping(monitor='val_loss',
                          patience=self.model_params['patience'])

    def load_model(self):
        '''loading a trained model'''
        self.create_model()

        self.model.load_weights('./LSTM_Models/Models/LSTM_T1-Best-'
                                + self.datasource)

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=self.opt, metrics=['accuracy'])

    def input_data(self):
        '''timeseries data creation'''
        self.weights = self.get_weights(self.train_generator.df.iloc[:, -1])

        self.ttl_batches = int((len(self.train_generator.df) -
                                self.model_params['SEQ_LEN']) /
                               self.model_params['batch_size'])

    def train(self, train_gen, validation_gen):
        '''train on the dataset provided'''
        self.train_generator = train_gen
        self.validation_generator = validation_gen
        self.input_data()
        self.create_model()

        history = \
            self.model.fit(
                x=self.train_generator,
                epochs=self.model_params['epochs'],
                shuffle=self.model_params['shuffle'],
                validation_data=self.validation_generator,
                steps_per_epoch=self.model_params['steps_per_epoch'],
                # class_weight=self.weights,
                callbacks=[# self.tensorboard,
                           self.checkpoint,
                           self.early_stopping])
        print(history)

    def test(self, start_date=dt.datetime.today() -
             dt.timedelta(days=15), end_date=dt.datetime.today()):
        '''test and return confusion_matrix and classification_report'''

        self.set_params()
        data = dataset(self.datasource,
                       max_date=end_date)  # TEST
        data.prepare_test_dataset()
        self.test_data_gen = datagen(data.sdf,
                                     gen_length=self.model_params['SEQ_LEN'],
                                     start_date=start_date)
        self.x_train =\
            self.test_data_gen.df.loc[self.test_data_gen.df.date.between(
                start_date, end_date)].iloc[:, :-1]
        self.y_train =\
            self.test_data_gen.df.loc[self.test_data_gen.df.date.between(
                start_date, end_date)].iloc[:, -1]
        self.load_model()
        y_lstm_pred =\
            self.model.predict_generator(self.test_data_gen,
                                         steps=len(self.test_data_gen))
        y_lstm_pred = np.argmax(y_lstm_pred, axis=1)
        self.test_data_gen.result = self.test_data_gen.result[:-10]
        print("LSTM: Predictions have finished")
        cm_lstm = confusion_matrix(self.test_data_gen.result,
                                   y_lstm_pred)
        o_acc = np.around(np.sum(np.diag(cm_lstm)) / np.sum(cm_lstm)*100, 1)
        plt.title(f'Confusion Matrix \n Accuracy={o_acc}%', size=18)
        sn.heatmap(cm_lstm, fmt=".0f", annot=True, cbar=False,
                   annot_kws={"size": 15}, xticklabels=['Sell', 'Hold', 'Buy'],
                   yticklabels=['Sell', 'Hold', 'Buy'])
        plt.xlabel('Predicted Label', size=15)
        plt.ylabel('True Label', size=15)
        print(np.diag(cm_lstm).sum())
        cr_lstm = classification_report(self.test_data_gen.result,
                                        y_lstm_pred)
        print(cr_lstm)
        return cm_lstm, cr_lstm

    def predict(self, start_date=dt.datetime.today().date(),
                end_date=dt.datetime.today().date()):
        self.set_params()
        data = dataset(self.datasource,
                       min_date=start_date, max_date=end_date)  # TEST
        print("The dataset is", len(data), "datapoints long")
        data.prepare_test_dataset()
        self.test_data_gen = datagen(data.sdf,
                                     gen_length=self.model_params['SEQ_LEN'],
                                     start_date=start_date)
        print(self.test_data_gen.df.head())
        self.x_train =\
            self.test_data_gen.df.loc[self.test_data_gen.df.date.between(
                start_date, end_date)].iloc[:, :-1]
        print(self.x_train.columns)
        self.y_train =\
            self.test_data_gen.df.loc[self.test_data_gen.df.date.between(
                start_date, end_date)].iloc[:, -1]
        self.load_model()
        y_lstm_pred =\
            self.model.predict_generator(self.test_data_gen,
                                         steps=len(self.test_data_gen))
        return self.test_data_gen.ticks, y_lstm_pred

    def predict2(self, start_date=dt.datetime.today().date() -
                 dt.timedelta(days=120),
                 end_date=dt.datetime.today().date()):
        self.set_params()
        data = dataset(self.datasource,
                       min_date=start_date, max_date=end_date)
        print("The dataset is", len(data), "datapoints long")
        data.prepare_test_dataset()
        self.x_train = data.sdf.drop(['target'], axis=1)
        self.x_train = np.array(self.x_train.iloc[-90:, :])
        self.load_model()
        predictions = pd.DataFrame(columns=['sell', 'hold', 'buy'])
        results = pd.DataFrame(columns=['sym', 'date', 'target'])
        for sym in data.sdf.symbol.unique():
            temp = data.sdf.loc[data.sdf.symbol == sym]
            for date in temp.date[90:]:
                results =\
                    results.append({'sym': sym, 'date': date,
                                    'target':
                                    temp.loc[temp.date == date,
                                             ['adjusted_close']].values[0][0]},
                                   ignore_index=True)
                self.x_train = temp.loc[temp.date <= date].iloc[-90:, :]\
                    .drop(['symbol', 'date', 'target'], axis=1)
                self.x_train = np.array(self.x_train)
                self.x_train =\
                    np.reshape(self.x_train,
                               (1, self.x_train.shape[0],
                                self.x_train.shape[1])).astype('float32')
                predictions = predictions.append(pd.DataFrame(
                    self.model.predict(self.x_train),
                    columns=['sell', 'hold', 'buy']), ignore_index=True)
        results = pd.concat([results, predictions], axis=1)
        return results


if __name__ == '__main__':
    pass
    #model = LSTM_model(1, 'cry')
    #model.train()

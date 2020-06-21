# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:52:33 2020

@author: Ignacio
"""
import pandas as pd
from data_gen import dataset
import datetime as dt
from data_gen import datagen
from model import LSTM_model

# https://cloud.google.com/resource-manager/docs/creating-managing-projects
project_id = 'investing-management'

query = f'''SELECT * FROM price_data.SNP
    WHERE symbol in ('AAPL','ABBV','ABMD','MSFT','GOOG')'''

df = pd.io.gbq.read_gbq(query, project_id=project_id)

dataset = dataset(df, table='snp',
                  min_date=dt.datetime(2010, 1, 1, 0, 0, 0,
                                       tzinfo=dt.timezone.utc),
                  max_date=dt.datetime(2020, 6, 1, 0, 0, 0,
                                       tzinfo=dt.timezone.utc))
dataset.prepare_dataset()
model = LSTM_model(1, 'snp')
train_data_generator = datagen(df=dataset.traindata,
                               gen_length=model.model_params['SEQ_LEN'],
                               shuffle=True)
test_data_generator = datagen(df=dataset.testdata,
                              gen_length=model.model_params['SEQ_LEN'],
                              shuffle=True)
model.train(train_gen=train_data_generator, validation_gen=test_data_generator)

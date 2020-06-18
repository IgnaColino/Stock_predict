# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:52:33 2020

@author: Ignacio
"""
import pandas as pd

# https://cloud.google.com/resource-manager/docs/creating-managing-projects
project_id = 'investing-management'

query = f'''SELECT * FROM price_data.SNP
    WHERE symbol in ('AAPL','ABBV','ABMD')'''

df = pd.io.gbq.read_gbq(query, project_id=project_id)

from data_gen import dataset
import datetime as dt

dataset = dataset(df, table='snp',
                  min_date=dt.datetime(2010, 1, 1, 0, 0, 0,
                                       tzinfo=dt.timezone.utc),
                  max_date=dt.datetime(2020, 6, 1, 0, 0, 0,
                                       tzinfo=dt.timezone.utc))
dataset.prepare_dataset()
data = dataset.sdf

data2 = data.loc[
    data.date.between(dt.datetime(2010, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
                      dt.datetime.today())]
print(data.date.iloc[0], dt.datetime(2010, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
      dt.datetime.now(dt.timezone.utc))

from data_gen import datagen

datag = datagen(df=data)


datag.next()

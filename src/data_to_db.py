# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:28:49 2019

@author: Ignacio
"""
import pandas as pd
from io import StringIO


def to_pg(df, table_name, con):
    table_name = table_name.lower()
    raw = con.raw_connection()
    curs = raw.cursor()
    curs.execute("select * from information_schema.tables where table_name=%s",
                 (table_name,))
    if not bool(curs.rowcount):
        empty_table = pd.io.sql.get_schema(df, table_name, con=con)
        empty_table = empty_table.replace('"', '')
        curs.execute(empty_table)
        curs.execute("ALTER TABLE " + table_name +
                     " ADD PRIMARY KEY (symbol, date);")
    curs.execute("select symbol, date from " + table_name)
    ids = pd.DataFrame(curs.fetchall(), columns=['symbol', 'date'])
    ids['indicator'] = 1
    df = pd.merge(df, ids, how='left')
    df = df.loc[df.indicator != 1]
    df.drop('indicator', inplace=True, axis=1)
    data = StringIO()
    df.to_csv(data, header=False, index=False)
    data.seek(0)
    curs.copy_from(data, table_name, sep=',')
    curs.connection.commit()
    print(len(df), 'daily records added for',
          len(df.symbol.unique()), 'tickers')

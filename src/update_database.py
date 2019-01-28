# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:08:02 2019

@author: Ignacio
"""

from insert_data import to_pg
from get_data import get_adjusted_data
from sqlalchemy import create_engine
import os
import ast


def update_market_table(table='SNP'):
    data = get_adjusted_data(market='SNP', num_stocks=2)
    security = ast.literal_eval(os.getenv('PSQL_USER'))
    conn = create_engine("postgresql://" +
                         security['user'] + ":" +
                         security['password'] +
                         "@localhost/stock_data")
    to_pg(data, table, conn)


if __name__ == '__main__':
    update_market_table()

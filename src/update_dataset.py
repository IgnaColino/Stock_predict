  # -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:08:02 2019

@author: Ignacio
"""

from data_to_db import to_pg
from data_fetch import get_adjusted_data
from sqlalchemy import create_engine
import os
import ast


class data_updater:

    def __init__(self, table, num_stocks):
        self.table = table
        self.num_stocks = num_stocks

    def update_market_table(self):
        data = get_adjusted_data(market=self.table, num_stocks=self.num_stocks)
        security = ast.literal_eval(os.getenv('PSQL_USER'))
        conn = create_engine("postgresql://" +
                             security['user'] + ":" +
                             security['password'] +
                             "@localhost/stock_data")
        to_pg(data, self.table, conn)


def main():
    updater = data_updater('ASX', 5)
    updater.update_market_table()


if __name__ == '__main__':
    main()

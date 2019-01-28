# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 20:17:57 2019

@author: Ignacio
"""
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import ast


def create_db(db_name="stock_data"):
    security = ast.literal_eval(os.getenv('PSQL_USER'))
    con = psycopg2.connect("dbname='postgres' user=" +
                           security['user'] +
                           " host='localhost' password=" +
                           security['password'])
    try:
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) = 0 FROM pg_catalog.pg_database\
                   WHERE datname = '"+db_name+".db'")
        not_exists_row = cur.fetchone()
        not_exists = not_exists_row[0]
        print(not_exists)
        if not_exists:
            cur.execute("CREATE DATABASE "+db_name)
    except Exception as e:
        print(e)
    finally:
        if con:
            con.close()


if __name__ == '__main__':
    create_db()

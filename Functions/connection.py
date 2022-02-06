import psycopg2

# Connect
def conn():
    sqluser = 'sqluser'
    password = 'password'
    dbname = 'dbname'
    schema_name = 'schema_name'
    sqlhost = '127.0.0.1'
    sqlport = 5432

    conn = psycopg2.connect(dbname=dbname, user=sqluser, host=sqlhost, port=sqlport, password=password)
    query_schema = 'SET search_path to public,' + schema_name + ';'

    return conn, query_schema

import psycopg2
conn = psycopg2.connect(
    dbname="TechNoVanza",
    user="postgres",
    password="4315",
    host="localhost",
    port=5432
)
print("Connection successful")
conn.close()

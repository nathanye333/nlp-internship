import mysql.connector
import pandas as pd
conn = mysql.connector.connect(
host='127.0.0.1', user='root', password='root', database='real_estate', port=3308)
query = """
SELECT L_ListingID, L_Address, L_City, L_Keyword2 as beds,
LM_Dec_3 as baths, L_SystemPrice as price, L_Remarks as remarks
FROM rets_property
WHERE L_Remarks IS NOT NULL AND LENGTH(L_Remarks) > 50
ORDER BY RAND() LIMIT 1000
"""
df = pd.read_sql(query, conn)
df.to_csv('data/processed/listing_sample.csv', index=False)
conn.close()
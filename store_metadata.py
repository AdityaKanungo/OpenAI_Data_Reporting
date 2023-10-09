import sqlite3
import json

# Connect to the SQLite database
conn = sqlite3.connect('healthcare.db')
cursor = conn.cursor()

# Fetch list of tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [table[0] for table in cursor.fetchall()]

metadata = {}

# Fetch schema for each table
for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    
    column_data = []
    for column in columns:
        column_info = {
            'id': column[0],
            'name': column[1],
            'type': column[2],
            'not_null': bool(column[3]),
            'default_value': column[4],
            'primary_key': bool(column[5])
        }
        column_data.append(column_info)
    
    metadata[table] = {'columns': column_data}

    # Fetch indices related to the table
    cursor.execute(f"PRAGMA index_list({table})")
    indices = cursor.fetchall()

    index_data = []
    for index in indices:
        cursor.execute(f"PRAGMA index_info({index[1]})")
        index_info = cursor.fetchall()
        indexed_columns = [col[2] for col in index_info]
        index_data.append({
            'name': index[1],
            'unique': bool(index[2]),
            'columns': indexed_columns
        })

    metadata[table]['indices'] = index_data

    # Fetch foreign key relationships for the table
    cursor.execute(f"PRAGMA foreign_key_list({table})")
    foreign_keys = cursor.fetchall()
    
    fk_data = []
    for fk in foreign_keys:
        fk_info = {
            'id': fk[0],
            'seq': fk[1],
            'table': fk[2],
            'from': fk[3],
            'to': fk[4],
            'on_update': fk[5],
            'on_delete': fk[6],
            'match': fk[7]
        }
        fk_data.append(fk_info)

    metadata[table]['foreign_keys'] = fk_data

# Store metadata in a JSON file
with open('db_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

conn.close()

print("Metadata stored in db_metadata.json")

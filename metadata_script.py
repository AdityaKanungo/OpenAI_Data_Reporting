import re
import json

sql_script = """ ... """  # Use your SQL script here

segments = [seg.strip() for seg in sql_script.split("GO") if "CREATE TABLE" in seg]

metadata_dict = {}

for segment in segments:
    # Extract table name
    table_match = re.search(r"CREATE TABLE \[?(\w+)\]?", segment)
    table_name = table_match.group(1) if table_match else None

    # Extract columns
    columns = re.findall(r"\[?(\w+)\]?\s+(\w+\s*\(?[\w\s,]*\)?\s*\w*)( NOT NULL)?( PRIMARY KEY)?[,)]", segment)
    
    columns_list = []
    for idx, (col_name, col_type, not_null, primary_key) in enumerate(columns):
        columns_list.append({
            "id": idx,
            "name": col_name,
            "type": col_type.strip(),
            "not_null": True if not_null else False,
            "default_value": None,  # This can be enhanced further to extract actual default values
            "primary_key": True if primary_key else False
        })

    # As of now, we're not extracting foreign_keys and indices. Placeholders are kept for future extensions
    metadata_dict[table_name] = {
        "columns": columns_list,
        "indices": [],
        "foreign_keys": []
    }

# Serialize the dictionary to JSON and save to a file
with open('output.json', 'w') as json_file:
    json.dump(metadata_dict, json_file, indent=4)

import re
import json

sql_script = """
CREATE TABLE sample_table (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    birthdate DATE
);
"""


# Split the script into segments using 'GO' as delimiter
segments = [seg.strip() for seg in sql_script.split("GO") if "CREATE TABLE" in seg]

all_tables_metadata = []

for segment in segments:
    # Extract table name
    table_match = re.search(r"CREATE TABLE \[?(\w+)\]?", segment)
    table_name = table_match.group(1) if table_match else None

    # Extract columns
    columns = re.findall(r"\[?(\w+)\]?\s+(\w+\s*\(?[\w\s,]*\)?\s*\w*)[,)]", segment)
    
    # Construct the dictionary for each table
    table_metadata = {
        "table_name": table_name,
        "columns": []
    }

    for col_name, col_type in columns:
        col_type = col_type.strip()  # Clean up the column type string
        table_metadata["columns"].append({
            "column_name": col_name,
            "column_type": col_type
        })

    all_tables_metadata.append(table_metadata)

# Serialize the list of dictionaries to JSON and save to a file
with open('output.json', 'w') as json_file:
    json.dump(all_tables_metadata, json_file, indent=4)

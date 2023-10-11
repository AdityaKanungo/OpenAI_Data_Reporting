import re
import json

sql_script = """
CREATE TABLE sample_table (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    birthdate DATE
);
"""


# Find all CREATE TABLE segments
create_statements = re.findall(r"CREATE TABLE \[?(\w+)\]? \{(.*?)\} ON \[?\w+\]?", sql_script, re.DOTALL)

# Initialize an empty list to store table metadata for all tables
all_tables_metadata = []

for create_stmt in create_statements:
    table_name = create_stmt[0]
    columns_section = create_stmt[1]
    
    columns = re.findall(r"\[?(\w+)\]?\s+(\w+\s*\(?[\w\s,]*\)?\s*\w*)[,)]", columns_section)
    
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

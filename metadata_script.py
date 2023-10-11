import re
import json

sql_script = """
CREATE TABLE sample_table (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    birthdate DATE
);
"""

# Extract table name
table_match = re.search(r"CREATE TABLE \[?(\w+)\]?", sql_script)
table_name = table_match.group(1) if table_match else None

# Extract columns
columns = re.findall(r"\[?(\w+)\]?\s+(\w+\s*\(?[\w\s,]*\)?\s*\w*)[,)]", sql_script)

# Construct the dictionary for JSON output
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

# Serialize the dictionary to JSON and print
json_output = json.dumps(table_metadata, indent=4)
print(json_output)

import re
import json

def extract_metadata_from_sql(filename):
    with open(filename, 'r') as f:
        content = f.read()

    metadata = {}

    # Regular expression patterns
    table_pattern = re.compile(r"CREATE TABLE ([^\s(]+)\s*\(([^;]+)\)", re.IGNORECASE | re.DOTALL)
    column_pattern = re.compile(r"([^\s,]+)\s+([^\s,]+)", re.IGNORECASE)
    primary_key_pattern = re.compile(r"PRIMARY KEY\s*\(([^\)]+)\)", re.IGNORECASE)
    unique_pattern = re.compile(r"UNIQUE\s*\(([^\)]+)\)", re.IGNORECASE)
    foreign_key_pattern = re.compile(r"FOREIGN KEY\s*\(([^\)]+)\)\s*REFERENCES\s*([^\s(]+)\s*\(([^\)]+)\)", re.IGNORECASE)
    default_pattern = re.compile(r"DEFAULT\s+([^\s,]+)", re.IGNORECASE)

    for table_match in table_pattern.findall(content):
        table_name = table_match[0].strip()
        columns_content = table_match[1].strip()
        columns = [col.strip() for col in columns_content.split(',')]

        table_data = {
            "columns": {},
            "primary_key": [],
            "unique": [],
            "foreign_keys": [],
            "defaults": {}
        }

        for column in columns:
            # Columns and data types
            match = column_pattern.search(column)
            if match:
                column_name = match.group(1).strip()
                column_type = match.group(2).split(' ')[0].strip()  # Simplified type extraction
                table_data["columns"][column_name] = column_type

                # Default values
                default_match = default_pattern.search(column)
                if default_match:
                    default_value = default_match.group(1).strip()
                    table_data["defaults"][column_name] = default_value

        # Primary keys
        pk_match = primary_key_pattern.search(columns_content)
        if pk_match:
            keys = pk_match.group(1).strip().split(',')
            table_data["primary_key"] = [key.strip() for key in keys]

        # Unique constraints
        unique_match = unique_pattern.search(columns_content)
        if unique_match:
            unique_cols = unique_match.group(1).strip().split(',')
            table_data["unique"] = [col.strip() for col in unique_cols]

        # Foreign keys
        for fk_match in foreign_key_pattern.findall(columns_content):
            fk_data = {
                "columns": [col.strip() for col in fk_match[0].split(',')],
                "reference_table": fk_match[1].strip(),
                "reference_columns": [col.strip() for col in fk_match[2].split(',')]
            }
            table_data["foreign_keys"].append(fk_data)

        metadata[table_name] = table_data

    return metadata

def save_metadata_to_json(metadata, output_filename):
    with open(output_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

# Example usage:
filename = "path_to_your_sql_file.sql"
metadata = extract_metadata_from_sql(filename)
save_metadata_to_json(metadata, "output_metadata.json")

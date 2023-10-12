# Splitting the SQL script based on the "CREATE TABLE" keyword


import re
import json

sql_script = """


CREATE TABLE [Categories] (

[CategoryID] int IDENTITY(1,1) NOT NULL,

[CategoryName] nvarchar(15) NOT NULL,

[Description] varchar(MAX) NULL,

[Picture] image NULL) 

CREATE TABLE [CustomerCustomerDemo] (

[CustomerID] nchar(5) NOT NULL, [CustomerTypeID] nchar(10) NOT NULL)


CREATE TABLE [Customer Demographics] ( [CustomerTypeID] nchar(10) NOT NULL,
[CustomerDesc] varchar(MAX) NULL) 

CREATE TABLE [Customers] (

[CustomerID] nchar(5) NOT NULL,

[CompanyName] nvarchar(40) NOT NULL, [ContactName] nvarchar(30) NULL,

[ContactTitle] nvarchar(30) NULL, [Email] nvarchar(70) NULL,

[Password] nvarchar(30) NULL, [Address] nvarchar(60) NULL,

[City] nvarchar(15) NULL,

[Region] nvarchar(15) NULL,

[PostalCode] nvarchar(10) NULL,

[Country] nvarchar(15) NULL,

[Phone] nvarchar(24) NULL,

[Fax] nvarchar(24) NULL)



CREATE TABLE [Employees] ( [EmployeeID] int IDENTITY(1,1) NOT NULL,

[LastName] nvarchar(20) NOT NULL, [FirstName] nvarchar(10) NOT NULL,

[Title] nvarchar(30) NULL, [TitleOfCourtesy] nvarchar(25) NULL,

[BirthDate] datetime NULL, [HireDate] datetime NULL,

[Address] nvarchar(60) NULL, [City] nvarchar(15) NULL,

[Region] nvarchar(15) NULL,

[PostalCode] nvarchar(10) NULL,

[Country] nvarchar(15) NULL,

[HomePhone] nvarchar(24) NULL, [Extension] nvarchar(4) NULL,

[Photo] image NULL,

[Notes] varchar(MAX) NULL,

[ReportsTo] int NULL,

[PhotoPath] nvarchar(255) NULL)



CREATE TABLE [EmployeeTerritories] (

[EmployeeID] int NOT NULL, [TerritoryID] nvarchar(20) NOT NULL)



CREATE TABLE [Order Details] ( [OrderID] int NOT NULL,

[ProductID] int NOT NULL,

[UnitPrice] money NOT NULL,

[Quantity] smallint NOT NULL,

[Discount] real NOT NULL)

CREATE TABLE [Orders] (

[OrderID] int IDENTITY(1,1) NOT NULL, [CustomerID] nchar(5) NULL,

[EmployeeID] int NULL,

[OrderDate] datetime NULL, [RequiredDate] datetime NULL,

[ShippedDate] datetime NULL,

[ShipVia] int NULL, [Freight] money NULL,

[ShipName] nvarchar(40) NULL,

[ShipAddress] nvarchar(60) NULL, [ShipCity] nvarchar(15) NULL,

[ShipRegion] nvarchar(15) NULL,

[ShipPostalCode] nvarchar(10) NULL,

[ShipCountry] nvarchar(15) NULL)

CREATE TABLE [Products] ( [ProductID] int IDENTITY(1,1) NOT NULL,

[ProductName] nvarchar(40) NOT NULL, [SupplierID] int NULL,

[CategoryID] int NULL, [QuantityPerUnit] nvarchar(20) NULL,

[UnitPrice] money NULL, [UnitsInStock] smallint NULL,

[UnitsOnOrder] smallint NULL,

[ReorderLevel] smallint NULL,

[Discontinued] bit NOT NULL) 

GO CREATE TABLE [Region] (

[RegionID] int NOT NULL, [RegionDescription] nchar(50) NOT NULL)


CREATE TABLE [Shippers] (

[ShipperID] int IDENTITY(1,1) NOT NULL, 
[CompanyName] nvarchar(40) NOT NULL,

[Phone] nvarchar(24) NULL)



CREATE TABLE [Suppliers] ( [SupplierID] int IDENTITY(1,1) NOT NULL,

[CompanyName] nvarchar(40) NOT NULL, 

[ContactName] nvarchar(30) NULL,

[ContactTitle] nvarchar(30) NULL,

[Address] nvarchar(60) NULL,

[City] nvarchar(15) NULL,

[Region] nvarchar(15) NULL, 
[PostalCode] nvarchar(10) NULL,

[Country] nvarchar(15) NULL, 
[Phone] nvarchar(24) NULL,

[Fax] nvarchar(24) NULL)

CREATE TABLE [Territories] (

[TerritoryID] nvarchar(20) NOT NULL,

[TerritoryDescription] nchar(50) NOT NULL, 

[RegionID] int NOT NULL)

"""


segments = [seg.strip() for seg in re.split(r"(?i)CREATE TABLE", sql_script) if seg.strip()]

# Adjusted code to capture table structure

metadata_dict = {}

for idx, segment in enumerate(segments):
    # Extract table name
    table_match = re.search(r"\[(\w+)\]", segment)
    table_name = table_match.group(1) if table_match else None
    
    # Check if table name is found
    if not table_name:
        continue

    # Isolate the column definitions portion of the segment
    columns_segment = re.search(r"\((.*)\)\s*$", segment, re.DOTALL)
    if not columns_segment:
        continue
    columns_segment = columns_segment.group(1).strip()

    # Adjust the columns pattern regex
    columns_pattern = r"\[(\w+)\]\s+([\w\(\),]+)(\s+IDENTITY\(\d+,\d+\))?(\s+NOT NULL)?(\s+NULL)?(\s+PRIMARY KEY)?"
    columns = re.findall(columns_pattern, columns_segment)

    columns_list = []
    for col_idx, (col_name, col_type, identity, not_null, is_null, primary_key) in enumerate(columns):
        columns_list.append({
            "id": col_idx,
            "name": col_name,
            "type": col_type.strip() + (identity.strip() if identity else ""),
            "not_null": True if not_null else False,
            "default_value": None, 
            "primary_key": True if primary_key else False
        })

    metadata_dict[table_name] = {
        "columns": columns_list,
        "indices": [],
        "foreign_keys": []
    }

# Serialize the dictionary to JSON and save to a file
output_file_path = 'output_fixed.json'
with open(output_file_path, 'w') as json_file:
    json.dump(metadata_dict, json_file, indent=4)

output_file_path

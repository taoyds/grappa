# result_present

In data_augment/, there are two scripts to generate augmented data using SCFG:
### 1. parse_table.py
Usage: `python parse_table.py wikitable`
Parse raw dataset into standard format and store it in json format.
The default dataset path is data/ and the default output path is `temp/wikitable_processed.json`
### 2. generate_augmented_data.py
Usage: `python generate_augmented_data.py temp/wikitable_processed.json data/nlsql_templates.txt data/sql_components.json [OUTPUT PATH] [Number of data]`. Some hyperparameters are defined in the first few lines of the script.

Generated augmented data is in data/augment_data.txt
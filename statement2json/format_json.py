import json
import pathlib

STATEMENTS_DIR = pathlib.Path("..") / "json_statements"
STATEMENTS_FILES = STATEMENTS_DIR.glob('*.json')

redo_files = [STATEMENTS_DIR / "valores_2022_page-0001.json"]

for json_file in redo_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        str_json = json.load(f)

    json_dict = json.loads(str_json)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f)
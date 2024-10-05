from pathlib import Path
import os
import base64
import requests
import json

from dotenv import load_dotenv
load_dotenv()

IMAGES = list(Path('..').glob('images/*.jpg'))
SYSTEM_MESSAGE = """Imagine you're a data entry especialist, especializing in transcribing pictures of bank statements into json objects. I'm going to provide with a page that summarizes the assets of the bank CITIBANK. The assets are grouped by different types of assets, but their names and components may change from year to year. The main types of assets may be "Fondos disponibles", "Inversiones", "Cartera de Creditos", "Propiedades", "Otros" and "TOTAL ACTIVOS", which is the total of assets in the bank. These sections may appear under different names, other sections may appear as well. All sections that appear in the bank statement should be transcribed to the object. Their values are provided in two columns: the assets of the corresponding year (year_1) and the assets of the previous year (year_2).

You will provide a json object, that provides the values for both years of each of the components of each section. For example:
{
"years": {
    "year_1": 2023,
    "year_2": 2022
  }
"Fondos disponibles":{
    "Caja": {
      "2023": 214,109,883,
      "2022": 210,377,002 
    }
    ...
  }
  ...
}
"""

# Function to encode the image
def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def convert_image(file_path: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    base64_image = encode_image(file_path)

    payload = {
        "model": "gpt-4o",
        "response_format": {
            "type": "json_object"
        },
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe the following bank statement following the provided instructions carefully, step by step. Take your time to return a very precise transcription."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res_obj = response.json()

    assets_json = res_obj['choices'][0]['message']['content']

    return assets_json

if __name__=="__main__":
    redo_images = [Path('..') / "images" / "valores_2022_page-0001.jpg"]

    for img in redo_images:
        response = convert_image(img)
        directory = Path("../json_statements")
        img_name = img.name.replace(".jpg", "")
        cur_path = directory / f"{img_name}.json"
        with open(cur_path, "w", encoding="utf8") as json_file:
            json.dump(response, json_file, indent=4)
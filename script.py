import json
from pathlib import Path


def get_practices():
    path = Path("./resources/android-hacking.json")
    text = path.read_text()
    response = json.loads(text)
    entries = response["results"]
    results = []
    for entry in entries:
        if entry["_class"] == "practice":
            results.append(entry)
    str_res = json.dumps(results)
    print(str_res)


get_practices()

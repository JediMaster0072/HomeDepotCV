import base64
import json

import requests


def run_detection() -> dict:
    image_path = "test_image.jpg"
    url = "http://localhost:9000/predictions/yolov7"

    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "instances": [
            {
                "model_name": "detection",
                "file": b64_image,
            }
        ]
    }

    print(f"Sending request to {url} ...")
    response = requests.post(url, json=payload)

    print(f"Status : {response.status_code}")
    result = response.json()
    print(f"Response:\n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    run_detection()

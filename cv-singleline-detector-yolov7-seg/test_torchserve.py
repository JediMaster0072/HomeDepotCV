import cv2
import json
import base64
import requests

URL = "http://127.0.0.1:9000/predictions/yolov7"

IMAGE_PATHS = [
    "strip_0.jpg",
    # "strip2.png"
]

instances = []

for idx, image_path in enumerate(IMAGE_PATHS):
    image = cv2.imread(image_path)

    _, buffer = cv2.imencode(".png", image)

    image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

    instances.append({"model_name": "segmentation", "strip_id": idx, "file": image_b64})

payload = {"instances": instances}

response = requests.post(URL, json=payload)

print("Status Code:", response.status_code)

try:
    result = response.json()
    print(json.dumps(result, indent=2)[:10000])

except Exception:
    print(response.text)

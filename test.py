import base64
import requests
import ujson
import numpy as np

with open("1.jpg", "rb") as f:
    img_str = f.read()

b64_image = base64.urlsafe_b64encode(img_str).decode('utf-8')

headers={'Content-Type': 'application/json'}
payload = ujson.dumps({"instances": [b64_image]})

resp = requests.post("http://127.0.0.1:8501/v1/models/faceboxes:predict", data=payload, headers=headers)

print(f"Response time: {resp.elapsed.microseconds / 1000}ms")

rtn = resp.json()

result = rtn["predictions"][0]

num_boxes = np.array(result["num_boxes"])
boxes = np.array(result["boxes"])
scores = np.array(result["scores"])

print(scores[:10])

#array([0.322722, 0.26222 , 0.711708, 0.752838])  0.31
#array([0.284169, 0.261221, 0.699398, 0.737265])  0.74

import pdb; pdb.set_trace()

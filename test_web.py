import requests
import base64

base_url = 'http://localhost:5000'
predict_url = base_url + '/predict'

test_path1 = "peoples.jpg"
img1 = base64.b64encode(open(test_path1, 'rb').read()).decode()

print("test {} use base64".format(predict_url))
print(requests.post(predict_url, json={"img_b64": img1, 'img_id': '123456'}).json())

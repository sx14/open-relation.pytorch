import requests, json

url = 'http://localhost:5000/predict-rela'
files = {'image': open('../hier_det/images/wolf.jpg', 'rb')}
r = requests.post(url, files=files)
print(r)
print(r.text)
print(r.content)




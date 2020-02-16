import requests, json

url = 'http://localhost:5000/predict'
files = {'image': open('../hier_det/images/4874188_684cc9482e_o.jpg', 'rb')}
r = requests.post(url, files=files)
print(r)
print(r.text)
print(r.content)




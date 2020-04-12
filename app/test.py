import requests, json
mode = '-rela'
url = 'http://localhost:5000/predict' + mode
files = {'image': open('../hier_det/images/zebra.jpeg', 'rb')}
r = requests.post(url, files=files)
print(r)
print(r.text)
print(r.content)
res = r.json()['result']
if mode == '-rela':
    for i in range(len(res["dets"])):
        print(res["sbj_labels"][i] + ' ' + res['pr_labels'][i] + ' ' + res['obj_labels'][i])




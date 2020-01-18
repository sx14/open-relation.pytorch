# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread, Lock
import requests
import time

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = 'http://localhost:5000/predict-rela'
files = {'image': open('../hier_det/images/4874188_684cc9482e_o.jpg', 'rb')}

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 100
SLEEP_COUNT = 0.05

total_time = 0
lock = Lock()


def call_predict_endpoint(n):
    global total_time
    files = {'image': open('../hier_det/images/4874188_684cc9482e_o.jpg', 'rb')}
    # submit the request
    begin_time = time.time()
    r = requests.post(KERAS_REST_API_URL, files=files).json()

    # ensure the request was sucessful
    if r["success"]:
        print("[INFO] thread {} OK".format(n))

    # otherwise, the request failed
    else:
        print("[INFO] thread {} FAILED".format(n))
    end_time = time.time()
    lock.acquire()
    total_time += (end_time - begin_time)
    lock.release()


thread_list = []
# loop over the number of threads
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    thread_list.append(t)
    t.start()
    time.sleep(SLEEP_COUNT)

for t in thread_list:
    t.join()

print("Avg: %d, Total: %d" % (total_time / NUM_REQUESTS, total_time))

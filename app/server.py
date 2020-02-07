# import the necessary packages
import json
import time

import numpy as np
import redis
import settings
import json
import helpers
from hier_rela.demo import load_model, infer as infer_rela
from hier_det.infer import load_hier_model, load_faster_model, infer as infer_det
from hier_rela.eval.temp_dsr import show as show_rela
from show_det import show as show_det
import threading
import time
model = None
class_name = None
transforms = None
score_threshold = 0
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


class det_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name='det_thread')

    def run(self):
        while True:
            queue = db.lrange(settings.DET_QUEUE, 0,
                              settings.BATCH_SIZE - 1)
            imageIDs = []
            batch = {}

            for q in queue:
                q = json.loads(q.decode("utf-8"))
                image = helpers.base64_decode_image(q["image"])
                image = np.array(image)
                batch[q["id"]] = image
                imageIDs.append(q["id"])
            if len(imageIDs) > 0:
                det_roidb = infer_det(batch)
                print(det_roidb)
                res = show_det(det_roidb)
                for i, imageID in enumerate(imageIDs):
                    db.set(imageID, json.dumps(res[imageID]))
                db.ltrim(settings.DET_QUEUE, len(imageIDs), -1)
            time.sleep(settings.SERVER_SLEEP)


class rela_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name='rela_thread')

    def run(self):
        while True:
            queue = db.lrange(settings.RELA_QUEUE, 0,
                              settings.BATCH_SIZE - 1)
            imageIDs = []
            batch = {}

            for q in queue:
                q = json.loads(q.decode("utf-8"))
                image = helpers.base64_decode_image(q["image"])
                image = np.array(image)
                batch[q["id"]] = image
                imageIDs.append(q["id"])
            if len(imageIDs) > 0:
                pred_roidb = infer_rela(batch)
                res = show_rela(pred_roidb)
                print(res)
                for i, imageID in enumerate(imageIDs):
                    db.set(imageID, json.dumps(res[i]))
                db.ltrim(settings.RELA_QUEUE, len(imageIDs), -1)
            time.sleep(settings.SERVER_SLEEP)


load_hier_model()
load_faster_model()
load_model()
threads = []
threads.append(det_thread())
threads.append(rela_thread())

for t in threads:
    t.start()
for t in threads:
    t.join()
print('END')
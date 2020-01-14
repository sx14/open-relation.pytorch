# import the necessary packages
import json
import time

import numpy as np
import redis

from hier_det.demo_hier2 import load_faster_model, load_hier_model, infer
import settings
import json
import helpers
from show_det import show

model = None
class_name = None
transforms = None
score_threshold = 0
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def classify_process():
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
            det_roidb = infer(batch)
            print(det_roidb)
            res = show(det_roidb)
            for i, imageID in enumerate(imageIDs):
                db.set(imageID, json.dumps(res[imageID]))
            db.ltrim(settings.DET_QUEUE, len(imageIDs), -1)
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    load_faster_model()
    load_hier_model()
    classify_process()


# import the necessary packages
import json
import time

import numpy as np
import redis
import settings
import json
import helpers
from hier_rela.infer import load_model, infer
from hier_det.infer import load_hier_model, load_faster_model
from show_rela_det import show

model = None
class_name = None
transforms = None
score_threshold = 0
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def classify_process():
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
            pred_roidb = infer(batch)
            res = show(pred_roidb)
            print(res)
            for i, imageID in enumerate(imageIDs):
                db.set(imageID, json.dumps(res[i]))
            db.ltrim(settings.RELA_QUEUE, len(imageIDs), -1)
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    load_hier_model()
    load_faster_model()
    load_model()
    classify_process()

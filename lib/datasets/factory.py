# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.coco import coco
from lib.datasets.imagenet import imagenet
from lib.datasets.vg200.vg import vg
from lib.datasets.vg200.vg_rela import vg_rela
from lib.datasets.vglsj.vg import vg_lsj
from lib.datasets.vglsj.vg_rela import vg_lsj_rela
from lib.datasets.vrd.vrd import vrd
from lib.datasets.vrd.vrd_rela import vrd_rela
from lib.datasets.vrd_voc import vrd_voc
from lib.datasets.vg_voc import vg_voc

import numpy as np

# Set up vrd_<year>_<split>
for year in ['2007', '2016']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'vrd_{}_{}'.format(year, split)
    if year == '2007':
        __sets[name] = (lambda split=split, year=year: vrd(split, year))
    else:
        __sets[name] = (lambda split=split, year='2007': vrd_rela(split, year))

# Set up vg_<year>_<split>
for year in ['2007', '2016']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'vg_{}_{}'.format(year, split)
    if year == '2007':
        __sets[name] = (lambda split=split, year=year: vg(split, year))
    else:
        __sets[name] = (lambda split=split, year='2007': vg_rela(split, year))

# Set up vrd_voc_<year>_<split>
for year in ['2007']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'vrd_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: vrd_voc(split, year))

# Set up vg_voc_<year>_<split>
for year in ['2007']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'vg_voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: vg_voc(split, year))

for year in ['2007']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'vg_lsj_{}_{}'.format(year, split)
    if year == '2007':
        __sets[name] = (lambda split=split, year=year: vg_lsj(split, year))
    else:
        __sets[name] = (lambda split=split, year='2007': vg_lsj_rela(split, year))


# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())

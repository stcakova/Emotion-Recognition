import cv2
import numpy as np
from constants import *
from os.path import join

class DatasetLoader(object):

  def __init__(self):
    pass

  def load_from_save(self):
    self._images      = np.load(join('./data/', IMAGES_FILENAME))
    self._labels      = np.load(join('./data/', LABELS_FILENAME))
    self._images_test = np.load(join('./data/', TEST_IMAGES_FILENAME))
    self._labels_test = np.load(join('./data/', TEST_LABELS_FILENAME))
    self._images      = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._images_test = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._labels      = self._labels.reshape([-1, len(EMOTIONS)])
    self._labels_test = self._labels.reshape([-1, len(EMOTIONS)])

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def images_test(self):
    return self._images_test

  @property
  def labels_test(self):
    return self._labels_test

  @property
  def num_examples(self):
    return self._num_examples
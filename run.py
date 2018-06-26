import cv2
import sys
import numpy as np

from constants import *
from draw import DrawUtils
from image_formatter import Formatter
from neural_network import EmotionRecognition

# INIT
network = EmotionRecognition()
network.build_network()

formatter = Formatter()
video_capture = cv2.VideoCapture(0)
feelings_faces = []

for index, emotion in enumerate(EMOTIONS):
  feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))
 
#FRAMES CAPTURING AND CLASSFICATION 
while True:
  ret, frame = video_capture.read()

  # Predict result with network
  result = network.predict(formatter.format(frame))
  if result is not None:
    for index, emotion in enumerate(EMOTIONS):
      draw_utils = DrawUtils(frame, index)
      draw_utils.draw_diagram(emotion)
      draw_utils.draw_text(result);

    # Add emotion image
    face_image = feelings_faces[result[0].tolist().index(max(result[0]))]

    for c in range(0, 3):
      frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)


  # Display the resulting frame
  cv2.imshow('Video', frame)

  # Quit on 'q' button press
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()
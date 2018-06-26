import cv2
from constants import *
from faces import Face

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
eye_cascade_classifier = cv2.CascadeClassifier(EYE_CASC_PATH)
sleep = cv2.imread('./emojis/sleep.png', -1)

def show_alert(frame):
  for c in range(0, 3):
    frame[200:320, 10:130, c] = sleep[:,:,c] * (sleep[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - sleep[:, :, 3] / 255.0)

class Formatter:
  def __init__(self):
     pass
  
  def format(self, image):
    frame = image
    if len(image.shape) > 2 and image.shape[2] == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor = 1.3,
        minNeighbors = 5
    )

    eyes_closed = eye_cascade_classifier.detectMultiScale(
        image,
        scaleFactor = 1.3,
        minNeighbors = 5
    )

    if not len(faces) > 0:
      return None

    #eyes of the person are closed
    if not len(eyes_closed) > 0:
      show_alert(frame)
      return
    
    max_area_face = Face(faces[0])
    for face in faces:
      face_area = Face(face).area()
      if face_area > max_area_face.area():
        max_area_face = face_area
    face = max_area_face
    image = image[face.get_y():(face.get_y() + face.get_height()), face.get_x():(face.get_x() + face.get_width())]
    try:
      image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
      print("Error: Resizing image failed")
      return None
    return image
import joblib
import cv2 
from  configparser import ConfigParser
from face_features_extraction import ExifOrientationNormalize
from PIL import Image
from _utils import draw_bb_on_img
import numpy as np

config = ConfigParser()
config.read('config.ini')

model_path = eval(config['INFERENCE']['model_path'])

cap = cv2.VideoCapture(0)
model = joblib.load(model_path)

orientation = ExifOrientationNormalize()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    img = Image.fromarray(frame)
    img = orientation(img)
    faces = model(img)

    if faces is not None:
        draw_bb_on_img(faces, img)
    
    cv2.imshow('video', np.array(img))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
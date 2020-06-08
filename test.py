#from keras.models import model_from_json
import cv2
import tensorflow as tf
import numpy as np


face_detection = cv2.CascadeClassifier('C:/Users/Ahmed.FATNASSI/Desktop/emotion/haar_cascade_face_detection.xml')
labels = ["Neutral", "Happy", "Sad", "Surprised", "Angry"]
model = tf.keras.models.load_model('C:/Users/Ahmed.FATNASSI/Desktop/emotion/expression.model',
    custom_objects=None,
   compile=False)


settings = {
    'scaleFactor': 1.1,
    'minNeighbors': 5,
    'minSize': (50, 50)
}
#C:/Users/Ahmed.FATNASSI/Pictures/anglais.png
img = cv2.imread("C:/Users/Ahmed.FATNASSI/Pictures/87385697_243921683266629_8356886552667750400_n.jpg")
window_name = 'image'
print("size : ",img.shape)

# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, img)

cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
facedetected = face_detection.detectMultiScale(gray, **settings)
print(type(facedetected))
if type(facedetected) == tuple:
    print("no face detected")
else:
    print(facedetected)
    test = facedetected[0]
    xmin = int(test[0])
    xmax = int(test[0] + test[2])
    ymin = int(test[1])
    ymax = int(test[1] + test[3])
    facedet = gray[ymin:ymax, xmin:xmax]
    cv2.imshow('face detected', facedet)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

    img = cv2.resize(facedet, (48, 48))
    img = img / 255.0

    modelprection = model.predict(np.array([img.reshape((48, 48, 1))])).argmax()
    modelpredection = model.predict(np.array([img.reshape((48, 48, 1))]))
    print("model predection value : ",modelpredection)
    emotion = labels[modelprection]
    print("emotion detected : ", emotion)


from flask import Flask, request, jsonify, render_template
import cv2
import tensorflow as tf
import numpy as np
import json

face_detection = cv2.CascadeClassifier('C:/Users/Ahmed.FATNASSI/Desktop/emotion/haar_cascade_face_detection.xml')
labels = ["Neutral", "Happy", "Sad", "Surprised", "Angry"]



settings = {
    'scaleFactor': 1.1,
    'minNeighbors': 5,
    'minSize': (50, 50)
}


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    image = request.form.get("image_path")
    model = tf.keras.models.load_model('C:/Users/Ahmed.FATNASSI/Desktop/emotion/expression.model',
                                       custom_objects=None,
                                       compile=False)
    #model.summary()
    #print("image :",image)

    img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facedetected = face_detection.detectMultiScale(gray, **settings)
    #print(facedetected)
    if type(facedetected) == tuple:
        #print("no face detected")
        return json.dumps({'expression': "no face detected"}), 200, {'ContentType': 'application/json'}
    else:
        
        test = facedetected[0]
        xmin = int(test[0])
        xmax = int(test[0] + test[2])
        ymin = int(test[1])
        ymax = int(test[1] + test[3])
        facedet = gray[ymin:ymax, xmin:xmax]
        #print("xmax : ",xmax)
        #print("xmin : ", xmin)
        #print("ymax : ", ymax)
        #print("ymin : ", ymin)

        img = cv2.resize(facedet, (48, 48))
        img = img / 255.0
        #print("img:",img)



        modelprection = model.predict(np.array([img.reshape((48, 48, 1))])).argmax()
        emotion = labels[modelprection]
        #print("emotion detected : ", emotion)

        emotion_detected = emotion

        return json.dumps({'expression': emotion_detected}), 200, {'ContentType': 'application/json'}



if __name__ == "__main__":
    app.run(debug=True, port=9095)
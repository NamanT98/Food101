from flask import Flask,render_template,request
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras
import numpy as np
import json

# model=pickle.load(open('model.pkl','rb'))
model=tf.keras.models.load_model('model.h5')
f=open('labels.json',"rb")
labels=dict(json.load(f))

app=Flask(__name__)

def preprocess_img(img):
    img=tf.image.resize(img,[224,224])
    img=tf.cast(img,tf.float16)
    img=img/255.
    return np.expand_dims(img,axis=0)

def predict(file,model=model,labels=labels):
    img=plt.imread(file)
    img=preprocess_img(img)
    pred=model.predict(img)
    print(pred.argmax())
    prediction=labels[f"{pred.argmax()}"]
    return prediction

@app.route("/")
def home():
    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def get_file():
    file=request.files['imagefile']
    img_path="static/"+file.name+".jpeg"
    file.save(img_path)

    prediction=predict(img_path)
    print(prediction)
    return render_template('index.html',prediction=prediction,img=img_path)

if __name__=="__main__":
    app.run(debug=True)
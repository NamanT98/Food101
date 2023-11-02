# import tensorflow as tf
# model=tf.keras.models.load_model('model.h5')
import json
f=open('labels.json','rb')
labels=json.load(f)
# print(labels)
print(dict(labels))
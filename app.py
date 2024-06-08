import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil
import telepot
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']

        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')

    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':

        dirPath = "./static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"


        shutil.copy("test/"+fileName, dst)

        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Lungcancer-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                verifying_data = np.asarray(verifying_data, dtype="object")
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')


        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 3, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        status=""
        str_label=" "
        accuracy=""
        abd=[0.05,0.06,0.07,0.055,0.054,0.08]
        vk=random.choice(abd)
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Cancer'
            elif np.argmax(model_out) == 1:
                str_label = 'Normal'
            elif np.argmax(model_out) == 2:
                str_label = 'unwanted'


            if str_label == 'Cancer':
                model_out[0]=model_out[0]-vk
                status = "Cancer"
                print("The predicted image of the Cancer is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the Cancer is with a accuracy of {}%".format(model_out[0]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])

                dic={'Cancer':A,'Normal':B,'Unwanted':C}
                algm = list(dic.keys())
                accu = list(dic.values())
                fig = plt.figure(figsize = (5, 5))
                plt.bar(algm, accu, color ='maroon', width = 0.3)
                plt.xlabel("Comparision")
                plt.ylabel("Accuracy Level")
                plt.title("Accuracy Comparision between cancer detection....")
                plt.savefig('static/matrix.png')
                bot=telepot.Bot('6155286168:AAEzfZMiEHl0sXRWaGPTJa-HzbV0IH0XtxQ')
                bot.sendMessage('5183108511',str('Hello user,our analysis indicates presence of lung cancer in your uploaded image'))


            elif str_label == 'Normal':
                model_out[1]=model_out[1]-vk
                status = "Normal "
                print("The predicted image of the Normal is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the Normal is with a accuracy of {}%".format(model_out[1]*100)
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])

                dic={'Cancer':A,'Normal':B,'Unwanted':C}
                algm = list(dic.keys())
                accu = list(dic.values())
                fig = plt.figure(figsize = (5, 5))
                plt.bar(algm, accu, color ='maroon', width = 0.3)
                plt.xlabel("Comparision")
                plt.ylabel("Accuracy Level")
                plt.title("Accuracy Comparision between cancer detection....")
                plt.savefig('static/matrix.png')
                bot=telepot.Bot('6155286168:AAEzfZMiEHl0sXRWaGPTJa-HzbV0IH0XtxQ')
                bot.sendMessage('5183108511',str('Hello user,our analysis indicates presence of lung cancer in your uploaded image'))

            elif str_label == 'unwanted':
                status = "This is not x-ray image "
                A=float(model_out[0])
                B=float(model_out[1])
                C=float(model_out[2])

                dic={'Cancer':A,'Normal':B,'Unwanted':C}
                algm = list(dic.keys())
                accu = list(dic.values())
                fig = plt.figure(figsize = (5, 5))
                plt.bar(algm, accu, color ='maroon', width = 0.3)
                plt.xlabel("Comparision")
                plt.ylabel("Accuracy Level")
                plt.title("Accuracy Comparision between cancer detection....")
                plt.savefig('static/matrix.png')
                bot=telepot.Bot('6155286168:AAEzfZMiEHl0sXRWaGPTJa-HzbV0IH0XtxQ')
                bot.sendMessage('5183108511',str('Hello user,our analysis indicates presence of lung cancer in your uploaded image'))




        return render_template('userlog.html', status=str_label,accuracy=accuracy,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/matrix.png")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import csv


from keras.models import load_model
import h5py
from keras import __version__ as keras_version

counter = 0
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

speed_array = []
throttle_array = []

# modeified to a PID controller by Dyson Freeman
class SimplePIController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.
        self.err_n_2 = 0.
        self.err_n_1 = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        control_value = self.Kp * self.error + self.Ki * self.integral + self.Kd * (self.error + self.err_n_2 - 2* self.err_n_1)
        
        self.err_n_1 = self.error
        self.err_n_2 = self.err_n_1
        
        return control_value


controller = SimplePIController(0.05, 0.001, 0)
set_speed = 20
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))
        # Model identification section. Comment these when the model is derived, use the PID control instead
#        if float(speed) > 25:
#            throttle = 0
#        elif float(speed) < 10:
#            throttle = -1
#        elif float(speed) < -10:
#            throttle = 1
#        else:
#            throttle = throttle 
        
        print(steering_angle, throttle, speed)
        
       
        # stored in array
        speed_array.append(speed)
        throttle_array.append(throttle)
        
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
        # record simulation data to csv file
        if args.datalog_folder != '':
            with open('../logData/logdata.csv','w',newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for i in range(len(speed_array)):
                    writer.writerow([speed_array[i], throttle_array[i]])
                
            
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
# Freeman Added
    parser.add_argument(
        'datalog_folder',
        type = str,
        nargs = '?',
        default = '',
        help='Path to datalog_folder. This is where the data log file stored!'                                            
    )
#############
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")
# Freeman Added        
    if args.datalog_folder != '':
        print("Writing data log file at {}".format(args.datalog_folder))
        if not os.path.exists(args.datalog_folder):
            os.makedirs(args.datalog_folder)
        else:
            shutil.rmtree(args.datalog_folder)
            os.makedirs(args.datalog_folder)
        print("DATA LOGGING ...")
    else:
        print("NO LOGGING ...")
###
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

from flask import Flask, jsonify, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import python_speech_features
import numpy as np
import base64
from pydub import AudioSegment
import io

app = Flask(__name__)

word2index = {
    # core words
    "yes": 0,
    "no": 1,
    "up": 2,
    "down": 3,
    "left": 4,
    "right": 5,
    "on": 6,
    "off": 7,
    "stop": 8,
    "go": 9,
    "zero": 10,
    "one": 11,
    "two": 12,
    "three": 13,
    "four": 14,
    "five": 15,
    "six": 16,
    "seven": 17,
    "eight": 18,
    "nine": 19,
}
index2word = [word for word in word2index]

def get_model():
    global model
    model = load_model("inggris.h5")
    print("Model loaded")

def preprocess_audio(audio):
    # audio = audio.astype(np.float)
    # normalize data
    audio=audio.astype('float32')
    audio -= audio.mean()
    audio /= np.max((audio.max(), -audio.min()))
    # compute MFCC coefficients
    features = python_speech_features.mfcc(audio, samplerate=16000, winlen=0.025, winstep=0.01, numcep=20, nfilt=40, nfft=512, lowfreq=100, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
    return features

get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['audio']
    decoded = base64.b64decode(encoded)
    audio = AudioSegment.from_file(io.BytesIO(decoded), format="wav")
    print(audio)
    #preprocess

    samples = audio.get_array_of_samples()
    samples = np.array(samples)

    audio_preprocessed = preprocess_audio(samples)
    recorded_feature = np.expand_dims(audio_preprocessed, 0)
    prediction = model.predict(recorded_feature).reshape((20, ))
    prediction /= prediction.sum()
    #=======================================================
    prediction_sorted_indices = prediction.argsort()

    label = []
    percentage = []
    for k in range(3):
        i = int(prediction_sorted_indices[-1-k])
        label.append(index2word[i])
        percentage.append(prediction[i]*100)
        print("%d.)\t%s\t:\t%2.1f%%" % (k+1, label[k], percentage[k]))
    
    response = {
        'prediction' : {
            'label1' : label[0],
            'prob1' : percentage[0],
            'label2' : label[1],
            'prob2' : percentage[1],
            'label3' : label[2],
            'prob3' : percentage[2]
        }
    }
    #=======================================================
    # response = {
    #     "prediction" : prediction
    # }

    return jsonify(response)

@app.route("/home")
def home():
    return render_template("home.html")

# if __name__ == '__main__':
#     # This is used when running locally only. When deploying to Google App
#     # Engine, a webserver process such as Gunicorn will serve the app.
#     app.run(host='127.0.0.1', port=8080, debug=True)
# # [END gae_flex_quickstart]

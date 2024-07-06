from flask import Flask, render_template, request
import numpy as np
import re
import os
import tensorflow as tf
from numpy import array
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import json

IMAGE_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


def remove_time_major_from_model(model_path):
    # Load the model
    model = load_model(model_path)

    # Save the model's architecture to a JSON string
    model_json = model.to_json()

    # Parse the JSON string to modify it
    model_config = json.loads(model_json)

    # Function to recursively remove the `time_major` argument
    def remove_time_major(config):
        if isinstance(config, dict):
            if 'class_name' in config and config['class_name'] == 'LSTM':
                config['config'].pop('time_major', None)
            for key, value in config.items():
                remove_time_major(value)
        elif isinstance(config, list):
            for item in config:
                remove_time_major(item)

    remove_time_major(model_config)

    # Convert the modified configuration back to JSON
    modified_model_json = json.dumps(model_config)

    # Load the modified model from the JSON string
    modified_model = tf.keras.models.model_from_json(modified_model_json)

    # Load the original weights into the modified model
    modified_model.load_weights(model_path)

    # Save the modified model to a new file
    modified_model.save('modified_product_review_analyzer_model.h5')

    return 'modified_product_review_analyzer_model.h5'


def init():
    global model

    modified_model_path = remove_time_major_from_model(
        'product_review_analyzer_model.h5')

    # Load the modified model
    model = load_model(modified_model_path)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/sentiment_analysis_prediction', methods=['POST'])
def sentiment_analysis_prediction():
    if request.method == 'POST':
        text = request.form['text']
        x_text = text
        Sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        x_text = x_text.lower().replace("<br />", " ")
        x_text = re.sub("[^A-Za-z0-9]", " ", x_text)
        words = x_text.split()
        # x_test = [[word_to_id[word]] if (
        #     word in word_to_id and word_to_id[word] <= 5000) else 0 for word in words]
        x_test = []
        for word in words:
            if word in word_to_id and word_to_id[word] < 100000:
                x_test.append(word_to_id[word])
            else:
                x_test.append(0)
        x_test = np.array([x_test])
        x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
        
        probability = model.predict(x_test)[0][0]
        if probability > 0.5:
            Sentiment = 'Positive'
            img_filename = os.path.join(
                app.config['UPLOAD_FOLDER'], 'happy_emoji.jpg')
        else:
            Sentiment = 'Negative'
            img_filename = os.path.join(
                app.config['UPLOAD_FOLDER'], 'sad_emoji.jpg')

        return render_template('home.html', text=text, sentiment=Sentiment, probability=probability, image=img_filename)


if __name__ == "__main__":
    init()
    app.run()

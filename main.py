import tensorflow as tf
import os
import app
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

app = Flask(__name__)

# Загрузка обученной модели
if not os.path.exists("test_DS.h5"):
    print("Ошибка: Модель не найдена. Сначала обучите модель и сохраните ее.")
    exit(1)

model = tf.keras.models.load_model("test_DS.h5")
tokenizer = None
data_dir = "./aclImdb"

def load_data(subset):
    texts = []
    labels = []
    for label in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, subset, label)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

train_texts, train_labels = load_data("train")
test_texts, test_labels = load_data("test")

max_words = 10000
max_sequence_length = 300

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

@app.route('/', methods=['post', 'get'])
def index():
    messageScore = ''
    messageSentiment = ''
    comment = ''
    if request.method == 'POST':
        username = request.form.get('comment')
        comment = username
        user_data = app.preprocess_user_data(comment, tokenizer, max_sequence_length)
        prediction = app.get_prediction(model, user_data)
        messageScore = str(prediction)[2]
        if prediction >= 0.5:
            sentiment = "Положительный комментарий."
        else:
            sentiment = "Отрицательный комментарий."
        messageSentiment = sentiment
    return render_template('index.html', messageScore=messageScore,messageSentiment = messageSentiment, comment = comment)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

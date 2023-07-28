import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os
import tarfile
import urllib.request

# Загрузка данных
# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
data_dir = "./aclImdb"
# if not os.path.exists(data_dir):
#     urllib.request.urlretrieve(url, "aclImdb_v1.tar.gz")
#     with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
#         tar.extractall()
#     os.remove("aclImdb_v1.tar.gz")

# Загрузка текстовых отзывов и их меток
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

# Предобработка данных
max_words = 10000
max_sequence_length = 300

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Создание и обучение модели RNN
# embedding_dim = 100
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Embedding(max_words, embedding_dim, input_length=max_sequence_length),
#     tf.keras.layers.LSTM(128),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.summary()

# Обучение модели
# epochs = 5
# batch_size = 32
# model.fit(train_data, np.array(train_labels), epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Сохранение модели на диск
# model.save("test_DS.h5")

# Загрузка модели с диска, если она уже обучена
if os.path.exists("test_DS.h5"):
    model = tf.keras.models.load_model("test_DS.h5")

# Оценка модели на тестовых данных
# loss, accuracy = model.evaluate(test_data, np.array(test_labels))
# print(f'Test accuracy: {accuracy}')

# Ввод данных с клавиатуры для теста
def get_user_input():
    user_input = input("Введите свой комментарий: ")
    return user_input

# Предобработка данных пользователя
def preprocess_user_data(user_input, tokenizer, max_sequence_length):
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_data = pad_sequences(user_sequence, maxlen=max_sequence_length)
    return user_data

# Получение предсказания от модели
def get_prediction(model, user_data):
    prediction = model.predict(user_data)
    return prediction[0][0]

# Загрузка токенизатора
max_words = 10000
max_sequence_length = 300

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

# Предложим пользователю ввести комментарий и получим его предсказание
# user_input = get_user_input()
    
# user_data = preprocess_user_data(user_input, tokenizer, max_sequence_length)
# prediction = get_prediction(model, user_data)
    
# if prediction >= 0.5:
#   print("Положительный комментарий.")
# else:
#   print("Отрицательный комментарий.")

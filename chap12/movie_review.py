import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(x_train[0])

# 단어 ->정수 인덱스 딕셔너리
word_to_index = imdb.get_word_index()

# 처음 몇 개의 인덱스는 특수 용도로 사용된다. 
word_to_index = {k:(v+3) for k,v in word_to_index.items()}
word_to_index["<PAD>"] = 0		# 문장을 채우는 기호
word_to_index["<START>"] = 1		# 시작을 표시
word_to_index["<UNK>"] = 2  		# 알려지지 않은 토큰 
word_to_index["<UNUSED>"] = 3

index_to_word = dict([(value, key) for (key, value) in word_to_index.items()])

print(' '.join([index_to_word[index] for index in x_train[0]]))

from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

vocab_size = 10000

model = Sequential()
model.add(Embedding(vocab_size, 64,
                    input_length=100))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
          batch_size=64, epochs=20, verbose=1,
          validation_data=(x_test, y_test))

results = model.evaluate(x_test,  y_test, verbose=2)
print(results)

review = "What can I say about this movie that was already said? It is my favorite time travel sci-fi, adventure epic comedy in the 80's and I love this movie to death! When I saw this movie I was thrown out by its theme. An excellent sci-fi, adventure epic, I LOVE the 80s. It's simple the greatest time travel movie ever happened in the history of world cinema. I love this movie to death, I love, LOVE, love it!"

import re
review = re.sub("[^0-9a-zA-Z ]", "", review).lower()

review_encoding = []
# 리뷰의 각 단어 대하여 반복한다. 
for w in review.split():
		index = word_to_index.get(w, 2)	# 딕셔너리에 없으면 2 반환
		if index <= 10000:		# 단어의 개수는 10000이하
			review_encoding.append(index)
		else:
			review_encoding.append(word_to_index["UNK"])

# 2차원 리스트로 전달하여야 한다. 
test_input = pad_sequences([review_encoding], maxlen = 100) 
value = model.predict(test_input) # 예측
if(value > 0.5):
	print("긍정적인 리뷰입니다.")
else:
	print("부정적인 리뷰입니다.")



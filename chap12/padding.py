from tensorflow.keras.preprocessing.sequence import pad_sequences

X = pad_sequences([[7, 8, 9], [1, 2, 3, 4, 5], [7]], maxlen=3, padding='pre')
print(X)
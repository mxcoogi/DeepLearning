import numpy as np
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.models import Sequential

# 입력 형태: (batch_size, input_length)=(32, 3)
# 출력 형태: (None, 3, 4)
model = Sequential()
model.add(Embedding(100, 4, input_length=3))

input_array = np.random.randint(100, size=(32, 3))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array.shape)
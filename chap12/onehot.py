import numpy as np
from tensorflow.keras.utils import to_categorical

# 우리가 변환하고 싶은 텍스트
text = ["cat", "dog", "cat", "bird"]

# 단어 집합
total_pets = ["cat", "dog", "turtle", "fish", "bird"]

print("text=", text)

# 변환에 사용되는 딕셔너리를 만든다. 
mapping = {}
for x in range(len(total_pets)):
  mapping[total_pets[x]] = x	#“cat"->0, "dog"->1, ...
print(mapping)

# 단어들을 순차적인 정수 인덱스로 만든다. 
for x in range(len(text)):
  text[x] = mapping[text[x]]

print("text=", text)

# 순차적인 정수 인덱스를 원-핫 인코딩으로 만든다. 
one_hot_encode = to_categorical(text)
print("text=", one_hot_encode)
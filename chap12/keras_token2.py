from tensorflow.keras.preprocessing.text import Tokenizer

t  = Tokenizer()
text = """Deep learning is part of a broader family of machine learning methods 
	based on artificial neural networks with representation learning."""

t.fit_on_texts([text])
print("단어집합 : ", t.word_index) 

seq = t.texts_to_sequences([text])[0]
print(text,"->", seq) 
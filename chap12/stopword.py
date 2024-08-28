import nltk
nltk.download('punkt')		# ①

from nltk.tokenize import word_tokenize	# ②

text = "This is a dog."			# ③
print(word_tokenize(text))

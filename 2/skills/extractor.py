from hazm import *
posTagger = POSTagger(model = 'skills\pos_tagger.model')
words=word_tokenize("باید از رفتن به جشنواره برج میلاد تهران صرف نظر کنم")
print(posTagger.tag(tokens =words))
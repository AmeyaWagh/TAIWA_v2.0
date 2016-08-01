import numpy
import nltk
import random
from sklearn import svm
import pickle

class SVM_Classifier:

	def __init__(self):
		self.bag_of_words=list()
		self.bag_of_categories=list()
		self.classifier=svm.LinearSVC() 
		
	def _get_sentences_from_csv(self,filename):
		sentences=list()
		categories=list()
		file_p=open(filename)
		for line in file_p.readlines():
			sentence,category=line.lower().strip("\n").split(",")
			sentences.append(sentence)
			categories.append(category)
		file_p.close()
		return sentences,categories
		
	def _prepare_feature_vector(self,sentences):
#		print sentences
		sent_vect=numpy.zeros((len(sentences),len(self.bag_of_words)))
		for i in range(len(sentences)):
			sentence=sentences[i].split()
			sentence=self._remove_stopwords(sentence)
			for j in range(len(sentence)):
				if sentence[j] in self.bag_of_words:
					sent_vect[i][self.bag_of_words.index(sentence[j])]=1.0
		return sent_vect
	
	def _prepare_output_vector(self,categories):
		out_vect=numpy.zeros((len(categories),),dtype='uint8')
		for i in range(len(categories)):
			out_vect[i]=self.bag_of_categories.index(categories[i])
		return out_vect
		
	def _remove_stopwords(self,words):
		return [word for word in words if word not in nltk.corpus.stopwords.words("english")] 
		
	def prepare_from_csv(self,filename):
		sentences,categories=self._get_sentences_from_csv(filename)
		self._prepare_language_model(sentences,categories)
		return self.prepare_data(sentences,categories)
		
	def prepare_data(self,sentences,categories):	
		return self._prepare_feature_vector(sentences),self._prepare_output_vector(categories)
		
	def _prepare_language_model(self,sentences,categories):
		words=nltk.word_tokenize(" ".join(sentences))
		words=self._remove_stopwords(words)
		for word in words:
			if word not in self.bag_of_words:
				self.bag_of_words.append(word)
		for category in categories:
			if category not in self.bag_of_categories:
				self.bag_of_categories.append(category)
				
	def train(self,X,Y):
		self.classifier.fit(X,Y)
		print "Trained"
		
	def _predict(self,vector):
#		print vector
		return self.classifier.predict(vector)[0]
		
	def predict(self,sentence):
#		print sentence
		return self.bag_of_categories[self._predict(self._prepare_feature_vector([sentence]))]
		
	def store_model(self):
		pickle.dump(self.classifier,open("svm_c.model","wb"))
		
if __name__=="__main__":
	svm_c=SVM_Classifier()
	sents,outs=svm_c.prepare_from_csv("data.csv")
#	for sent,out in zip(sents,outs):
#		print sent,out
	svm_c.train(sents,outs)
	svm_c.store_model()
	query=raw_input(">>> ")
	print svm_c.predict(query)
		
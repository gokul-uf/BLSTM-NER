import tensorflow as tf
import DataImporter as data

class DataFeeder:
	def __init__(self, file="", batchSize=100):
		self.dataImporter = data.DataImporter(file)
		self.numberOfSentences = len(self.dataImporter.codedSentences)
		self.batchSize = batchSize
		if(self.batchSize > self.numberOfSentences):
			self.batchSize = 100

		self.nextIndex = 0

	def getBatch(self, labelType):

		if(self.nextIndex+self.batchSize > self.numberOfSentences):
			thisBatchSize = self.numberOfSentences-self.nextIndex
		else:
			thisBatchSize = self.batchSize

		data = {"words": self.dataImporter.codedSentences[self.nextIndex:self.nextIndex + thisBatchSize],
				"batch_size": thisBatchSize,
				"seq_lens": self.dataImporter.sentenceLengths[self.nextIndex:self.nextIndex + thisBatchSize]}

		if(labelType == "ner"):
			data["labels"] = self.dataImporter.labelsNER[self.nextIndex:self.nextIndex + thisBatchSize]
		elif (labelType == "ffd"):
			data["labels"] = self.dataImporter.labelsFFD[self.nextIndex:self.nextIndex + thisBatchSize]
		else:
			raise ValueError("Wrong label type")

		if(self.nextIndex + thisBatchSize == self.numberOfSentences):
			tf.logging.warn("Exhausted training data, looping back...")
			self.nextIndex = 0
		else:
			self.nextIndex += thisBatchSize

		return data
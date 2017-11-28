import io

class DataImporter:

	def __init__(self, file=""):
		self.wordDictionary = {}
		self.words = []

		self.nerLabelsDictionary = {}

		self.codedSentences = []
		self.sentenceLengths = []
		self.labelsNER = []
		self.labelsFFD = []

		if file:
			self.file = file
		else:
			self.file = "gecoTrainingDataAllWordsWithNewlines.3.tsv"

		self.__processData()

	def __processData(self):
		trainingData = io.open(self.file, mode="r", encoding="utf-8")
		currentSentenceCoded = []
		currentNerLabels = []
		currentFfdLabels = []

		for line in trainingData.read().splitlines():
			if not line:
				self.codedSentences.append(currentSentenceCoded)
				self.sentenceLengths.append(len(currentSentenceCoded))
				self.labelsNER.append(currentNerLabels)
				self.labelsFFD.append(currentFfdLabels)

				currentSentenceCoded = []
				currentNerLabels = []
				currentFfdLabels = []
				continue

			values = line.split("\t")

			word = values[0]
			#Maybe transform the word (capitalization, etc.)

			if word not in self.wordDictionary:
				self.wordDictionary[word] = len(self.words)
				self.words.append(word)

			currentSentenceCoded.append(self.wordDictionary[word])

			nerLabel = values[1]
			if nerLabel not in self.nerLabelsDictionary:
				self.nerLabelsDictionary[nerLabel]=len(self.nerLabelsDictionary)

			currentNerLabels.append(self.nerLabelsDictionary[nerLabel])

			ffdLabel = values[4]
			currentFfdLabels.append(int(ffdLabel)-1)
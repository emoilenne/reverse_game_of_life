import numpy as np

class TrainingModel:
    def __init__(self, size, steps=0, modelHash=0, fields=None, row=None):
        """
            Initiate training model, which stores predictions of the start grid
            and number of occurrences of this model in the testing.
        """
        self.size = size
        if fields and row:
            self.parseRow(fields, row)
        else:
            self.occurrences = 0
            self.data = np.zeros((size, size), dtype=np.int32)
            self.steps = steps
            self.modelHash = modelHash

    def addPrediction(self, window):
        """
            Add start grid values of this model to predictions.
        """
        self.occurrences += 1
        self.data += window.data

    def predict(self):
        """
            Return predictions grid, that was formed on the training data.
            Each cell represents the probability of that cell to be alive.
        """
        if self.occurrences == 0:
            return np.zeros((self.size, self.size))
        return self.data / float(self.occurrences)

    def parseRow(self, fields, row):
        try:
            data = {fields[index]: row[index] for index in range(len(fields))}
            self.modelHash = int(data['hash'])
            del(data['hash'])
            self.steps = int(data['steps'])
            del(data['steps'])
            self.occurrences = int(data['occurrences'])
            del(data['occurrences'])
        except:
            raise Exception("CSV file is not valid.")

        if len(data) != self.size ** 2:
            raise Exception("Wrong number of cells for window size #%d" % self.size)

        window = []
        try:
            for index in range(self.size ** 2):
                window.append(int(data['model.' + str(index + 1)]))
            self.data = np.array(window, dtype=np.int32).reshape(self.size, self.size)
        except Exception as e:
            raise Exception("Unable to create model #%d for #%d steps" % (self.modelHash, self.steps))

    def createRow(self):
        """
            Create row of CSV that will store data from the model.
        """
        return [self.modelHash, self.steps, self.occurrences] + list(self.data.reshape(self.size ** 2))

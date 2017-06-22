from trainingAgent import TrainingAgent
from sys import argv

MAPS = 100

agent = TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5, windowSize=7)

if (len(argv) != 2):
    print "usage: ./%s {filename.csv or 'generate'}"
    exit(1)
if argv[1] == 'generate':
    agent.generateAndTrain(MAPS)
else:
    agent.train('csv/train100.csv')
agent.saveModels('csv/models100.csv')

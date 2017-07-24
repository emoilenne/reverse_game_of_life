from trainingAgent import TrainingAgent
from sys import argv

MAPS = 100

try:
    agent = TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5)

    if len(argv) == 0 or len(argv) > 2:
        print "usage: python %s [generate]"
        exit(1)
    if len(argv) == 2 and argv[1] == 'generate':
        agent.generateAndTrain(MAPS)
    else:
        agent.train('csv/train.csv')
    agent.saveModels()
except Exception as e:
    print "An error occured: " + str(e)

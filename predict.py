from trainingAgent import TrainingAgent

try:
    agent = TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5)

    agent.loadModels()
    agent.predict('csv/test.csv', 'csv/submission.csv')
except Exception as e:
    print "An error occured: " + str(e)

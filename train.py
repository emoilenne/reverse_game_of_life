from trainingAgent import TrainingAgent

agent = TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5, windowSize=7)

agent.train('csv/train100.csv')
agent.saveModels('csv/models100.csv')

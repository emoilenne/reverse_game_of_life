from trainingAgent import TrainingAgent

agent = TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5, windowSize=4)

agent.loadModels('csv/models100.csv')
agent.predict('csv/test100.csv', 'csv/submission100.csv')

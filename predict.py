from trainingAgent import TrainingAgent

agent = TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5)

agent.loadModels()
agent.predict('csv/test.csv', 'csv/submission.csv')

from trainingAgent import TrainingAgent

if __name__ == "__main__":
    # try:
        agent = TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5)
        # agent.loadModels('csv/models100.csv')
        agent.train('csv/train.csv')
        agent.predict('csv/test.csv', 'csv/submission.csv')
        agent.saveModels('csv/models.csv')
    # except Exception as e:
    #     print(e)

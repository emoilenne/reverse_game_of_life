from trainingAgent import TrainingAgent

if __name__ == "__main__":
    # try:
        agent = TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5, windowSize=4)
        # agent.loadModels('csv/models3.csv')
        agent.train('csv/train100.csv')
        agent.saveModels('csv/models100.csv')
        agent.predict('csv/test100.csv', 'csv/submission100.csv')
    # except Exception as e:
    #     print(e)

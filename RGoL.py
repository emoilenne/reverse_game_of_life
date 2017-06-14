import model

if __name__ == "__main__":
    # try:
        agent = model.TrainingAgent(height=20, width=20, minSteps=1, maxSteps=5)
        agent.train('train.csv')
        agent.predict('test.csv', 'submission.csv')
    # except Exception as e:
    #     print(e)

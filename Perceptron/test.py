from perceptron import *


def test():


    tt_d = pd.read_csv('bank-note/test.csv', header=None)
    t_d = pd.read_csv('bank-note/train.csv', header=None)
    cols = ['var', 'skew', 'curt', 'ent', 'label']
    ft = cols[:-1]
    op = cols[-1]
    tt_d.cols = cols
    t_d.cols = cols
    t_i = t_d.iloc[:, :-1].values
    tt_i = tt_d.iloc[:, :-1].values
    t_l = t_d.iloc[:, -1].values
    tt_l = tt_d.iloc[:, -1].values


    perceptron_s = StandardPerceptron(4)
    perceptron_s.train(t_i, t_l)
    error_s = perceptron_s.evaluate(tt_i, tt_l)
    print("Standard Perceptron Test Error: " + str(error_s))
    w = []
    err = []
    for i in range(100):
        perceptron = StandardPerceptron(4)
        w.append(perceptron.train(t_i, t_l))
        err.append(perceptron.evaluate(tt_i, tt_l))

    print(np.mean(w, axis=0)), print(np.mean(err))


    perceptron_v = VotedPerceptron(4)
    perceptron_v.train(t_i, t_l)
    error_v = perceptron_v.evaluate(tt_i, tt_l)
    print("Voted Perceptron Test Error: " + str(error_v))
    w = []
    err = []
    for i in range(100):
        perceptron = VotedPerceptron(4)
        w.append(perceptron.train(t_i, t_l))
        err.append(perceptron.evaluate(tt_i, tt_l))
    print(np.mean(w, axis=0)), print(np.mean(err))


    perceptron_a = AveragePerceptron(4)
    perceptron_a.train(t_i, t_l)
    error_a = perceptron_a.evaluate(tt_i, tt_l)
    print("Average Perceptron Test Error: " + str(error_a))
    w = []
    err = []
    for i in range(100):
        perceptron = AveragePerceptron(4)
        w.append(perceptron.train(t_i, t_l))
        err.append(perceptron.evaluate(tt_i, tt_l))
    print(np.mean(w, axis=0)), print(np.mean(err))


if __name__ == "__main__":
    test()
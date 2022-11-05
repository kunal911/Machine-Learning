import numpy as np
import pandas as pd


class StandardPerceptron(object):
    def __init__(self, i, epoch_level=10, r=0.01):
        self.epoch_level = epoch_level
        self.r = r  
        self.w = np.zeros(i + 1)  

    def predy(self, inputs):
        s = np.dot(inputs, self.w[1:]) + self.w[0]
        if s > 0:
            act = 1
        else:
            act = 0
        return act

    def train(self, train_inputs, lab):
        lab = np.expand_dims(lab, axis=1)
        data = np.hstack((train_inputs, lab))
        for e in range(self.epoch_level):
            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                pd = self.predy(inputs)
                self.w[1:] += self.r * (label - pd) * inputs
                self.w[0] += self.r * (label - pd)

        return self.w

    def evaluate(self, test_inputs, lab):
        errs = []
        for inputs, label in zip(test_inputs, lab):
            pd = self.predy(inputs)
            errs.append(np.abs(label-pd))

        return sum(errs) / float(test_inputs.shape[0])


class VotedPerceptron(object):
    def __init__(self, i, epoch_level=10, r=0.01):
        self.epoch_level = epoch_level
        self.r = r   
        self.w = np.zeros(i + 1)  
        self.C = [0]

    def predy(self, inputs, w):
        s = np.dot(inputs, w[1:]) + w[0]
        if s > 0:
            act = 1
        else:
            act = 0
        return act

    def train(self, train_inputs, lab):
        w = np.zeros(train_inputs.shape[1] + 1)
        w_set = [np.zeros(train_inputs.shape[1]+1)]
        lab = np.expand_dims(lab, axis=1)
        data = np.hstack((train_inputs, lab))
        m = 0
        for e in range(self.epoch_level):

            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                pd = self.predy(inputs, w)
                err = label - pd
                if err:
                    w[1:] += self.r * (label - pd) * inputs
                    w[0] += self.r * (label - pd)
                    w_set.append(np.copy(w))

                    self.C.append(1)
                    m += 1

                else:
                    self.C[m] += 1

        self.w = w
        self.w_set = w_set

        return self.w

    def evaluate(self, test_inputs, lab):
        errs = []
        n_w = len(self.w_set)
        for inputs, label in zip(test_inputs, lab):
            pds = []
            for k in range(n_w):
                pred = self.predy(inputs, w=self.w_set[k])
                if not pred:
                    pred = -1
                pds.append(self.C[k]*pred)

            pd = np.sign(sum(pds))
            if pd == -1:
                pd = 0

            errs.append(np.abs(label-pd))
        return sum(errs) / float(test_inputs.shape[0])


class AveragePerceptron(object):
    def __init__(self, i, epoch_level=10, r=0.01):
        self.epoch_level = epoch_level
        self.r = r   
        self.w = np.zeros(i + 1)  
        self.a = np.zeros(i + 1)

    def predy(self, inputs, w):
        s = np.dot(inputs, w[1:]) + w[0]
        if s > 0:
            act = 1
        else:
            act = 0
        return act

    def train(self, train_inputs, lab):
        w = np.zeros(train_inputs.shape[1] + 1)
        w_set = [np.zeros(train_inputs.shape[1]+1)]
        lab = np.expand_dims(lab, axis=1)
        data = np.hstack((train_inputs, lab))
        m = 0
        for e in range(self.epoch_level):
            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                pd = self.predy(inputs, w)
                err = label - pd
                w[1:] += self.r * (label - pd) * inputs
                w[0] += self.r * (label - pd)
                self.a += np.copy(w)

        self.w = w
        return self.a

    def evaluate(self, test_inputs, lab):
        errs = []
        for inputs, label in zip(test_inputs, lab):
            pd = self.predy(inputs, w=self.a)
            errs.append(np.abs(label-pd))

        return sum(errs) / float(test_inputs.shape[0])
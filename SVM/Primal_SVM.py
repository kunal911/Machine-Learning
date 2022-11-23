import numpy as np
from numpy.linalg import inv

b_t = "./bank-note/train.csv"
b_tt = "./bank-note/test.csv"

with open(b_t, mode='r') as f:
    t_b = []
    for line in f:
        trms = line.strip().split(',')
        t_b.append(trms)
with open(b_tt, mode='r') as f:
    tt_b = []
    for line in f:
        trms = line.strip().split(',')
        tt_b.append(trms)


def c_t_f(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data


def c_f(data):
    l = [row[-1] for row in data]
    tp = data
    for i in range(len(data)):
        tp[i][-1] = 1.0
    for i in range(len(data)):
        tp[i].append(l[i])
    return tp


def p_l(data):
    tp = data
    for i in range(len(data)):
        tp[i][-1] = 2 * data[i][-1] - 1
    return tp


bank_train = c_t_f(t_b)
bank_test = c_t_f(tt_b)

train_data = c_f(p_l(bank_train))
test_data = c_f(p_l(bank_test))

train_len = len(train_data)
test_len = len(test_data)


def r_w_d(x, gamma_0, d):
    return gamma_0 / (1 + gamma_0 * x / d)


def r_wo_d(x, gamma_0):
    return gamma_0 / (1 + x)


def sub_gradient(curr_wt, sp, iteration, r_f, C, gamma_0, d):
    w = list(np.zeros(len(sp) - 1))
    w_0 = curr_wt[0:len(curr_wt) - 1];
    w_0.append(0)
    w_00 = w_0
    if r_f == 1:
        tp_1 = 1 - r_w_d(iteration, gamma_0, d)
        tp_2 = r_w_d(iteration, gamma_0, d)
        tp_3 = tp_2 * C * train_len * sp[-1]
        if sp[-1] * np.inner(sp[0:len(sp) - 1], curr_wt) <= 1:
            w_1 = [x * tp_1 for x in w_00]
            w_2 = [x * tp_3 for x in sp[0:len(sp) - 1]]
            w = [w_1[i] + w_2[i] for i in range(len(w_1))]
        else:
            w = [x * tp_1 for x in w_00]
    if r_f == 2:
        tp_1 = 1 - r_wo_d(iteration, gamma_0)
        tp_2 = r_wo_d(iteration, gamma_0)
        tp_3 = tp_2 * C * train_len * sp[-1]
        if sp[-1] * np.inner(sp[0:len(sp) - 1], curr_wt) <= 1:
            w_1 = [x * tp_1 for x in w_00]
            w_2 = [x * tp_3 for x in sp[0:len(sp) - 1]]
            w = [w_1[i] + w_2[i] for i in range(len(w_1))]
        else:
            w = [x * tp_1 for x in w_00]
    return w


def svm_single(w, iteration, p, train_data, C, r_f, gamma_0, d):
    l_ = [];
    for i in range(train_len):
        w = sub_gradient(w, train_data[p[i]], iteration, r_f, C, gamma_0, d)
        l_.append(l_func(w, C, train_data))
        iteration = iteration + 1
    return [w, iteration, l_]


def svm_epoch(w, T, train_data, C, r_f, gamma_0, d):
    iteration = 1
    l = []
    for i in range(T):
        p = np.random.ptation(train_len)
        [w, iteration, l_] = svm_single(w, iteration, p, train_data, C, r_f, gamma_0, d)
        l.extend(l_)
    return [w, l]


def sign_func(x):
    y = 0
    if x > 0:
        y = 1
    else:
        y = -1
    return y


def calculate_error(xx, yy):
    cnt = 0
    length = len(xx)
    for i in range(length):
        if xx[i] != yy[i]:
            cnt = cnt + 1
    return cnt / length


def pred(w, data):
    pred_seq = [];
    for i in range(len(data)):
        pred_seq.append(sign_func(np.inner(data[i][0:len(data[0]) - 1], w)))
    l = [row[-1] for row in data]
    return calculate_error(pred_seq, l)


def l_func(w, C, train_data):
    tp = [];
    for i in range(train_len):
        tp.append(max(0, 1 - train_data[i][-1] * np.inner(w, train_data[i][0:len(train_data[0]) - 1])))
    val = 0.5 * np.linalg.norm(w) ** 2 + C * sum(tp)
    return val


def svm(r_f, T, gamma_0, d):
    C_global = [x / 873 for x in [100, 500, 700]]
    for C_glo in C_global:
        wt = list(np.zeros(len(train_data[0]) - 1))
        [ww, l_val] = svm_epoch(wt, T, train_data, C_glo, r_f, gamma_0, d)
        print('LEARNED w:', ww)
        err_train = pred(ww, train_data)
        err_test = pred(ww, test_data)
        print('TRAIN ERROR:', err_train)
        print('TEST ERROR:', err_test)


r_f = 2
T = 100
gamma_0 = 2.3
d = 1
svm(r_f, T, gamma_0, d)
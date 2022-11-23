import math
import numpy as np
from scipy.optimize import minimize

B_T = "./bank-note/train.csv"
B_TT = "./bank-note/test.csv"

with open(B_T, mode='r') as f:
    t_b = []
    for line in f:
        terms = line.strip().split(',')  
        t_b.append(terms)
with open(B_TT, mode='r') as f:
    tt_b = []
    for line in f:
        terms = line.strip().split(',')
        tt_b.append(terms)


def c_t_f(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data


def constant_feature(data):
    lbl = [row[-1] for row in data]
    tmp = data;
    for i in range(len(data)):
        tmp[i][-1] = 1.0
    for i in range(len(data)):
        tmp[i].append(lbl[i])
    return tmp


def p_lbl(data):
    tmp = data;
    for i in range(len(data)):
        tmp[i][-1] = 2 * data[i][-1] - 1
    return tmp


b_t = c_t_f(t_b)
b_tt = c_t_f(tt_b)

trn_d = constant_feature(p_lbl(b_t))
tst_d = constant_feature(p_lbl(b_tt))

lgt_t = len(trn_d)
test_len = len(tst_d)
dim_s = len(trn_d[0]) - 1


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


def predict(w, data):
    pred_seq = [];
    for i in range(len(data)):
        pred_seq.append(sign_func(np.inner(data[i][0:len(data[0]) - 1], w)))
    lbl = [row[-1] for row in data]
    return calculate_error(pred_seq, lbl)


def gaussian_kernel(s_1, s_2, gamma):
    dim = len(s_1) - 1
    s_11 = s_1[0:dim]
    s_22 = s_2[0:dim]
    diff = [s_11[i] - s_22[i] for i in range(dim)]
    kernel = math.e ** (-np.linalg.norm(diff) ** 2 / gamma)
    return kernel


def calculate_matrix():
    k_hat_t = np.ndarray([lgt_t, lgt_t])
    for i in range(lgt_t):
        for j in range(lgt_t):
            k_hat_t[i, j] = (trn_d[i][-1]) * (trn_d[j][-1]) * np.inner(trn_d[i][0:dim_s],
            trn_d[j][0:dim_s])
    return k_hat_t


def objective_function(x):
    tp1 = x.dot(K_hat_)
    tp2 = tp1.dot(x)
    tp3 = -1 * sum(x)
    return 0.5 * tp2 + tp3


def constraint(x):
    return np.inner(x, np.asarray(lbl_))


def svm_dual(C):
    bd = (0, C)
    bds = tuple([bd for i in range(lgt_t)])
    x0 = np.zeros(lgt_t)
    cons = {'type': 'eq', 'fun': constraint}
    sol = minimize(objective_function, x0, method='SLSQP', bounds=bds, constraints=cons)
    return [sol.fun, sol.x]


def recover_ws(dual_x):
    lenn = len(dual_x)
    ll = []
    for i in range(lenn):
        ll.append(dual_x[i] * trn_d[i][-1] * np.asarray(trn_d[i][0: dim_s]))
    return sum(ll)


def svm_dual_main(C):
    [sol_f, sol_x] = svm_dual(C)
    w = recover_ws(sol_x)
    train_error = predict(w, trn_d)
    test_error = predict(w, tst_d)
    print('w=', w)
    print('train err=', train_error)
    print('test err=', test_error)


K_hat_ = calculate_matrix()
lbl_ = [row[-1] for row in trn_d]
CC = [100 / 873, 500 / 873, 700 / 873]
for C_ in CC:
    svm_dual_main(C_)
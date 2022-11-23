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
    for r in data:
        for j in range(len(data[0])):
            r[j] = float(r[j])
    return data


def c_f(data):
    lbl = [r[-1] for r in data]
    tp = data;
    for i in range(len(data)):
        tp[i][-1] = 1.0
    for i in range(len(data)):
        tp[i].append(lbl[i])
    return tp


def p_lbl(data):
    tp = data;
    for i in range(len(data)):
        tp[i][-1] = 2 * data[i][-1] - 1;
    return tp


bk_t = c_t_f(t_b)
bk_tt = c_t_f(tt_b)

t_d = c_f(p_lbl(bk_t))
tt_d = c_f(p_lbl(bk_tt))

lgt_t = len(t_d)  
lgt_tt = len(tt_d)
dim_s = len(t_d[0]) - 1  


def sign_func(x):
    y = 0
    if x > 0:
        y = 1
    else:
        y = -1
    return y


def err_calculation(xx, yy):
    cnt = 0
    lgt = len(xx)
    for i in range(lgt):
        if xx[i] != yy[i]:
            cnt = cnt + 1
    return cnt / lgt


def kernel_predict(d_x, data, gamma):
    true_lbl = [r[-1] for r in data]
    pred_seq = [];
    for r in data:
        ll = []
        for i in range(len(d_x)):
            ll.append(d_x[i] * t_d[i][-1] * gaussian_kernel(t_d[i][0:dim_s], r[0:dim_s], gamma))
        pred = sign_func(sum(ll))
        pred_seq.append(pred)
    return err_calculation(pred_seq, true_lbl)


def gaussian_kernel(s_1, s_2, gamma):
    s_1_ = np.asarray(s_1)
    s_2_ = np.asarray(s_2)
    return math.e ** (-np.linalg.norm(s_1_ - s_2_) ** 2 / gamma)


def c_k_m(gamma):
    K_hat_t = np.ndarray([lgt_t, lgt_t])
    for i in range(lgt_t):
        for j in range(lgt_t):
            K_hat_t[i, j] = gaussian_kernel(t_d[i][0:dim_s], t_d[j][0:dim_s], gamma)
    return K_hat_t


def objective_function(x):
    tp1 = x.dot(K_mat_)
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


def c_s_v(d_x):
    ll = []
    for i in range(len(d_x)):
        if d_x[i] != 0.0:
            ll.append(i)
    return [np.count_nonzero(d_x), set(ll)]


def svm_main(C):
    [sol_f, sol_x] = svm_dual(C)
    [cnt, gg] = c_s_v(sol_x)
    return [cnt, gg]


lbl_ = [r[-1] for r in t_d]
CC_ = [100 / 873, 500 / 873, 700 / 873]
Gamma_ = [0.1, 0.5, 1, 5, 100]
for C in CC_:
    for gamma in Gamma_:
        print('C=',C, 'gamma=', gamma)
        K_mat_ = c_k_m(gamma)
        svm_main(C)


C = 5/873
gamma = 10
K_mat_ = c_k_m(gamma)
svm_main(C)
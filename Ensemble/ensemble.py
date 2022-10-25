import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from part import AB, DS, DT, Bgg, RF
from pathlib import Path
np.random.seed(2022)

def er(predy,y):
    if np.reshape(predy,(-1)).shape != np.reshape(y,(-1)).shape:
        raise ValueError("The sample size are not equal.")
    return np.mean(np.reshape(predy,(-1,1))!=np.reshape(y,(-1,1)))




# W_1 = np.ones(len(train_bank))/len(train_bank)   
# train_bank = weight_append(train_bank, W_1)      
# [aa_pred, bb_vote, weights] = ada_boost(50, delta, train_bank)
# train_bank = weight_update_data(train_bank, weights) 
# tree = tree_build(train_bank, 1)
# [pp, qq] =label_return(train_bank, tree)


# def compare(x,y):
#     count =0
#     for i in range(len(x)):
#         if x[i] != y[i]:
#             count += 1
#     return count
# print(compare(pp,qq))

# print(wt_error(to_binary(pp), to_binary(qq), weights))


# fin_pred = fin_dec(aa_pred, bb_vote, len(train_bank), 50)
# true_label =to_binary([row[-2] for row in train_bank])  
# print(_error(true_label, fin_pred))





# BANK_TRAINING = "./datasets/bank/train.csv"
# BANK_TESTING = "./datasets/bank/test.csv"

# with open(BANK_TRAINING,mode='r') as f:
#     train_bank=[]
#     for line in f:
#         terms=line.strip().split(',') 
#         train_bank.append(terms)

# num_set={0,5,9,11,12,13,14}  

# def convert_to_float(mylist):
#     temp_list = mylist
#     for k in range(len(temp_list)):
#         for i in {0,5,9,11,12,13,14}:
#             temp_list[k][i] = float(mylist[k][i])
#     return temp_list

# train_bank = convert_to_float(train_bank)


# with open('../DT/bank/train.csv', 'r') as f:
#     for line in f:
#         terms = line.strip().split(',')
#         ag.append(terms[0])
#         b.append(terms[5])
#         .append(terms[9])
#         durations.append(terms[11])
#         camps.append(terms[12])
#         .append(terms[13])
#         p.append(terms[14])

#     ag.sort()
#     b.sort()
#     .sort()
#     durations.sort()
#     camps.sort()
#     .sort()
#     p.sort()
#     f.seek(0)
#     for line in f:
#         listToAdd = []
#         terms = line.strip().split(',')
#         numericBoolean(listToAdd, terms, 0, ag)
#         # restoreUnknown(listToAdd, terms, 1, 'blue-collar')
#         listToAdd.append(terms[1])
#         listToAdd.append(terms[2])
#         # restoreUnknown(listToAdd, terms, 3, 'secondary')
#         listToAdd.append(terms[3])
#         listToAdd.append(terms[4])
#         numericBoolean(listToAdd, terms, 5, b)
#         listToAdd.append(terms[6])
#         listToAdd.append(terms[7])
#         # restoreUnknown(listToAdd, terms, 8, 'cellular')
#         listToAdd.append(terms[8])
#         numericBoolean(listToAdd, terms, 9, )
#         listToAdd.append(terms[10])
#         numericBoolean(listToAdd, terms, 11, durations)
#         numericBoolean(listToAdd, terms, 12, camps)
#         numericBoolean(listToAdd, terms, 13, )
#         numericBoolean(listToAdd, terms, 14, p)
#         # restoreUnknown(listToAdd, terms, 15, 'failure')
#         listToAdd.append(terms[15])
#         listToAdd.append(1 if terms[16] == 'yes' else -1)
#         exampleSet4.append(listToAdd)


# testData = []
# with open('../DT/bank/test.csv', 'r') as f:
#     for line in f:
#         listToAdd = []
#         terms = line.strip().split(',')
#         numericBoolean(listToAdd, terms, 0, ag)
#         # restoreUnknown(listToAdd, terms, 1, 'blue-collar')
#         listToAdd.append(terms[1])
#         listToAdd.append(terms[2])
#         # restoreUnknown(listToAdd, terms, 3, 'secondary')
#         listToAdd.append(terms[3])
#         listToAdd.append(terms[4])
#         numericBoolean(listToAdd, terms, 5, b)
#         listToAdd.append(terms[6])
#         listToAdd.append(terms[7])
#         # restoreUnknown(listToAdd, terms, 8, 'cellular')
#         listToAdd.append(terms[8])
#         numericBoolean(listToAdd, terms, 9, )
#         listToAdd.append(terms[10])
#         numericBoolean(listToAdd, terms, 11, durations)
#         numericBoolean(listToAdd, terms, 12, camps)
#         numericBoolean(listToAdd, terms, 13, )
#         numericBoolean(listToAdd, terms, 14, prev)
#         # restoreUnknown(listToAdd, terms, 15, 'failure')
#         listToAdd.append(terms[15])
#         listToAdd.append(1 if terms[16] == 'yes' else -1)
#         testData.append(listToAdd)

# weights = [1/len(exampleSet4)]*len(exampleSet4)






dp = Path('./data/bank')
col = ['age', 'job', 'marital', 'education', 'default', 'bal', 'housing', 'loan', 'contact', 
          'd', 'month', 'duration', 'camp', 'pds', 'prev', 'poutcome', 'y']
training_path = dp/'train.csv'
test_path = dp/'test.csv'

train_data = pd.read_csv(training_path, header=None, names=col) 
test_data = pd.read_csv(test_path, header=None, names=col) 

th = train_data[["age", "bal", "d", "duration", "camp", "pds", "prev"]].median()
def bp(df):
    for col in ["age", "bal", "d", "duration", "camp", "pds", "prev"]:
        df.loc[df[col] <= th[col], col] = 0
        df.loc[df[col] > th[col], col] = 1
        df[col] = df[col].map({0: "low", 1: "high"})

    return df



# bank_attributes ={'age':['yes','no'],
#              'job':['admin.','unknown','unemployed','management',
#                     'housemaid','entrepreneur','student','blue-collar',
#                     'self-employed','retired','technician','services'],
#                     'martial':['married','divorced','single'],
#                     'education':['unknown','secondary','primary','tertiary'],
#                      'default':['yes','no'],
#                      'b':['yes','no'],
#                      'housing':['yes','no'],
#                      'loan':['yes','no'],
#                      'contact':['unknown','telephone','cellular'],
#                      '':['yes','no'],
#                      'month':['jan', 'feb', 'mar', 'apr','may','jun','jul','aug','sep','oct', 'nov', 'dec'],
#                      'duration': ['yes','no'],
#                      'camp':['yes','no'],
#                      '':['yes','no'],
#                      'p':['yes','no'],
#                      'poutcome':[ 'unknown','other','failure','success']}








train_data = bp(train_data)
test_data = bp(test_data)
print(train_data.head()) 

cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)

bt = AB(tx = x, ty = y, cl = cl, entropy_base = 16)
bt.fit()
train_error = np.zeros(500)
test_error = np.zeros(500)
tp, tsp = bt.predict(train_data.to_numpy(copy=True))
ttp, ttsp = bt.predict(test_data.to_numpy(copy=True))

for i in range(len(tp)):
    train_error[i] = er( tp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er(ttp[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of iterations')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2a1.png')
print(" 2a1.png") 
plt.show()

for i in range(len(tp)):
    train_error[i] = er( tsp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er( ttsp[i], test_data.y.to_numpy(copy=True) )
plt.plot(train_error)
plt.plot(test_error)
plt.title('stump error')
plt.ylabel('error')
plt.xlabel('stump')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2a2.png') 
print(" 2a2.png") 
plt.show()



cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
bgg_t = Bgg(tx = x, ty = y, cl = cl, m_d = 16)
bgg_t.fit()
train_error = np.zeros(500)
test_error = np.zeros(500)
tp = bgg_t.predict(train_data.to_numpy(copy=True))
ttp = bgg_t.predict(test_data.to_numpy(copy=True))

for i in range(len(tp)):
    train_error[i] = er( tp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er( ttp[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2b.png') 
print(" 2b.png") 
plt.show()


def bias(y_pred, y_true):
    """
    y_pred: (max_ite, p)
    y_true: (p,)
    """
    return np.square(np.subtract(np.mean(y_pred, axis = 0),y_true)).mean()
bias.__name__='bias^2'

def variance(y_pred, y_true):
    """
    y_pred: (max_ite, p)
    y_true: (p,)
    """
    return np.mean([np.square( np.subtract( y_hat, np.mean(y_pred, axis = 0) ) )  for y_hat in y_pred], axis = 0).mean()
variance.__name__="variance"

cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
n_sample = x.shape[0]
y_pred_s = []
y_pred_b = []
for i in range(100):
    print("repeat: ", i+1)
    index = np.random.choice(np.arange(n_sample), size=1000, replace=False)
    bgg_t = Bgg(tx = x[index,:], ty = y[index], cl = cl, m_d = 16)
    bgg_t.fit()
    single_tree = bgg_t.tree[0]
    tempy = single_tree.predict(test_data.to_numpy(copy=True))
    y_pred_s.append(tempy)

    tempy = bgg_t.predict(test_data.to_numpy(copy=True))
    y_pred_b.append(tempy[499])

def helper(x):
    temp = np.ones(x.shape)
    temp[x=='no']=0
    return temp

y_pred_s_a = helper(np.array(y_pred_s))
y_pred_b_a = helper(np.array(y_pred_b))
y_true = helper(test_data.y.to_numpy(copy=True))

bias_s = bias(y_pred_s_a, y_true)
bias_b = bias(y_pred_b_a, y_true)
variance_s = variance(y_pred_s_a, y_true)
variance_b = variance(y_pred_b_a, y_true)
print("bias^2 (single, bgg): ", bias_s, bias_b)
print("variance (single, bgg): ", variance_s, variance_b)
print("total (single, bgg): ", bias_s + variance_s, bias_b + variance_b)




np.random.seed(2022)
cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
tree_RF = RF(tx = x, ty = y, cl = cl, m_d = 16, select_features = 2)
tree_RF.fit()

train_error = np.zeros(500)
test_error = np.zeros(500)
tp = tree_RF.predict(train_data.to_numpy(copy=True))
ttp = tree_RF.predict(test_data.to_numpy(copy=True))

for i in range(len(tp)):
    train_error[i] = er( tp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er( ttp[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('RF error (features = 2)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d1.png') 
print(" 2d1.png") 
plt.show()

print("no.offeatures: 4")
cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
tree_RF = RF(tx = x, ty = y, cl = cl, m_d = 16, select_features = 4)
tree_RF.fit()

train_error = np.zeros(500)
test_error = np.zeros(500)
tp = tree_RF.predict(train_data.to_numpy(copy=True))
ttp = tree_RF.predict(test_data.to_numpy(copy=True))

for i in range(len(tp)):
    train_error[i] = er( tp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er( ttp[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('RF error (features = 4)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d2.png') 
print(" 2d2.png") 
plt.show()


cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
tree_RF = RF(tx = x, ty = y, cl = cl, m_d = 16, select_features = 6)
tree_RF.fit()

train_error = np.zeros(500)
test_error = np.zeros(500)
tp = tree_RF.predict(train_data.to_numpy(copy=True))
ttp = tree_RF.predict(test_data.to_numpy(copy=True))

for i in range(len(tp)):
    train_error[i] = er( tp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er( ttp[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('RF error (features = 6)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d3.png') 
print(" 2d3.png") 
plt.show()

def bias(y_pred, y_true):
    """
    y_pred: (max_ite, p)
    y_true: (p,)
    """
    return np.square(np.subtract(np.mean(y_pred, axis = 0),y_true)).mean()
bias.__name__='bias^2'

def variance(y_pred, y_true):
    """
    y_pred: (max_ite, p)
    y_true: (p,)
    """
    return np.mean([np.square( np.subtract( y_hat, np.mean(y_pred, axis = 0) ) )  for y_hat in y_pred], axis = 0).mean()
variance.__name__="variance"

cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
n_sample = x.shape[0]
y_pred_s = []
y_pred_b = []
for i in range(100):
    print("repeat: ", i+1)
    index = np.random.choice(np.arange(n_sample), size=1000, replace=False)
    bgg_t = RF(tx = x[index,:], ty = y[index], cl = cl, m_d = 16, select_features = 6)
    bgg_t.fit()
    single_tree = bgg_t.tree[0]
    tempy = single_tree.predict(test_data.to_numpy(copy=True))
    y_pred_s.append(tempy)

    tempy = bgg_t.predict(test_data.to_numpy(copy=True))
    y_pred_b.append(tempy[499])

def helper(x):
    temp = np.ones(x.shape)
    temp[x=='no']=0
    return temp

y_pred_s_a = helper(np.array(y_pred_s))
y_pred_b_a = helper(np.array(y_pred_b))
y_true = helper(test_data.y.to_numpy(copy=True))

bias_s = bias(y_pred_s_a, y_true)
bias_b = bias(y_pred_b_a, y_true)
variance_s = variance(y_pred_s_a, y_true)
variance_b = variance(y_pred_b_a, y_true)
print("bias^2 (single, bgg): ", bias_s, bias_b)
print("variance (single, bgg): ", variance_s, variance_b)
print("total (single, bgg): ", bias_s + variance_s, bias_b + variance_b)

print("\n\n-------------3: credit card data")
dp = Path('./data/credit_card/data.csv')
col = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'P_0', 'P_2', 'P_3', 'P_4', 
          'P_5', 'P_6', 'B_A1', 'B_A2', 'B_A3', 'B_A4', 'B_A5', 
          'B_A6', 'P_A1', 'P_A2', 'P_A3', 'P_A4', 'P_A5', 'P_A6', 'y'] 
          																							

data = pd.read_csv(dp, header=None, names=col) 
print("Original data")
print(data.head())

cts_f = ["LIMIT_BAL", "AGE", 'B_A1', 'B_A2', 'B_A3', 'B_A4', 'B_A5', 
          'B_A6', 'P_A1', 'P_A2', 'P_A3', 'P_A4', 'P_A5', 'P_A6']
th = data[cts_f].median()
def data_process(df):
    for col in cts_f:
        df.loc[df[col] <= th[col], col] = 0
        df.loc[df[col] > th[col], col] = 1
        df[col] = df[col].map({0: "low", 1: "high"})

    return df
data = data_process(data)
s = np.random.choice(np.arange(30000), size = 30000, replace=False)
data = data.iloc[s]


print(data.head())
train_data = data.iloc[:24000,:]
test_data = data.iloc[24000:,:]
print(train_data.head())


cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
bt = AB(tx = x, ty = y, cl = cl, entropy_base = 16)
bt.fit()
train_error = np.zeros(500)
test_error = np.zeros(500)
tp, _ = bt.predict(train_data.to_numpy(copy=True))
ttp, _ = bt.predict(test_data.to_numpy(copy=True))

for i in range(len(tp)):
    train_error[i] = er( tp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er( ttp[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of iterations')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('3ADA.png') 
print("3ADA.png") 
plt.show()


cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
bgg_t = Bgg(tx = x, ty = y, cl = cl, m_d = 16)
bgg_t.fit()
train_error = np.zeros(500)
test_error = np.zeros(500)
tp = bgg_t.predict(train_data.to_numpy(copy=True))
ttp = bgg_t.predict(test_data.to_numpy(copy=True))

for i in range(len(tp)):
    train_error[i] = er( tp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er( ttp[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('Bgg.png') 
print(" Bgg.png") 
plt.show()

print("no.offeatures: 6")
cl = train_data.cls.to_numpy(copy=True)[:-1]
x = train_data[cl].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
tree_RF = RF(tx = x, ty = y, cl = cl, m_d = 23, select_features = 6)
tree_RF.fit()

train_error = np.zeros(500)
test_error = np.zeros(500)
tp = tree_RF.predict(train_data.to_numpy(copy=True))
ttp = tree_RF.predict(test_data.to_numpy(copy=True))

for i in range(len(tp)):
    train_error[i] = er( tp[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = er( ttp[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('RF error (features = 6)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('3RF.png') 
print(" 3RF.png") 
plt.show()



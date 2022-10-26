import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from main import AdaBoost, DecisionStump, DT, Bgg, RF
from pathlib import Path
np.random.seed(2022)

def e_r(prediction,y):
    if np.reshape(prediction,(-1)).shape != np.reshape(y,(-1)).shape:
        raise ValueError("The sample size are not equal.")
    return np.mean(np.reshape(prediction,(-1,1))!=np.reshape(y,(-1,1)))


data_path = Path('./data/bank')
colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 
          'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
training_path = data_path/'train.csv'
test_path = data_path/'test.csv'

t_d = pd.read_csv(training_path, header=None, names=colnames) 
tt_d = pd.read_csv(test_path, header=None, names=colnames) 


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


thresholds = t_d[["age", "balance", "day", "duration", "campaign", "pdays", "previous"]].median()
def bank_preprocessing(df):
   
    for col in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        df.loc[df[col] <= thresholds[col], col] = 0
        df.loc[df[col] > thresholds[col], col] = 1
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







t_d = bank_preprocessing(t_d)
tt_d = bank_preprocessing(tt_d)
print(t_d.head()) 

column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)

Boost_tree = AdaBoost(tx = x, ty = y, column = column, entropy_base = 16)
Boost_tree.fit()
t_e = np.zeros(500)
tt_e = np.zeros(500)
train_prediction, train_stump_prediction = Boost_tree.predict(t_d.to_numpy(copy=True))
test_prediction, test_stump_prediction = Boost_tree.predict(tt_d.to_numpy(copy=True))

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_prediction[i], tt_d.y.to_numpy(copy=True) )

plt.plot(t_e)
plt.plot(tt_e)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of iterations')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2a1.png')
print("image save 2a1.png") 
plt.show()

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_stump_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_stump_prediction[i], tt_d.y.to_numpy(copy=True) )
plt.plot(t_e)
plt.plot(tt_e)
plt.title('stump error')
plt.ylabel('error')
plt.xlabel('stump')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2a2.png') 
print("image save 2a2.png") 
plt.show()



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

column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
Bgg_tree = Bgg(tx = x, ty = y, column = column, m_d = 16)
Bgg_tree.fit()
t_e = np.zeros(500)
tt_e = np.zeros(500)
train_prediction = Bgg_tree.predict(t_d.to_numpy(copy=True))
test_prediction = Bgg_tree.predict(tt_d.to_numpy(copy=True))

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_prediction[i], tt_d.y.to_numpy(copy=True) )

plt.plot(t_e)
plt.plot(tt_e)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2b.png') 
print("image save 2b.png") 
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

column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
n_sample = x.shape[0]
y_pred_s = []
y_pred_b = []
for i in range(100):
    print("repeat: ", i+1)
    index = np.random.choice(np.arange(n_sample), size=1000, replace=False)
    Bgg_tree = Bgg(tx = x[index,:], ty = y[index], column = column, m_d = 16)
    Bgg_tree.fit()
    single_tree = Bgg_tree.tree[0]
    tempy = single_tree.predict(tt_d.to_numpy(copy=True))
    y_pred_s.append(tempy)

    tempy = Bgg_tree.predict(tt_d.to_numpy(copy=True))
    y_pred_b.append(tempy[499])

def helper(x):
    temp = np.ones(x.shape)
    temp[x=='no']=0
    return temp

y_pred_s_a = helper(np.array(y_pred_s))
y_pred_b_a = helper(np.array(y_pred_b))
y_true = helper(tt_d.y.to_numpy(copy=True))

bias_s = bias(y_pred_s_a, y_true)
bias_b = bias(y_pred_b_a, y_true)
variance_s = variance(y_pred_s_a, y_true)
variance_b = variance(y_pred_b_a, y_true)
print("bias^2 (single, Bgg): ", bias_s, bias_b)
print("variance (single, Bgg): ", variance_s, variance_b)
print("total (single, Bgg): ", bias_s + variance_s, bias_b + variance_b)


np.random.seed(2022)
column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
tree_RF = RF(tx = x, ty = y, column = column, m_d = 16, select_features = 2)
tree_RF.fit()

t_e = np.zeros(500)
tt_e = np.zeros(500)
train_prediction = tree_RF.predict(t_d.to_numpy(copy=True))
test_prediction = tree_RF.predict(tt_d.to_numpy(copy=True))

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_prediction[i], tt_d.y.to_numpy(copy=True) )

plt.plot(t_e)
plt.plot(tt_e)
plt.title('RF error (features = 2)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d1.png') 
print("image save 2d1.png") 
plt.show()

print("number of features: 4")
column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
tree_RF = RF(tx = x, ty = y, column = column, m_d = 16, select_features = 4)
tree_RF.fit()

t_e = np.zeros(500)
tt_e = np.zeros(500)
train_prediction = tree_RF.predict(t_d.to_numpy(copy=True))
test_prediction = tree_RF.predict(tt_d.to_numpy(copy=True))

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_prediction[i], tt_d.y.to_numpy(copy=True) )

plt.plot(t_e)
plt.plot(tt_e)
plt.title('RF error (features = 4)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d2.png') 
print("image save 2d2.png") 
plt.show()

print("number of features: 6")
column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
tree_RF = RF(tx = x, ty = y, column = column, m_d = 16, select_features = 6)
tree_RF.fit()

t_e = np.zeros(500)
tt_e = np.zeros(500)
train_prediction = tree_RF.predict(t_d.to_numpy(copy=True))
test_prediction = tree_RF.predict(tt_d.to_numpy(copy=True))

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_prediction[i], tt_d.y.to_numpy(copy=True) )

plt.plot(t_e)
plt.plot(tt_e)
plt.title('RF error (features = 6)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d3.png') 
print("image save 2d3.png") 
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

column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
n_sample = x.shape[0]
y_pred_s = []
y_pred_b = []
for i in range(100):
    print("repeat: ", i+1)
    index = np.random.choice(np.arange(n_sample), size=1000, replace=False)
    Bgg_tree = RF(tx = x[index,:], ty = y[index], column = column, m_d = 16, select_features = 6)
    Bgg_tree.fit()
    single_tree = Bgg_tree.tree[0]
    tempy = single_tree.predict(tt_d.to_numpy(copy=True))
    y_pred_s.append(tempy)

    tempy = Bgg_tree.predict(tt_d.to_numpy(copy=True))
    y_pred_b.append(tempy[499])

def helper(x):
    temp = np.ones(x.shape)
    temp[x=='no']=0
    return temp

y_pred_s_a = helper(np.array(y_pred_s))
y_pred_b_a = helper(np.array(y_pred_b))
y_true = helper(tt_d.y.to_numpy(copy=True))

bias_s = bias(y_pred_s_a, y_true)
bias_b = bias(y_pred_b_a, y_true)
variance_s = variance(y_pred_s_a, y_true)
variance_b = variance(y_pred_b_a, y_true)
print("bias^2 (single, Bgg): ", bias_s, bias_b)
print("variance (single, Bgg): ", variance_s, variance_b)
print("total (single, Bgg): ", bias_s + variance_s, bias_b + variance_b)


data_path = Path('./data/credit_card/data.csv')
colnames = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 
          'PAY_5', 'PAY_6', 'B_A1', 'B_A2', 'B_A3', 'B_A4', 'B_A5', 
          'B_A6', 'P_A1', 'P_A2', 'P_A3', 'P_A4', 'P_A5', 'P_A6', 'y'] 
          																							

data = pd.read_csv(data_path, header=None, names=colnames) 
print("Original data")
print(data.head())

cts_f = ["LIMIT_BAL", "AGE", 'B_A1', 'B_A2', 'B_A3', 'B_A4', 'B_A5', 
          'B_A6', 'P_A1', 'P_A2', 'P_A3', 'P_A4', 'P_A5', 'P_A6']
thresholds = data[cts_f].median()
def preprocessing(df):

    for col in cts_f:
        df.loc[df[col] <= thresholds[col], col] = 0
        df.loc[df[col] > thresholds[col], col] = 1
        df[col] = df[col].map({0: "low", 1: "high"})

    return df
data = preprocessing(data)
shuffle = np.random.choice(np.arange(30000), size = 30000, replace=False)
data = data.iloc[shuffle]


print(data.head())
t_d = data.iloc[:24000,:]
tt_d = data.iloc[24000:,:]
print(t_d.head())


column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
Boost_tree = AdaBoost(tx = x, ty = y, column = column, entropy_base = 16)
Boost_tree.fit()
t_e = np.zeros(500)
tt_e = np.zeros(500)
train_prediction, _ = Boost_tree.predict(t_d.to_numpy(copy=True))
test_prediction, _ = Boost_tree.predict(tt_d.to_numpy(copy=True))

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_prediction[i], tt_d.y.to_numpy(copy=True) )

plt.plot(t_e)
plt.plot(tt_e)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of iterations')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('3ADA.png') 
print("image save 3ADA.png") 
plt.show()


column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
Bgg_tree = Bgg(tx = x, ty = y, column = column, m_d = 16)
Bgg_tree.fit()
t_e = np.zeros(500)
tt_e = np.zeros(500)
train_prediction = Bgg_tree.predict(t_d.to_numpy(copy=True))
test_prediction = Bgg_tree.predict(tt_d.to_numpy(copy=True))

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_prediction[i], tt_d.y.to_numpy(copy=True) )

plt.plot(t_e)
plt.plot(tt_e)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('3Bgg.png') 
print("image save 3Bgg.png") 
plt.show()


column = t_d.columns.to_numpy(copy=True)[:-1]
x = t_d[column].to_numpy(copy=True)
y = t_d.y.to_numpy(copy=True)
tree_RF = RF(tx = x, ty = y, column = column, m_d = 23, select_features = 6)
tree_RF.fit()

t_e = np.zeros(500)
tt_e = np.zeros(500)
train_prediction = tree_RF.predict(t_d.to_numpy(copy=True))
test_prediction = tree_RF.predict(tt_d.to_numpy(copy=True))

for i in range(len(train_prediction)):
    t_e[i] = e_r( train_prediction[i], t_d.y.to_numpy(copy=True) )
    tt_e[i] = e_r( test_prediction[i], tt_d.y.to_numpy(copy=True) )

plt.plot(t_e)
plt.plot(tt_e)
plt.title('RF error (features = 6)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('3RF.png') 
print("image save 3RF.png") 
plt.show()
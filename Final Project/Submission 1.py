import math
import statistics
import numpy as np
import csv

t_data = "income2022f/train_final.csv"
tt_data = "income2022f/test_final.csv"

with open(t_data, mode='r') as f:
    data = []
    next(f)
    for line in f:
        terms = line.strip().split(',')
        data.append(terms)

n_set = {0, 2, 4, 10, 11, 12}

def get_float_val(mylist):
    t_list = mylist
    for k in range(len(t_list)):
        for i in {0, 2, 4, 10, 11, 12}:
            t_list[k][i] = float(mylist[k][i])
    return t_list


data = get_float_val(data)

object = {0: 0, 2: 2, 4: 4, 10: 10, 11: 11, 12: 12}
for i in object:
    object[i] = statistics.median([row[i] for row in data])

for row in data:
    for i in object:
        if row[i] >= object[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'


lab = []
for i in range(13):
    majority_labels = [row[i] for row in data if row[i] != '?']
    lb = max(set(majority_labels), key=majority_labels.count)
    lab.append(lb)

for i in range(len(data)):
    for j in range(13):
        if data[i][j] == '?':
            data[i][j] = lab[j]



with open(tt_data, mode='r') as f:
    tt_bank = []
    next(f)
    for line in f:
        terms = line.strip().split(',') 
        terms.pop(0)
        tt_bank.append(terms)

tt_bank = get_float_val(tt_bank)
for i in object:
    object[i] = statistics.median([row[i] for row in tt_bank])

for row in tt_bank:
    for i in object:
        if row[i] >= object[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'


lab_test = []
for i in range(13):
    majority_labels = [row[i] for row in tt_bank if row[i] != '?']
    lb = max(set(majority_labels), key=majority_labels.count)
    lab_test.append(lb)

for i in range(len(tt_bank)):
    for j in range(13):
        if tt_bank[i][j] == '?':
            tt_bank[i][j] = lab_test[j]

bank_attributes = {
             'age': ['yes', 'no'],
             'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
             'fnwlwgt': ['yes', 'no'],
             'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 
                            'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
             'education-num': ['yes', 'no'],
             'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'],
             'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
                            'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
             'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
             'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
             'sex': ['Female', 'Male', '?'],
             'capital-gain': ['yes', 'no'],
             'capital-loss': ['yes', 'no'],
             'hours-per-week': ['yes', 'no'],
             'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
                                'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                                'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
                                'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?'],
            }

def bank_pos(attribute):
    pos = 0
    if attribute == 'age':
        pos = 0
    if attribute == 'workclass':
        pos = 1
    if attribute == 'fnlwgt':
        pos = 2
    if attribute == 'education':
        pos = 3
    if attribute == 'education-num':
        pos = 4
    if attribute == 'marital-status':
        pos = 5
    if attribute == 'occupation':
        pos = 6
    if attribute == 'relationship':
        pos = 7
    if attribute == 'race':
        pos = 8
    if attribute == 'sex':
        pos = 9
    if attribute == 'capital-gain':
        pos = 10
    if attribute == 'capital-loss':
        pos = 11
    if attribute == 'hours-per-week':
        pos = 12
    if attribute == 'native-country':
        pos = 13
    return pos


def create_list_bank(attribute):
    object = {}
    for attr_val in bank_attributes[attribute]:
        object[attr_val] = []
    return object



def build_empty_list(attribute):
    object = {}
    for attr_val in attribute:
        object[attr_val] = 0
    return object


def G_I(groups, classes):
    n = float(sum([len(groups[attr_val]) for attr_val in groups])) 
    gini = 0.0
    for attribute_value in groups:
        size = float(len(groups[attribute_value]))
        if size == 0:
            continue
        score = 0.0
        for value in classes:
            p = [row[-1] for row in groups[attribute_value]].count(value) / size
            score += p * p
        gini += (1.0 - score) * (size / n)
    return gini


def I_G(groups, classes):
    n = float(sum([len(groups[attr_val]) for attr_val in groups]))
    exp_ent = 0.0
    for attribute_value in groups:
        size = float(len(groups[attribute_value]))
        if size == 0:
            continue
        score = 0.0
        for value in classes:
            p = [row[-1] for row in groups[attribute_value]].count(value) / size
            if p == 0:
                temp = 0
            else:
                temp = p * math.log2(1 / p)
            score += temp
        exp_ent += score * (size / n)
    return exp_ent


def M_E(groups, classes):
    n = float(sum([len(groups[attr_val]) for attr_val in groups]))
    M_E = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue
        score = 0.0
        temp = 0
        for class_val in classes:
            p = [row[-1] for row in groups[attr_val]].count(class_val) / size
            temp = max(temp, p)
            score = 1 - temp
        M_E += score * (size / n)
    return M_E


def data_split_bank(attributes, dataset):
    branch_object = create_list_bank(attributes)
    for row in dataset:
        for attr_val in bank_attributes[attributes]:
            if row[bank_pos(attributes)] == attr_val:
                branch_object[attr_val].append(row)
    return branch_object  


def find_best_split(dataset, attribute):
    if dataset == []:
        return
    if attribute == "bank":
        label_values = list(set(row[-1] for row in dataset))
        metric_object = build_empty_list(bank_attributes)
        for attr in bank_attributes:
            groups = data_split_bank(attr, dataset)
            metric_object[attr] = M_E(groups, label_values)  
        best_attr = min(metric_object, key=metric_object.get)
        b_grps = data_split_bank(best_attr, dataset)
        return {'best_attr': best_attr, 'b_grps': b_grps}



def leaf_node_label(group):
    majority_labels = [row[-1] for row in group]
    return max(set(majority_labels), key=majority_labels.count)



def if_node_divisible(branch_object):
    non_empty_indices = [key for key in branch_object if not (not branch_object[key])]
    if len(non_empty_indices) == 1:
        return False
    else:
        return True


def child_node(node, m_d, c_d, dataset):
    if not if_node_divisible(node['b_grps']):
        for key in node['b_grps']:
            if node['b_grps'][key]:
                node[key] = leaf_node_label(node['b_grps'][key])
            else:
                node[key] = leaf_node_label(sum(node['b_grps'].values(), []))
        return
    if c_d >= m_d:
        for key in node['b_grps']:
            if node['b_grps'][key]:
                node[key] = leaf_node_label(node['b_grps'][key])
            else:
                node[key] = leaf_node_label(sum(node['b_grps'].values(), []))
        return
    for key in node['b_grps']:
        if dataset == "car":
            if node['b_grps'][key]:
                node[key] = find_best_split(node['b_grps'][key], dataset)
                child_node(node[key], m_d, c_d + 1, dataset)
            else:
                node[key] = leaf_node_label(sum(node['b_grps'].values(), []))
        if dataset == "bank":
            if node['b_grps'][key]:
                node[key] = find_best_split(node['b_grps'][key], dataset)
                child_node(node[key], m_d, c_d + 1, dataset)
            else:
                node[key] = leaf_node_label(sum(node['b_grps'].values(), []))

            

def build_t_b(train, m_d):
    root = find_best_split(train, "bank")
    child_node(root, m_d, 1, "bank")
    return root



def label_predict_bank(node, inst):
    if isinstance(node[inst[bank_pos(node['best_attr'])]], dict):
        return label_predict_bank(node[inst[bank_pos(node['best_attr'])]], inst)
    else:
        return node[inst[bank_pos(node['best_attr'])]]  


def error(true_label, predicted):
    count = 0
    for i in range(len(true_label)):
        if true_label[i] != predicted[i]:
            count += 1
    return count / float(len(true_label)) * 100.0



def error_calc_training_bank(tree):
    true_label = []
    pred_seq = []
    for row in data:
        true_label.append(row[-1])
        pre = label_predict_bank(tree, row)
        pred_seq.append(pre)
    return error(true_label, pred_seq)


def error_calc_testing_bank(tree):
    true_label = []
    pred_seq = []
    id = 0
    with open('decisiontree_output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID'] + ['Prediction'])
        for row in tt_bank:
            id += 1
            true_label.append(row[-1])
            pre = label_predict_bank(tree, row)
            pred_seq.append(pre)
            writer.writerow([str(id)] + [str(pre)])
    return error(true_label, pred_seq)



t_d = build_t_b(data, 13)

print("Error Prediction in Training data:", error_calc_training_bank(t_d))
print("Error Prediction in Testing data:", error_calc_testing_bank(t_d))
import numpy as np 
import copy


class Node(object):
    def __init__(self, split_name, column_name, split: np.ndarray, child: np.ndarray, lbs: np.ndarray, weight) -> None:
        self.split_name = split_name 
        self.column_name = column_name 
        self.predict_value = []

        self.child_value = np.unique(split) 
        for v in self.child_value:

            y = lbs[split == v]
            w = weight[split == v]
            yval = np.unique(y)
            ps = np.array([w[y == vy].sum() for vy in yval])
            ps = ps/ps.sum()

            self.predict_value.append(yval[np.argmax( ps)])
           
        yval = np.unique(lbs)
        ps = np.array([weight[lbs == v].sum() for v in yval])
        ps = ps/ps.sum()
        self.predict_value.append(yval[np.argmax( ps)])
            

    def predict(self, x):
        if x not in self.child_value:
            return self.predict_value[-1]
        return self.predict_value[np.argwhere(self.child_value == x).item()]

class DecisionStump(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, column: np.ndarray, criterion: str = "entropy", entropy_base = None) -> None:
        if criterion not in ["entropy"]:
            raise ValueError("criterion should be \"entropy\"")
        self.criterion = criterion
        self.tree = None
        self.tx = tx
        self.ty = ty
        self.column = column 
        self.base = entropy_base 
        if self.criterion == "entropy":
            self.H = self._entropy

    def _entropy(self, lbs: np.ndarray, weight: np.ndarray) -> np.float32:

        yval = np.unique(lbs)
        ps = np.array([weight[lbs == v].sum() for v in yval])
        ps = ps/ps.sum()
        return -(ps * np.log(ps)).sum() if self.base is None else -(ps * np.log(ps)/np.log(self.base)).sum()

    def _IG(self, x: np.ndarray, lbs: np.ndarray, weight: np.ndarray) -> np.float32:

        IG = self.H(lbs, weight)
        values = np.unique(x)
        pxs = np.array([weight[x == v].sum() for v in values])
        pxs = pxs/pxs.sum()
        
        for (value, px) in zip(values, pxs):
            IG -= px * self.H(lbs[x == value], weight[x == value])
        return IG
    
    def _splitter(self, datas: np.ndarray, lbs: np.ndarray, weight: np.ndarray) -> np.int :
        n_feature = datas.shape[1]
        gain = np.zeros(n_feature)
        for i in range(n_feature):
            gain[i] = self._IG(datas[:,i], lbs, weight)
        
        return np.argmax(gain)

    def fit(self, weight) -> None:
        
        split = self._splitter(self.tx, self.ty, weight)

        self.tree = Node(split_name = self.column[split], column_name = np.delete(self.column, split, axis=0), 
                        split = np.squeeze(self.tx[:,split]), child = np.squeeze(np.delete(self.tx,split,axis=1)), 
                        lbs = np.squeeze(self.ty), weight = weight
                        )
        self.prediction = self.predict(self.tx)
        return
    
    def predict_instance(self, obs: np.ndarray, column = None, node = None) -> str:
        
        column = self.column if column is None else column
        temp = self.tree if node is None else node
        index = np.where( column == temp.split_name )
        return temp.predict(obs[index])       

    def predict(self, testx: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("Please fit the model before predicting.")
        n_s = testx.shape[0]
        prediction = np.empty(n_s, dtype=object)
        for i in range(n_s):

            prediction[i] = self.predict_instance(testx[i,:])
        return prediction

    def remove_train_data(self):
        self.tx, self.ty = None, None


class AdaBoost(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, column: np.ndarray, entropy_base = None) -> None:
        self.tx = tx
        self.ty = ty
        self.column = column 
        self.base = entropy_base

        self.n_s = ty.shape[0]
        self.D = np.ones(self.n_s, dtype = np.float64)/self.n_s
        self.alpha = []
        self.h = []

        self.yval = np.unique(ty)
        self.ymap = dict()
        self.ymap[self.yval[0]]=-1
        self.ymap[self.yval[1]]=1

    
    def fit(self, max_iter: int=500):
        for iter in range(1,max_iter+1):
            print("Adaboost fitting, iteration: {}".format(iter))

            mytree = DecisionStump(tx = self.tx, ty = self.ty, column = self.column, criterion = "entropy", entropy_base = self.base)
            mytree.fit(self.D)

            incorrect = mytree.prediction != mytree.ty 
            h_times_y = np.ones(self.n_s)
            h_times_y[incorrect] = -1
            
            epsilon = np.average(incorrect, weights = self.D)       
            
            
            
            if epsilon== 0:
                print("error = 0 at iter={}".format(iter))
                break
            
            store_tree = copy.deepcopy(mytree)
            store_tree.remove_train_data()
            self.h.append(store_tree)
            del store_tree
            

            alpha_t = 0.5*np.log((1-epsilon)/epsilon)
            self.alpha.append(alpha_t)

            self.D = self.D*np.exp(-alpha_t*h_times_y)
            
            
            self.D = self.D/(self.D.sum())
           
    def predict(self, testx: np.ndarray) -> np.ndarray:
        n_s = testx.shape[0]
        prediction = np.zeros(n_s)
        n_models = len(self.h)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))

        stump = []
        predict = []
        for i in range(n_models):
            print("predict y using {} of weak learners".format(i+1))
            h, alpha = self.h[i], self.alpha[i]
            tempy = h.predict(testx)
            stump.append(tempy)
            tempy = np.vectorize(self.ymap.get)(tempy)

            prediction += alpha*tempy

            final_y = np.empty(n_s, dtype=object)
            final_y[prediction<0] = self.yval[0]
            final_y[prediction>=0] = self.yval[1]
            predict.append(final_y.copy())
        return predict, stump
        


class treeNode(object):
    def __init__(self, split_name, column_name, split: np.ndarray, child: np.ndarray, lbs: np.ndarray, depth: int, m_d: int=6) -> None:
        self.split_name = split_name 
        self.column_name = column_name
        self.end = True if np.unique(lbs).shape[0] == 1 or depth==m_d else False
        self.depth = depth

        self.predict_value = []
        self.child = [] 
        self.child_value = np.unique(split) 
        for v in self.child_value:
            self.child.append((child[split == v,:], lbs[split == v]))
            value, counts = np.unique(lbs[split == v], return_counts=True)
            self.predict_value.append(value[np.argmax( counts)])
           
        value, counts = np.unique(lbs, return_counts=True)
        self.predict_value.append(value[np.argmax( counts)]) 
        self.child_nodes = None

    def _set_child_node(self, child_nodes):
        self.child_nodes = child_nodes

    def predict(self, x):
        if x not in self.child_value:
            return self.predict_value[-1]
        return self.predict_value[np.argwhere(self.child_value == x).item()]

    def visit_child(self, x):
        if self.end:

            return 0
        else:
            if x not in self.child_value:
                
                return 0
            visit = np.argwhere(self.child_value == x).item()
            
            return visit

class DT(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, column: np.ndarray, criterion: str = "entropy", m_d: int=16, entropy_base = None) -> None:
        if criterion not in ["entropy", "gini", "me"]:
            raise ValueError("criterion should be \"entropy\", \"gini\", or \"me\"")
        if m_d<1 or not isinstance(m_d, int):
            raise ValueError("m_d should be a positive integer.")
        self.criterion = criterion 
        self.m_d = min(column.shape[0], m_d) #
        self.tree = None
        self.tx = tx
        self.ty = ty
        self.column = column 
        self.base = entropy_base 
        if self.criterion == "entropy":
            self.H = self._entropy
        elif self.criterion == "gini":
            self.H = self._gini
        elif self.criterion == "me":
            self.H = self._ME

    def _entropy(self, lbs: np.ndarray) -> np.float32:

        _, counts = np.unique(lbs, return_counts=True)
        ps = counts / np.size(lbs)
        return -(ps * np.log(ps)).sum() if self.base is None else -(ps * np.log(ps)/np.log(self.base)).sum()

    def _ME(self, lbs: np.ndarray) -> np.float32:

        _, counts = np.unique(lbs, return_counts=True)
        ps = counts / np.size(lbs)
        return 1-ps.max() 

    def _gini(self, lbs: np.ndarray) -> np.float32:

        _, counts = np.unique(lbs, return_counts=True)
        ps = counts / np.size(lbs)
        return 1-np.square(ps).sum() 

    def _IG(self, x: np.ndarray, lbs: np.ndarray) -> np.float32:

        IG = self.H(lbs)
        values, counts = np.unique(x, return_counts=True)
        pxs = counts / np.size(x)
        for (value, px) in zip(values, pxs):
            IG -= px * self.H(lbs[x == value])
        return IG
    
    def _splitter(self, datas: np.ndarray, lbs: np.ndarray) -> None :
        n_feature = datas.shape[1]
        gain = np.zeros(n_feature)
        for i in range(n_feature):
            gain[i] = self._IG(datas[:,i], lbs)
        return np.argmax(gain)

    def fit(self) -> None:
      
        split = self._splitter(self.tx, self.ty)
        depth = 1
        root = treeNode(split_name = self.column[split], column_name = np.delete(self.column, split, axis=0), 
                        split = self.tx[:,split], child = np.delete(self.tx,split,axis=1), lbs = self.ty,
                        depth = depth, m_d = self.m_d)
        def build_tree(node: treeNode, depth: int)-> treeNode:
            
            child_node = []

            for (temp_x, temp_y) in node.child:
                split = self._splitter(temp_x, temp_y)
                temp = treeNode(split_name = node.column_name[split], column_name = np.delete(node.column_name, split, axis=0), 
                                split = temp_x[:,split], child = np.delete(temp_x, split, axis=1), lbs = temp_y, 
                                depth = depth+1, m_d = self.m_d)

                if depth <self.m_d-1 and not temp.end:
                    temp = build_tree(temp, depth+1)
                child_node.append(temp)
            node._set_child_node(child_node)
            return node

        self.tree = root if depth==self.m_d else build_tree(root, depth)
        return
    
    def predict_instance(self, obs: np.ndarray, column = None, node = None) -> str:
        
        column = self.column if column is None else column
        temp = self.tree if node is None else node
        index = np.where( column == temp.split_name )
        if temp.end:
            return temp.predict(obs[index])
        visit = temp.visit_child(obs[index])
        if visit == 0: 
            
            return temp.predict(obs[index])
        else:
            return self.predict_instance(np.delete(obs, index, axis=0), column = temp.column_name, node = temp.child_nodes[visit])        

    def predict(self, testx: np.ndarray) -> np.ndarray:
        n_s = testx.shape[0]
        prediction = np.empty(n_s, dtype=object)
        for i in range(n_s):

            prediction[i] = self.predict_instance(testx[i,:])
        return prediction


class Bgg(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, column: np.ndarray, m_d: int=16) -> None:
        self.n_s = ty.shape[0]
        self.tree = []

        self.tx = tx
        self.ty = ty
        self.column = column
        self.m_d = min(column.shape[0], m_d)
        self.entropy_base = self.m_d 

        self.yval = np.unique(ty)
        self.ymap = dict()
        self.ymap[self.yval[0]]=-1
        self.ymap[self.yval[1]]=1

    
    def fit(self, max_trees: int=500):
        for iter in range(1, max_trees+1):
            print("Bgg fitting, subtree: {}".format(iter))

            index = np.random.choice(np.arange(self.n_s), size=self.n_s, replace=True)
            tx = self.tx[index,:]
            ty = self.ty[index]
            mytree = DT(tx = tx, ty = ty, column = self.column, criterion = "entropy", 
                                  m_d = self.m_d, entropy_base = self.entropy_base)
            mytree.fit()
            self.tree.append(mytree)


    def predict(self, testx: np.ndarray) -> np.ndarray:
        n_s = testx.shape[0]
        prediction = np.zeros(n_s)
        n_models = len(self.tree)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))

        predict = []
        for i in range(n_models):
            print("predict y using {} of Bgg trees".format(i+1))
            tree = self.tree[i]
            tempy = tree.predict(testx)
            tempy = np.vectorize(self.ymap.get)(tempy)
            prediction += tempy

            final_y = np.empty(n_s, dtype=object)
            final_y[prediction<0] = self.yval[0]
            final_y[prediction>=0] = self.yval[1]
            predict.append(final_y.copy())
        return predict

class RF(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, column: np.ndarray, m_d: int=16, select_features: int=6) -> None:
        self.n_s = ty.shape[0]
        self.n_feature = column.shape[0]
        self.tree = []

        self.tx = tx
        self.ty = ty
        self.column = column
        self.select_features = select_features
        self.m_d = min(column.shape[0], m_d)
        self.entropy_base = self.m_d 

        self.yval = np.unique(ty)
        self.ymap = dict()
        self.ymap[self.yval[0]]=-1
        self.ymap[self.yval[1]]=1

    
    def fit(self, max_trees: int=500):
        self.selects = []
        for iter in range(1, max_trees+1):
            print("RF fitting, subtree: {}".format(iter))

            index = np.random.choice(np.arange(self.n_s), size=self.n_s, replace=True)
            select_feature = np.random.choice(np.arange(self.n_feature), size=self.select_features, replace=False)
            self.selects.append(select_feature)
            tx = self.tx[:, select_feature]
            tx = tx[index,:]
            ty = self.ty[index]
            mytree = DT(tx = tx, ty = ty, column = self.column[select_feature], criterion = "entropy", 
                                  m_d = self.m_d, entropy_base = self.entropy_base)
            mytree.fit()
            self.tree.append(mytree)


    def predict(self, testx: np.ndarray) -> np.ndarray:
        n_s = testx.shape[0]
        prediction = np.zeros(n_s)
        n_models = len(self.tree)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))

        predict = []
        for i in range(n_models):
            select_feature = self.selects[i]
            print("predict y using {} of Bgg trees".format(i+1))
            tree = self.tree[i]
            tempy = tree.predict(testx[:,select_feature])
            tempy = np.vectorize(self.ymap.get)(tempy)
            prediction += tempy

            final_y = np.empty(n_s, dtype=object)
            final_y[prediction<0] = self.yval[0]
            final_y[prediction>=0] = self.yval[1]
            predict.append(final_y.copy())
        return predict
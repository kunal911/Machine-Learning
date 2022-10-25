import numpy as np 
import copy


class Node(object):
    def __init__(self, split_name, col_name, split: np.ndarray, child: np.ndarray, lab: np.ndarray, w) -> None:
        self.split_name = split_name 
        self.col_name = col_name
        self.pred_value = []

        self.child_value = np.unique(split) 
        for v in self.child_value:

            y = lab[split == v]
            w = w[split == v]
            yval = np.unique(y)
            probs = np.array([w[y == vy].sum() for vy in yval])
            probs = probs/probs.sum()

            self.pred_value.append(yval[np.argmax( probs)])
            
        yval = np.unique(lab)
        probs = np.array([w[lab == v].sum() for v in yval])
        probs = probs/probs.sum()
        self.pred_value.append(yval[np.argmax( probs)]) 
            

    def pred(self, x):
        if x not in self.child_value:
            return self.pred_value[-1]
        return self.pred_value[np.argwhere(self.child_value == x).item()]

class DS(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, col: np.ndarray, criterion: str = "entropy", entropy_base = None) -> None:
        if criterion not in ["entropy"]:
            raise ValueError("criterion should be \"entropy\"")
        self.criterion = criterion
        self.tree = None
        self.tx = tx
        self.ty = ty
        self.col = col 
        self.base = entropy_base 
        if self.criterion == "entropy":
            self.H = self._entropy

    def _entropy(self, lab: np.ndarray, w: np.ndarray) -> np.float32:

        yval = np.unique(lab)
        probs = np.array([w[lab == v].sum() for v in yval])
        probs = probs/probs.sum()
        return -(probs * np.log(probs)).sum() if self.base is None else -(probs * np.log(probs)/np.log(self.base)).sum()

    def _IG(self, x: np.ndarray, lab: np.ndarray, w: np.ndarray) -> np.float32:

        IG = self.H(lab, w)
        values = np.unique(x)
        pxs = np.array([w[x == v].sum() for v in values])
        pxs = pxs/pxs.sum()
        for (value, px) in zip(values, pxs):
            IG -= px * self.H(lab[x == value], w[x == value])
        return IG
    
    def _splitter(self, datas: np.ndarray, lab: np.ndarray, w: np.ndarray) -> np.int :
        n_feature = datas.shape[1]
        gain = np.zeros(n_feature)
        for i in range(n_feature):
            gain[i] = self._IG(datas[:,i], lab, w)
        return np.argmax(gain)

    def fit(self, w) -> None:
        split = self._splitter(self.tx, self.ty, w)
        self.tree = Node(split_name = self.col[split], col_name = np.delete(self.col, split, axis=0), 
                        split = np.squeeze(self.tx[:,split]), child = np.squeeze(np.delete(self.tx,split,axis=1)), 
                        lab = np.squeeze(self.ty), w = w
                        )
        self.predy = self.pred(self.tx)
        return
    
    def pred_instance(self, obs: np.ndarray, col = None, node = None) -> str:
        col = self.col if col is None else col
        temp = self.tree if node is None else node
        index = np.where( col == temp.split_name )
        return temp.pred(obs[index])       

    def pred(self, testx: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("Please fit the model before preding.")
        n_sample = testx.shape[0]
        predy = np.empty(n_sample, dtype=object)
        for i in range(n_sample):
            #print(i)
            predy[i] = self.pred_instance(testx[i,:])
        return predy

    def remove_train_data(self):
        self.tx, self.ty = None, None


class AB(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, col: np.ndarray, entropy_base = None) -> None:
        self.tx = tx
        self.ty = ty
        self.col = col 
        self.base = entropy_base

        self.n_sample = ty.shape[0]
        self.D = np.ones(self.n_sample, dtype = np.float64)/self.n_sample
        self.alpha = []
        self.h = []

        self.yval = np.unique(ty)
        self.ymap = dict()
        self.ymap[self.yval[0]]=-1
        self.ymap[self.yval[1]]=1

    
    def fit(self, max_iter: int=500):
        for iter in range(1,max_iter+1):
            print("AB fitting, iteration: {}".format(iter))
            mytree = DS(tx = self.tx, ty = self.ty, col = self.col, criterion = "entropy", entropy_base = self.base)
            mytree.fit(self.D)
            incorrect = mytree.predy != mytree.ty 
            h_times_y = np.ones(self.n_sample)
            h_times_y[incorrect] = -1
            epsilon = np.average(incorrect, ws = self.D)       
            
            
            
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
    def pred(self, testx: np.ndarray) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.zeros(n_sample)
        n_models = len(self.h)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))

        stump = []
        pred = []
        for i in range(n_models):
            print("pred y using {} of weak learners".format(i+1))
            h, alpha = self.h[i], self.alpha[i]
            tempy = h.pred(testx)
            stump.append(tempy)
            tempy = np.vectorize(self.ymap.get)(tempy)

            predy += alpha*tempy

            final_y = np.empty(n_sample, dtype=object)
            final_y[predy<0] = self.yval[0]
            final_y[predy>=0] = self.yval[1]
            pred.append(final_y.copy())
        return pred, stump
        


class treeNode(object):
    def __init__(self, split_name, col_name, split: np.ndarray, child: np.ndarray, lab: np.ndarray, depth: int, max_depth: int=6) -> None:
        self.split_name = split_name 
        self.col_name = col_name 
        self.end = True if np.unique(lab).shape[0] == 1 or depth==max_depth else False
        self.depth = depth

        self.pred_value = []
        self.child = []
        self.child_value = np.unique(split) 
        for v in self.child_value:
            self.child.append((child[split == v,:], lab[split == v]))
            value, counts = np.unique(lab[split == v], return_counts=True)
            self.pred_value.append(value[np.argmax( counts)])
        value, counts = np.unique(lab, return_counts=True)
        self.pred_value.append(value[np.argmax( counts)]) 
        self.child_nodes = None

    def _set_child_node(self, child_nodes):
        self.child_nodes = child_nodes

    def pred(self, x):
        if x not in self.child_value:
            return self.pred_value[-1]
        return self.pred_value[np.argwhere(self.child_value == x).item()]

    def visit_child(self, x):
        if self.end:
            return 0
        else:
            if x not in self.child_value:
                return 0
            visit = np.argwhere(self.child_value == x).item()
            return visit

class DT(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, col: np.ndarray, criterion: str = "entropy", max_depth: int=16, entropy_base = None) -> None:
        if criterion not in ["entropy", "gini", "me"]:
            raise ValueError("criterion should be \"entropy\", \"gini\", or \"me\"")
        if max_depth<1 or not isinstance(max_depth, int):
            raise ValueError("max_depth should be a positive integer.")
        self.criterion = criterion 
        self.max_depth = min(col.shape[0], max_depth) 
        self.tree = None
        self.tx = tx
        self.ty = ty
        self.col = col 
        self.base = entropy_base 
        if self.criterion == "entropy":
            self.H = self._entropy
        elif self.criterion == "gini":
            self.H = self._gini
        elif self.criterion == "me":
            self.H = self._ME

    def _entropy(self, lab: np.ndarray) -> np.float32:
        _, counts = np.unique(lab, return_counts=True)
        probs = counts / np.size(lab)
        return -(probs * np.log(probs)).sum() if self.base is None else -(probs * np.log(probs)/np.log(self.base)).sum()

    def _ME(self, lab: np.ndarray) -> np.float32:
        _, counts = np.unique(lab, return_counts=True)
        probs = counts / np.size(lab)
        return 1-probs.max() 

    def _gini(self, lab: np.ndarray) -> np.float32:
        _, counts = np.unique(lab, return_counts=True)
        probs = counts / np.size(lab)
        return 1-np.square(probs).sum() 

    def _IG(self, x: np.ndarray, lab: np.ndarray) -> np.float32:
        IG = self.H(lab)
        values, counts = np.unique(x, return_counts=True)
        pxs = counts / np.size(x)
        for (value, px) in zip(values, pxs):
            IG -= px * self.H(lab[x == value])
        return IG
    
    def _splitter(self, datas: np.ndarray, lab: np.ndarray) -> None :
        n_feature = datas.shape[1]
        gain = np.zeros(n_feature)
        for i in range(n_feature):
            gain[i] = self._IG(datas[:,i], lab)
        return np.argmax(gain)

    def fit(self) -> None:
        split = self._splitter(self.tx, self.ty)
        depth = 1
        root = treeNode(split_name = self.col[split], col_name = np.delete(self.col, split, axis=0), 
                        split = self.tx[:,split], child = np.delete(self.tx,split,axis=1), lab = self.ty,
                        depth = depth, max_depth = self.max_depth)
        def build_tree(node: treeNode, depth: int)-> treeNode:
            
            child_node = []
            for (temp_x, temp_y) in node.child:
                split = self._splitter(temp_x, temp_y)
                temp = treeNode(split_name = node.col_name[split], col_name = np.delete(node.col_name, split, axis=0), 
                                split = temp_x[:,split], child = np.delete(temp_x, split, axis=1), lab = temp_y, 
                                depth = depth+1, max_depth = self.max_depth)

                if depth <self.max_depth-1 and not temp.end:
                    temp = build_tree(temp, depth+1)
                child_node.append(temp)
            node._set_child_node(child_node)
            return node

        self.tree = root if depth==self.max_depth else build_tree(root, depth)
        return
    
    def pred_instance(self, obs: np.ndarray, col = None, node = None) -> str:
        col = self.col if col is None else col
        temp = self.tree if node is None else node
        index = np.where( col == temp.split_name )
        if temp.end:
            return temp.pred(obs[index])
        visit = temp.visit_child(obs[index])
        if visit == 0: 
            return temp.pred(obs[index])
        else:
            return self.pred_instance(np.delete(obs, index, axis=0), col = temp.col_name, node = temp.child_nodes[visit])        

    def pred(self, testx: np.ndarray) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.empty(n_sample, dtype=object)
        for i in range(n_sample):
            predy[i] = self.pred_instance(testx[i,:])
        return predy


class Bgg(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, col: np.ndarray, max_depth: int=16) -> None:
        self.n_sample = ty.shape[0]
        self.tree = []
        self.tx = tx
        self.ty = ty
        self.col = col
        self.max_depth = min(col.shape[0], max_depth)
        self.entropy_base = self.max_depth 

        self.yval = np.unique(ty)
        self.ymap = dict()
        self.ymap[self.yval[0]]=-1
        self.ymap[self.yval[1]]=1

    
    def fit(self, max_trees: int=500):
        for iter in range(1, max_trees+1):
            print("Bgg fitting, subtree: {}".format(iter))
            index = np.random.choice(np.arange(self.n_sample), size=self.n_sample, replace=True)
            tx = self.tx[index,:]
            ty = self.ty[index]
            mytree = DT(tx = tx, ty = ty, col = self.col, criterion = "entropy", 
                                  max_depth = self.max_depth, entropy_base = self.entropy_base)
            mytree.fit()
            self.tree.append(mytree)

    def pred(self, testx: np.ndarray) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.zeros(n_sample)
        n_models = len(self.tree)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))
        pred = []
        for i in range(n_models):
            print("pred y using {} of bgg trees".format(i+1))
            tree = self.tree[i]
            tempy = tree.pred(testx)
            tempy = np.vectorize(self.ymap.get)(tempy)
            predy += tempy

            final_y = np.empty(n_sample, dtype=object)
            final_y[predy<0] = self.yval[0]
            final_y[predy>=0] = self.yval[1]
            pred.append(final_y.copy())
        return pred

class RF(object):
    def __init__(self, tx: np.ndarray, ty: np.ndarray, col: np.ndarray, max_depth: int=16, select_features: int=6) -> None:
        self.n_sample = ty.shape[0]
        self.n_feature = col.shape[0]
        self.tree = []
        self.tx = tx
        self.ty = ty
        self.col = col
        self.select_features = select_features
        self.max_depth = min(col.shape[0], max_depth)
        self.entropy_base = self.max_depth 

        self.yval = np.unique(ty)
        self.ymap = dict()
        self.ymap[self.yval[0]]=-1
        self.ymap[self.yval[1]]=1

    
    def fit(self, max_trees: int=500):
        self.selects = []
        for iter in range(1, max_trees+1):
            print("RF fitting, subtree: {}".format(iter))
            index = np.random.choice(np.arange(self.n_sample), size=self.n_sample, replace=True)
            select_feature = np.random.choice(np.arange(self.n_feature), size=self.select_features, replace=False)
            self.selects.append(select_feature)
            tx = self.tx[:, select_feature]
            tx = tx[index,:]
            ty = self.ty[index]
            mytree = DT(tx = tx, ty = ty, col = self.col[select_feature], criterion = "entropy", 
                                  max_depth = self.max_depth, entropy_base = self.entropy_base)
            mytree.fit()
            self.tree.append(mytree)

    def pred(self, testx: np.ndarray) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.zeros(n_sample)
        n_models = len(self.tree)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))
        pred = []
        for i in range(n_models):
            select_feature = self.selects[i]
            print("pred y using {} of bgg trees".format(i+1))
            tree = self.tree[i]
            tempy = tree.pred(testx[:,select_feature])
            tempy = np.vectorize(self.ymap.get)(tempy)
            predy += tempy

            final_y = np.empty(n_sample, dtype=object)
            final_y[predy<0] = self.yval[0]
            final_y[predy>=0] = self.yval[1]
            pred.append(final_y.copy())
        return pred
import numpy as np
import math
import pandas
import os
import random

from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt
from morphopy.neurontree import NeuronTree as nt
from scipy.spatial import ConvexHull
from scipy.io import scio


def add_edge(father, son):
    father.add_son(son)
    son.father = father


def resample_branch_by_length(branch, length, seq_len=None, eps=1e-5):
    if seq_len is None:
        seq_len = len(branch)

    curr_pos, res = branch[0], length
    answer = [branch[0]]
    for i in range(1, seq_len):
        tres = euclidean_dist(branch[i - 1], branch[i])
        unorm = branch[i] - branch[i - 1]

        unorm = unorm / (np.linalg.norm(unorm) + 1e-9)
        while tres >= res - eps:
            curr_pos = curr_pos + unorm * res
            answer.append(curr_pos)
            tres -= res
            res = length
        curr_pos = branch[i]
        res = res - tres
    return np.array(answer)


def resample_branch_by_step(branch, step, seq_len, eps=1e-5):
    total_length = sum([
        euclidean_dist(branch[i - 1], branch[i])
        for i in range(1, seq_len)
    ])
    return resample_branch_by_length(
        branch, total_length / (step - 1), seq_len, eps
    )


def plot_branch(branch, ax=None, fig=None, projection='xy', **kwargs):
    if not ax:
        if not fig:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.gca()
    proj2idx = {'x': 0, 'y': 1, 'z': 2}
    ax.plot(
        branch[:, proj2idx[projection[0]]],
        branch[:, proj2idx[projection[1]]],
        **kwargs
    )
    return ax


def plot_compare_branch(
        original_branch, new_branch,
        style1=dict(), style2=dict()
):
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 10))
    axs = axs.flatten()
    for tidx, pj in enumerate(['xy', 'xz', 'yz']):
        plot_branch(original_branch, ax=axs[tidx], projection=pj, **style1)
        plot_branch(new_branch, ax=axs[tidx], projection=pj, **style2)
        axs[tidx].set_aspect('equal')
    plt.tight_layout()
    return fig


class Node(object):
    def __init__(self, Idx=None, sons=None, father=None, data=None):
        super(Node, self).__init__()
        self.data = data
        self.sons = sons
        self.father = father
        self.Idx = Idx

    def add_son(self, son):
        if self.sons is None:
            self.sons = []
        self.sons.append(son)


def euclidean_dist(point1, point2):
    assert len(point1) == len(point2)
    dim, answer = len(point1), 0
    for x in range(0, dim):
        answer += (point1[x] - point2[x]) ** 2
    return math.sqrt(answer)


def get_reindex(x, tdict):
    if x not in tdict:
        tdict[x] = len(tdict)
    return tdict[x]


def solve_quadratic_equation(A, B, C, require_real=True):
    delt = B * B - 4 * A * C
    if require_real and delt < 0:
        return None
    x1 = (-B + (delt ** 0.5)) / (A + A)
    x2 = (-B - (delt ** 0.5)) / (A + A)
    return x1, x2


def intersection_seg_ball(point1, point2, center, radius, eps=1e-5):
    if abs(euclidean_dist(point1, center) - radius) < eps:
        return point1
    if abs(euclidean_dist(point2, center) - radius) < eps:
        return point2

    outside1 = euclidean_dist(point1, center) > radius
    outside2 = euclidean_dist(point2, center) > radius
    if not (outside2 ^ outside1):
        return None
    A = sum((point1[i] - point2[i]) ** 2 for i in [0, 1, 2])
    C = sum((point2[i] - center[i]) ** 2 for i in [0, 1, 2])
    B = sum(
        2 * (point1[i] - point2[i]) * (point2[i] - center[i])
        for i in [0, 1, 2]
    )
    x1, x2 = solve_quadratic_equation(A, B, C)
    t = x1 if 0 <= x1 <= 1 else x2
    return np.array((t * point1[i] + (1 - t) * point2[i] for i in [0, 1, 2]))


class Tree(object):
    def __init__(self, root=None):
        super(Tree, self).__init__()
        self.nodes = [root] if root is not None else []

    ######## for measure start ########
    def CompartmentNumber(self):
        return len(self.nodes) - 1

    def dist(self, x, y):
        return np.linalg.norm(np.array(x.data['pos']) - np.array(y.data['pos']))

    def angle(self, x, y, z):
        x = np.array(x.data['pos'])
        y = np.array(y.data['pos'])
        z = np.array(z.data['pos'])
        u = y - x
        v = z - x
        return np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    # return format Total_Sum Minimum Average Maximum S.D.
    # Only Total_Sum counters
    def N_stems(self):
        root = self.nodes[0]
        if root.sons == None:
            l = 0
        else:
            l = len(root.sons)
        return l, 1, 1, 1, 0

    def N_branch_dfs(self, x, depth, Branch_order=None, bif=False):
        if x.sons is None or len(x.sons) == 0:
            # if depth==Branch_order:
            #    print("###*")
            return 0 if bif else 1
        if len(x.sons) == 1:
            return self.N_branch_dfs(x.sons[0], depth, Branch_order, bif)
        tsum = 1
        # if depth==Branch_order:
        #    print("###")
        # print(x.data['pos'])
        for son in x.sons:
            tsum += self.N_branch_dfs(son, depth + 1, Branch_order, bif)
        return tsum

    def _N_branch(self, root, Branch_order=None):
        tsum = 0
        if root.sons != None:
            for son in root.sons:
                tsum += self.N_branch_dfs(son, 0, Branch_order=Branch_order, bif=False)
        return tsum, 1, 1, 1, 0

    def N_branch(self):
        return self._N_branch(self.nodes[0])

    def N_branch_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._N_branch(son))
        return forests

    def _N_bifs(self, root, Branch_order=None):
        tsum = 0
        if root.sons != None:
            for son in root.sons:
                tsum += self.N_branch_dfs(son, 0, Branch_order=Branch_order, bif=True)
        return tsum, 1, 1, 1, 0


    def fetch_all_walks(self, align=True, with_leaf_idx=False):
        answer = []
        if self.nodes[0].sons is not None:
            for son in self.nodes[0].sons:
                self.dfs_all_walks(son, [self.nodes[0].data['pos']], answer)
        leaf_idx = [x[1] for x in answer]
        answer = [x[0] for x in answer]
        if align:
            for idx, lin in enumerate(answer):
                answer[idx] -= lin[0]
        return answer if not with_leaf_idx else (answer, leaf_idx)

    def dfs_all_walks(self, curr, curr_path, answer):
        curr_path.append(curr.data['pos'])
        if curr.sons is None or len(curr.sons) == 0:
            answer.append([np.array(curr_path), curr.Idx])
        else:
            for son in curr.sons:
                self.dfs_all_walks(son, curr_path, answer)
        curr_path.pop()

    def N_bifs(self):
        return self._N_bifs(self.nodes[0])

    def N_bifs_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None or len(son.sons) == 0:
                    # print(_son.Idx,son.Idx)
                    continue
                # else:
                #    print(_son.Idx,son.Idx,[x.Idx for x in son.sons] )
                forests.append(self._N_bifs(son))
                # print(self._N_bifs(son))
        return forests

    def Branch_order_dfs(self, x, orders, depth):
        orders.append(depth)
        if x.sons is None or len(x.sons) == 0:
            return
        if len(x.sons) == 1:
            self.Branch_order_dfs(x.sons[0], orders, depth)
            return
        for son in x.sons:
            self.Branch_order_dfs(son, orders, depth + 1)

    def _Branch_order(self, root):
        orders = []
        if root.sons != None:
            for son in root.sons:
                self.Branch_order_dfs(son, orders, 0)
        if orders == []:
            orders.append(0)
        total = sum(orders)
        l = len(orders)
        sd = np.std(orders)
        return total, min(orders), total / l, max(orders), sd

    def Branch_order(self):
        return self._Branch_order(self.nodes[0])

    def Branch_order_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._Branch_order(son))
        return forests

    def Contraction_dfs(self, x, contractions, depth, branch_pathlength, bif_start):
        if x.sons is None or len(x.sons) == 0:
            length = self.dist(x, bif_start)
            contractions.append(length / max(branch_pathlength, 1e-7))
            return
        if len(x.sons) == 1:
            son = x.sons[0]
            length = self.dist(x, son)
            self.Contraction_dfs(son, contractions, depth, branch_pathlength + length, bif_start)
            return
        length = self.dist(x, bif_start)
        contractions.append(length / max(branch_pathlength, 1e-7))
        for son in x.sons:
            length = self.dist(x, son)
            self.Contraction_dfs(son, contractions, depth + 1, length, x)

    def _Contraction(self, root):
        contractions = []
        if root.sons != None:
            for son in root.sons:
                length = self.dist(root, son)
                self.Contraction_dfs(son, contractions, 0, length, root)
        if contractions == []:
            contractions.append(0)
        total = sum(contractions)
        l = len(contractions)
        sd = np.std(contractions)

        return total, min(contractions), total / l, max(contractions), sd

    def Contraction(self):
        return self._Contraction(self.nodes[0])

    def Contraction_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._Contraction(son))
        return forests

    def Length_dfs(self, x, pa, lengths, depth):
        length = self.dist(x, pa)
        lengths.append(length)
        if x.sons is None or len(x.sons) == 0:
            return
        if len(x.sons) == 1:
            self.Length_dfs(x.sons[0], x, lengths, depth)
            return
        for son in x.sons:
            self.Length_dfs(son, x, lengths, depth)

    def _Length(self, root):
        lengths = []
        if root.sons != None:
            for son in root.sons:
                self.Length_dfs(son, root, lengths, 0)
        if lengths == []:
            lengths.append(0)
        total = sum(lengths)
        l = len(lengths)
        sd = np.std(lengths)
        return total, min(lengths), total / l, max(lengths), sd

    def Length(self):
        return self._Length(self.nodes[0])

    def Length_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._Length(son))
        return forests

    def Branch_pathlength_dfs(self, x, Branch_pathlengths, depth, branch_pathlength, bif_start):
        if x.sons is None or len(x.sons) == 0:
            Branch_pathlengths.append(branch_pathlength)
            return
        if len(x.sons) == 1:
            son = x.sons[0]
            length = self.dist(x, son)
            self.Branch_pathlength_dfs(son, Branch_pathlengths, depth, branch_pathlength + length, bif_start)
            return
        Branch_pathlengths.append(branch_pathlength)
        for son in x.sons:
            length = self.dist(x, son)
            self.Branch_pathlength_dfs(son, Branch_pathlengths, depth + 1, length, x)

    def _Branch_pathlength(self, root):
        Branch_pathlengths = []
        if root.sons != None:
            for son in root.sons:
                length = self.dist(root, son)
                self.Branch_pathlength_dfs(son, Branch_pathlengths, 0, length, root)
        if Branch_pathlengths == []:
            Branch_pathlengths.append(0)
        total = sum(Branch_pathlengths)
        l = len(Branch_pathlengths)
        sd = np.std(Branch_pathlengths)
        return total, min(Branch_pathlengths), total / l, max(Branch_pathlengths), sd

    def Branch_pathlength(self):
        return self._Branch_pathlength(self.nodes[0])

    def Branch_pathlength_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._Branch_pathlength(son))
        return forests

    def PathDistance_dfs(self, x, PathDistances, depth, path_distance):
        PathDistances.append(path_distance)
        if x.sons is None or len(x.sons) == 0:
            return
        if len(x.sons) == 1:
            son = x.sons[0]
            length = self.dist(x, son)
            self.PathDistance_dfs(son, PathDistances, depth, path_distance + length)
            return
        for son in x.sons:
            length = self.dist(x, son)
            self.PathDistance_dfs(son, PathDistances, depth + 1, path_distance + length)

    def _PathDistance(self, root):
        PathDistances = []
        self.PathDistance_dfs(root, PathDistances, 0, 0)
        # for son in root.sons:
        #    length = self.dist(root,son)
        #    self.PathDistance_dfs(son,PathDistances,0,length)
        if PathDistances == []:
            PathDistances.append(0)
        total = sum(PathDistances)
        l = len(PathDistances)
        sd = np.std(PathDistances)
        return total, min(PathDistances), total / l, max(PathDistances), sd

    def PathDistance(self):
        return self._PathDistance(self.nodes[0])

    def PathDistance_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._PathDistance(son))
        return forests

    def EucDistance_dfs(self, x, root, EucDistances, depth):
        length = self.dist(x, root)
        EucDistances.append(length)
        if x.sons is None or len(x.sons) == 0:
            return
        for son in x.sons:
            self.EucDistance_dfs(son, root, EucDistances, depth + 1)

    def _EucDistance(self, root):
        EucDistances = []
        self.EucDistance_dfs(root, root, EucDistances, 0)
        # for son in root.sons:
        #    length = self.dist(root,son)
        #    self.EucDistance_dfs(son,EucDistances,0,length)
        if EucDistances == []:
            EucDistances.append(0)
        total = sum(EucDistances)
        l = len(EucDistances)
        sd = np.std(EucDistances)
        return total, min(EucDistances), total / l, max(EucDistances), sd

    def EucDistance(self):
        return self._EucDistance(self.nodes[0])

    def EucDistance_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._EucDistance(son))
        return forests

    def Bif_ampl_dfs(self, x, bif_ampls, depth, remote):
        if x.sons is None or len(x.sons) == 0:
            return x
        if len(x.sons) == 1:
            return self.Bif_ampl_dfs(x.sons[0], bif_ampls, depth, remote)
        assert len(x.sons) == 2
        lch = self.Bif_ampl_dfs(x.sons[0], bif_ampls, depth + 1, remote)
        rch = self.Bif_ampl_dfs(x.sons[1], bif_ampls, depth + 1, remote)
        if remote:
            _angle = self.angle(x, lch, rch)
            if _angle == _angle:
                bif_ampls.append(_angle)
        else:
            _angle = self.angle(x, x.sons[0], x.sons[1])
            if _angle == _angle:
                bif_ampls.append(_angle)
        return x

    def _Bif_ampl_remote(self, root):
        bif_ampls = []
        if root.sons != None:
            for son in root.sons:
                self.Bif_ampl_dfs(son, bif_ampls, 0, True)
        if bif_ampls == []:
            bif_ampls.append(0)
        total = sum(bif_ampls)
        l = len(bif_ampls)
        sd = np.std(bif_ampls)
        return total, min(bif_ampls), total / l, max(bif_ampls), sd

    def Bif_ampl_remote(self):
        return self._Bif_ampl_remote(self.nodes[0])

    def Bif_ampl_remote_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._Bif_ampl_remote(son))
        return forests

    def _Bif_ampl_local(self, root):
        bif_ampls = []
        if root.sons != None:
            for son in root.sons:
                self.Bif_ampl_dfs(son, bif_ampls, 0, False)
        if bif_ampls == []:
            bif_ampls.append(0)
        total = sum(bif_ampls)
        l = len(bif_ampls)
        sd = np.std(bif_ampls)
        return total, min(bif_ampls), total / l, max(bif_ampls), sd

    def Bif_ampl_local(self):
        return self._Bif_ampl_local(self.nodes[0])

    def Bif_ampl_local_forest(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._Bif_ampl_local(son))
        return forests

    def pc_angle_dfs(self, x, bif_ampls, ancestor, depth, remote):
        if x.sons is None or len(x.sons) == 0:
            return x
        if len(x.sons) == 1:
            return self.pc_angle_dfs(x.sons[0], bif_ampls, ancestor, depth, remote)
        assert len(x.sons) == 2
        lch = self.pc_angle_dfs(x.sons[0], bif_ampls, x, depth + 1, remote)
        rch = self.pc_angle_dfs(x.sons[1], bif_ampls, x, depth + 1, remote)
        if remote:
            _angle = 180 - self.angle(x, lch, ancestor)
            if _angle == _angle:
                bif_ampls.append(_angle)
            _angle = 180 - self.angle(x, rch, ancestor)
            if _angle == _angle:
                bif_ampls.append(_angle)
        else:
            _angle = 180 - self.angle(x, x.sons[0], x.father)
            if _angle == _angle:
                bif_ampls.append(_angle)
            _angle = 180 - self.angle(x, x.sons[1], x.father)
            if _angle == _angle:
                bif_ampls.append(_angle)
        return x

    def _pc_angle(self, root, remote):
        bif_ampls = []
        if root.sons != None:
            for son in root.sons:
                self.pc_angle_dfs(son, bif_ampls, root, 0, remote=remote)
        if bif_ampls == []:
            bif_ampls.append(0)
        total = sum(bif_ampls)
        l = len(bif_ampls)
        sd = np.std(bif_ampls)
        return total, min(bif_ampls), total / l, max(bif_ampls), sd

    def pc_angle_local(self):
        return self._pc_angle(self.nodes[0], False)

    def pc_angle_forest_local(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._pc_angle(son, False))
        return forests

    def pc_angle_remote(self):
        return self._pc_angle(self.nodes[0], True)

    def pc_angle_forest_remote(self):
        forests = []
        if self.nodes[0] != None and self.nodes[0].sons != None:
            for _son in self.nodes[0].sons:
                son = _son
                while (son.sons != None and len(son.sons) == 1):
                    son = son.sons[0]
                if son.sons == None:
                    continue
                forests.append(self._pc_angle(son, True))
        return forests

    ######## for measure end ########


    ######## for sort begin ########
    def sort_son(self, branches, dataset, need_length=False, need_angle=False):
        # mode = "N", "A"
        if not (need_length and need_angle):
            return dataset
        if need_angle:
            angle = self.metric_angle(branches, dataset)
        if need_length:
            length = self.metric_length(branches)
        for idx, data in enumerate(dataset):
            lson = data[1][0]
            rson = data[1][1]
            if need_angle:
                a = angle[lson]
                b = angle[rson]
                if abs(a-b)/max(a,b)>0.1 and b>a:
                    dataset[idx][1] = (data[1][1], data[1][0])
                    dataset[idx][2] = (data[2][1], data[2][0])
                    continue
            if need_length:
                a = length[lson]
                b = length[rson]
                if abs(a-b)/max(a,b)>0.1 and b>a:
                    dataset[idx][1] = (data[1][1], data[1][0])
                    dataset[idx][2] = (data[2][1], data[2][0])
        return dataset

    def calc_anlge(self, father, son):
        u = father[0] - father[-1]
        v = son[-1] - son[0]
        return np.degrees(np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))))

    def metric_angle(self, branches, dataset):
        angle = {}
        for data in dataset:
            father = branches[data[0][-1]]
            lson = branches[data[1][0]]
            rson = branches[data[1][1]]
            angle[data[1][0]] = self.calc_anlge(father, lson)
            angle[data[1][1]] = self.calc_anlge(father, rson)
        return angle

    def metric_length(self, branches):
        length = []
        for branch in branches:
            length.append(np.sum(np.linalg.norm(branch[1:]-branch[:-1], axis=1)))
        return length

    ######## for sort end ########

    def add_node(self, father_idx, data):
        if not isinstance(data, Node):
            data = Node(data=data)
        self.nodes.append(data)
        self.nodes[-1].Idx = len(self.nodes) - 1
        add_edge(self.nodes[father_idx], self.nodes[-1])
        return self.nodes[-1]

    def dfs_leaf_num(self, x):
        if x.sons is None or len(x.sons) == 0:
            x.data['leaf'] = 1
            return
        for son in x.sons:
            self.dfs_leaf_num(son)
        x.data['leaf'] = sum(t.data['leaf'] for t in x.sons)

    def calc_leaf_num(self):
        if self.nodes is None or len(self.nodes) == 0:
            return
        self.dfs_leaf_num(self.nodes[0])

    def check_binary(self):
        son_num = [
            len(node.sons) for node in self.nodes
            if node is not None and node.sons is not None and node.father is not None
        ]
        return all([x <= 2 for x in son_num])

    def fetch_soma_branch(self, align=True, with_leaf_num=False):
        self.calc_leaf_num()
        answer, leaf_num = [], []
        if self.nodes[0].sons is not None:
            for son in self.nodes[0].sons:
                t_path, curr = [self.nodes[0].data['pos']], son
                while True:
                    t_path.append(curr.data['pos'])
                    if curr.sons is None or len(curr.sons) != 1:
                        if with_leaf_num:
                            leaf_num.append(curr.data['leaf'])
                        break
                    else:
                        curr = curr.sons[0]
                answer.append(t_path)
        seq_len = [len(x) for x in answer]
        if with_leaf_num and len(seq_len) != len(leaf_num):
            raise ValueError('soma branch without leaf num found')
        max_len = max(seq_len)
        container = np.zeros([len(answer), max_len, 3])
        for idx, x in enumerate(answer):
            tarray = np.array(x) - np.array(x[0]) if align else np.array(x)
            container[idx][: seq_len[idx]] = tarray
        answer = container
        output = (answer, seq_len) + ((leaf_num,) if with_leaf_num else ())
        return output

    def fetch_all_branches(self, align=False, with_leaf_num=False):
        self.calc_leaf_num()
        branches, leafs = [], []
        if self.nodes[0].sons is not None:
            for son in self.nodes[0].sons:
                self.dfs_branches(son, self.nodes[0], True, branches, leafs)
        branches = [x - x[0] for x in branches] if align else branches
        return (branches, leafs) if with_leaf_num else branches

    def dfs_branches(self, curr, father, is_soma, branches, leafs):
        curr_branch = [father.data['pos']]
        while True:
            curr_branch.append(curr.data['pos'])
            if curr.sons is None or len(curr.sons) != 1:
                break
            curr = curr.sons[0]
        branches.append(np.array(curr_branch))
        leafs.append(curr.data['leaf'] if is_soma else -1)
        if curr.sons is not None:
            for x in curr.sons:
                self.dfs_branches(x, curr, False, branches, leafs)

    def fetch_branch_seq(self, align=False, move=False, need_length=False, need_angle=False, need_type=False, need_id=False):
        if not self.check_binary():
            raise NotImplementedError(
                'not binary-tree-like structure '
                'can\'t extract branch seq dataset from it'
            )
        if self.nodes[0].sons is None:
            return [], [], [], [], []
        self.calc_leaf_num()
        branches, dataset, layer, node = [], [], [], {}
        if need_type:
            neuron_types = []
        else:
            neuron_types = None
        if need_id:
            neuron_id = []
        else:
            neuron_id = None
        for son in self.nodes[0].sons:
            self.dfs_prefix_branch(son, self.nodes[0], branches, dataset, layer, node, [], neuron_types, neuron_id)


        if align > 1:
            offset = self.nodes[0].data['pos']
            for idx, x in enumerate(branches):
                branches[idx] = x - offset
                branches[idx] = resample_branch_by_step(branches[idx], align, len(branches[idx]))
        dataset = self.sort_son(branches, dataset, need_length=need_length, need_angle=need_angle)

        offsets = [0] * len(branches)
        if move:
            for idx, x in enumerate(branches):
                offset = x[0]
                offsets[idx] = offset
                branches[idx] = x - offset

        if need_type and need_id:
            return branches, offsets, dataset, layer, node, neuron_types, neuron_id
        if need_type:
            return branches, offsets, dataset, layer, node, neuron_types
        if need_id:
            return branches, offsets, dataset, layer, node, neuron_id
        return branches, offsets, dataset, layer, node

    def dfs_prefix_branch(self, curr, father, branches, dataset, layer, node, curr_prefix, neuron_types, neuron_id):
        curr_branch = [father.data['pos']]
        if neuron_id is not None:
            curr_id = []
        if neuron_types is not None:
            neuron_types.append(curr.data['type'])
        while True:
            curr_branch.append(curr.data['pos'])
            if neuron_id is not None:
                curr_id.append(curr.Idx)
            if curr.sons is None or len(curr.sons) != 1:
                break
            curr = curr.sons[0]
        branches.append(np.array(curr_branch))
        if neuron_id is not None:
            neuron_id.append(curr_id)
        branch_id = len(branches) - 1
        if curr_prefix:
            depth = layer[curr_prefix[-1]][1] + 1
            root = curr_prefix[0]
            layer.append((root, depth))
            if depth in node[root]:
                node[root][depth].append(branch_id)
            else:
                node[root][depth] = [branch_id]
        else:
            layer.append((branch_id, 0))
            node[branch_id] = {}
            node[branch_id][0] = [branch_id]
        curr_prefix.append(branch_id)
        if curr.sons is not None and len(curr.sons) == 2:
            left_son = self.dfs_prefix_branch(
                curr.sons[0], curr, branches, dataset, layer, node, curr_prefix, neuron_types, neuron_id
            )
            right_son = self.dfs_prefix_branch(
                curr.sons[1], curr, branches, dataset, layer, node, curr_prefix, neuron_types, neuron_id
            )
            if (curr.sons[0].data['leaf'] >= curr.sons[1].data['leaf']):
                dataset.append([
                    deepcopy(curr_prefix), (left_son, right_son),
                    (curr.sons[0].data['leaf'], curr.sons[1].data['leaf'])
                ])
            else:
                dataset.append([
                    deepcopy(curr_prefix), (right_son, left_son),
                    (curr.sons[1].data['leaf'], curr.sons[0].data['leaf'])
                ])
        curr_prefix.pop()
        return branch_id

    def resort_tree(self):
        self.fetch_branch_seq()
        new_Tree = Tree()
        new_Tree.add_node(-1, deepcopy(self.nodes[0].data))
        reidx = {0:0}
        if self.nodes[0].sons!=None and self.nodes[0].sons!=[]:
            for node in self.nodes[0].sons:
                self.dfs_tree(node, new_Tree, reidx)
        return new_Tree

    def dfs_tree(self, node, new_Tree, reidx):
        while True:
            new_Tree.add_node(reidx[node.father.Idx] if node.father is not None else -1, deepcopy(node.data))
            reidx[node.Idx] = new_Tree.nodes[-1].Idx
            if node.sons is None or len(node.sons) != 1:
                break
            node = node.sons[0]

        if node.sons is not None and len(node.sons) == 2:
            self.dfs_tree(node.sons[0], new_Tree, reidx)
            self.dfs_tree(node.sons[1], new_Tree, reidx)

    def tree_from_swc(self, swc, scaling=1.0):
        reidx_dict = {}
        if type(swc) == pandas.DataFrame:
            if swc.size == 0:
                raise ValueError('No points in swc file')
            n = swc['n'].values.astype(int)
            pos = np.array([swc['x'], swc['y'], swc['z']]).T / scaling
            radius = swc['radius'].values
            t = swc['type'].values.astype(int)
            pid = swc['parent'].values.astype(int)
        elif type(swc) == dict:
            if len(swc['n']) == 0:
                raise ValueError('No points in swc file')
            n = np.array(swc['n']).astype(np.int32)
            pos = np.array([swc['x'], swc['y'], swc['z']]).T / scaling
            radius = np.array(swc['radius']).astype(np.float32)
            t = np.array(swc['type']).astype(np.int32)
            pid = np.array(swc['parent']).astype(np.int32)

        else:
            raise NotImplementedError

        new_idx = [
            get_reindex(int(n[idx]), reidx_dict)
            if int(par) == -1 else 0 for idx, par in enumerate(pid)
        ]
        new_idx = [
            get_reindex(int(n[i]), reidx_dict)
            for i in range(len(n))
        ]

        asort = np.argsort(new_idx).tolist()
        n, pos, radius, t, pid = n[asort], pos[asort], \
                                radius[asort], t[asort], pid[asort]

        self.nodes = []
        for i in range(len(n)):
            data = {'pos': pos[i], 'radius': radius[i], 'type': t[i]}
            self.nodes.append(Node(i, None, None, data))

        for i in range(len(n)):
            if pid[i] == -1:
                self.nodes[i].father = None
            else:
                father_reidx = reidx_dict[int(pid[i])]
                self.nodes[i].father = self.nodes[father_reidx]
                self.nodes[father_reidx].add_son(self.nodes[i])
        return self

    def make_like_soma(self):
        if self.nodes[0].sons is None:
            answer = [[self.nodes[0]]]
        answer = [
            self.dfs_cut_branch(x, [self.nodes[0]])
            for x in self.nodes[0].sons
        ]
        ns, father, types = [0], [-1], [1]
        X, Y, Z = self.nodes[0].data['pos']
        radius = [self.nodes[0].data['radius']]
        X, Y, Z = [X], [Y], [Z]
        for pt in answer:
            for idx in range(1, len(pt)):
                ns.append(pt[idx].Idx)
                father.append(pt[idx - 1].Idx)
                types.append(pt[idx].data['type'])
                X.append(pt[idx].data['pos'][0])
                Y.append(pt[idx].data['pos'][1])
                Z.append(pt[idx].data['pos'][2])
                radius.append(pt[idx].data['radius'])
        return Tree().tree_from_swc({
            'n': ns, 'parent': father, 'x': X, 'type': types,
            'y': Y, 'z': Z, 'radius': radius
        })

    def update_radius_from_leaf(self):
        def dfs_update(curr):
            if curr.sons is None or len(curr.sons) == 0:
                return
            for x in curr.sons:
                dfs_update(x)
            curr.data['radius'] = sum(x.data['radius'] for x in curr.sons)

        dfs_update(self.nodes[0])

    def dfs_cut_branch(self, curr, path):
        path.append(curr)
        if curr.sons is None or len(curr.sons) == 0:
            return path
        p_len = list(range(len(curr.sons)))
        tdx = random.choice(p_len)
        return self.dfs_cut_branch(curr.sons[tdx], path)

    def to_swc(self, offset=[0, 0, 0], scaling=1.):
        n = [x.Idx for x in self.nodes]
        pid = [
            x.father.Idx if x.father is not None else -1
            for x in self.nodes
        ]
        radius = [x.data.get('radius', 1) for x in self.nodes]
        X = [x.data['pos'][0] * scaling + offset[0] for x in self.nodes]
        Y = [x.data['pos'][1] * scaling + offset[1] for x in self.nodes]
        Z = [x.data['pos'][2] * scaling + offset[2] for x in self.nodes]
        types = [x.data.get('type', 3) for x in self.nodes]
        df = pandas.DataFrame(
            data={
                'n': n, 'parent': pid, 'x': X, 'type': types,
                'y': Y, 'z': Z, 'radius': radius
            },
            columns=['n', 'type', 'x', 'y', 'z', 'radius', 'parent']
        )
        return df

    def write_to_swc(self, path, offset=[0, 0, 0], scaling=1.):
        df = self.to_swc(offset=offset, scaling=scaling)
        IO_config = {
            'header': False, 'index': False,
            'encoding': 'utf-8', 'sep': ' '
        }
        df.to_csv(path, **IO_config)

    def draw_2d(
            self, fig=None, ax=None, projection='xz',
            axon_color='darkgreen', dendrite_color='darkgrey',
            apical_dendrite_color='grey', x_offset=0, y_offset=0, **kwargs
    ):

        if not ax:
            if not fig:
                fig = plt.figure()
                ax = fig.gca()
            else:
                ax = fig.gca()
        colors = ['k', axon_color, dendrite_color, apical_dendrite_color]

        if projection == 'xy':
            indices = [0, 1]
        elif projection == 'xz':
            indices = [0, 2]
        elif projection == 'yz':
            indices = [1, 2]
        elif projection == 'yx':
            indices = [1, 0]
        elif projection == 'zx':
            indices = [2, 0]
        elif projection == 'zy':
            indices = [2, 1]
        else:
            raise ValueError('projection %s is not defined.' % projection)

        cs, V = [], []
        for x in self.nodes:
            if x.father is None:
                continue
            pos_data = [x.father.data['pos'], x.data['pos']]
            V.append(pos_data)
            cs.append(colors[int(x.data['type']) - 1])

        V = np.array(V)
        x, y = indices

        for pcolor in colors:
            plt_idx = [cs_i == pcolor for cs_i in cs]
            if any(plt_idx):
                ax.plot(
                    V[plt_idx, :, x].T + x_offset,
                    V[plt_idx, :, y].T + y_offset,
                    c=axon_color, linewidth=1.0, **kwargs
                )

        # ax.set_xlabel(projection[0].capitalize() + r'($\mu m$)')
        # ax.set_ylabel(projection[1].capitalize() + r'($\mu m$)')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        return fig, ax

    def draw_3d(self, fig=None, ax=None, projection='xyz',
                axon_color='darkgreen', dendrite_color='darkgrey',
                apical_dendrite_color='grey', x_offset=0, y_offset=0, z_offset=0, **kwargs):
        if not ax:
            if not fig:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
            else:
                ax = fig.gca(projection='3d')
        colors = ['k', axon_color, dendrite_color, apical_dendrite_color]
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_tick_params(color='white', direction='in')
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_tick_params(color='white', direction='in')
        ax.zaxis.set_ticklabels([])
        ax.zaxis.set_tick_params(color='white', direction='in')
        ax.tick_params(left=False, bottom=False, right=False, direction='in', length=0.0)
        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)
        if projection == 'xyz':
            indices = [0, 1, 2]
        elif projection == 'xzy':
            indices = [0, 2, 1]
        elif projection == 'yzx':
            indices = [1, 2, 0]
        elif projection == 'yxz':
            indices = [1, 0, 2]
        elif projection == 'zxy':
            indices = [2, 0, 1]
        elif projection == 'zyx':
            indices = [2, 1, 0]
        else:
            raise ValueError('projection %s is not defined.' % projection)

        cs, V = [], []
        for x in self.nodes:
            if x is None or x.father is None:
                continue
            pos_data = [x.father.data['pos'], x.data['pos']]
            V.append(pos_data)
            cs.append(colors[int(x.data['type']) - 1])

        cs = np.array(cs)
        V = np.array(V)
        x, y, z = indices

        for color, v in zip(cs, V):
            ax.plot3D(
                v[:, x].T + x_offset,
                v[:, y].T + y_offset,
                v[:, z].T + z_offset,
                c=axon_color, linewidth=1.0, **kwargs
            )
        '''
        for pcolor in colors:
            plt_idx = [cs_i == pcolor for cs_i in cs]
            print(V[plt_idx, :, x].shape)
            if any(plt_idx):
                ax.plot3D(
                    V[plt_idx, :, x].T + x_offset,
                    V[plt_idx, :, y].T + y_offset,
                    V[plt_idx, :, z].T + z_offset,
                    axon_color
                )
        '''

        # ax.set_xlabel(projection[0].capitalize() + r'($\mu m$)')
        # ax.set_ylabel(projection[1].capitalize() + r'($\mu m$)')
        # ax.set_zlabel(projection[2].capitalize() + r'($\mu m$)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return fig, ax

    def node_add(self, node):
        data = {'pos': (node.data['pos'] + node.sons[0].data['pos']) / 2,
                'radius': node.data['radius'], 'type': node.data['type']}
        self.nodes.append(Node(len(self.nodes), None, None, data))
        self.nodes[-1].father = self.nodes[node.Idx]
        self.nodes[node.Idx].sons[0] = self.nodes[-1]
        return self.nodes[-1]

    def node_del(self, node):
        node.father.sons.remove(node)
        node = None

    def node_change(self, node1, node2):
        node1.Idx, node2.Idx = node2.Idx, node1.Idx
        self.nodes[node2.Idx], self.nodes[node1.Idx] = node2, node1

    def create_soma(self, soma_list):
        data = {'pos': np.mean([x.data['pos'] for x in soma_list], axis=0),
                'radius': soma_list[0].data['radius'], 'type': soma_list[0].data['type']}
        self.nodes.append(Node(Idx=len(self.nodes), sons=soma_list, father=-1, data=data))

    def resample_tree(self, length=15, cut=10, mode=0):
        answer_tree = Tree(Node(Idx=0, sons=None, father=None, data=self.nodes[0].data))
        self.resample_dfs(self.nodes[0], length, answer_tree, cut, mode)
        return answer_tree

    def resample_dfs(self, curr, length, answer, cut, mode):
        new_cur = answer.nodes[-1]
        for son in curr.sons:
            # 搜寻branch
            data_list = []
            if son.sons is None:
                continue
            if len(son.sons) > 1:
                data_list.append(son.data['pos'])
            else:
                while len(son.sons) == 1:
                    son = son.sons[0]
                    data_list.append(son.data['pos'])
                    if son.sons is None:
                        break

            # 重采样
            if son.sons is None and len(data_list) < cut:
                # 剪枝操作
                continue
            if mode == 0:
                if len(data_list) > length:
                    new_data_list = resample_branch_by_step(data_list, length, len(data_list))
                else:
                    new_data_list = data_list
            else:
                new_data_list = resample_branch_by_length(data_list, mode, len(data_list))

            # 新节点加入新树
            flag = False
            for pos in new_data_list:
                data = {'pos': pos, 'radius': son.data['radius'], 'type': son.data['type']}
                answer.add_node(father_idx=len(answer.nodes) - 1 if flag else new_cur.Idx, data=data)
                flag = True

            # 继续搜索
            del data_list, new_data_list
            if son.sons is not None:
                self.resample_dfs(son, length, answer, cut, mode)

    def cut_node(self):
        for node in self.nodes:
            if node.Idx != 0:
                if node.sons is not None and len(node.sons) > 2:
                    while len(node.sons) > 2:
                        son_node = node.sons[-1]
                        node.sons.pop()
                        new_node = self.node_add(node)
                        son_node.father = new_node
                        new_node.add_son(son_node)
        return self

    def smooth(self, windowsize=10, cut_size=5, forward=True, backward=True, branch=True, mode=0):
        for node in self.nodes:
            if node is None:
                continue
            # 多分叉点预处理
            if node.Idx != 0:
                if node.sons is not None and len(node.sons) > 2:
                    while len(node.sons) > 2:
                        son_node = node.sons[-1]
                        node.sons.pop()
                        new_node = self.node_add(node)
                        son_node.father = new_node
                        new_node.add_son(son_node)

            smooth_list = []
            back_smooth_list = []
            if node.sons is None or len(node.sons) > 1:
                pass
            else:
                # 窗口搜索(前向搜索)
                if forward:
                    cur_node = node
                    door = 0
                    while door < windowsize:
                        if cur_node.father is None:
                            # 到根节点
                            break
                        cur_node = cur_node.father
                        door += 1
                        smooth_list.append(cur_node)
                        if len(cur_node.sons) > 1 and branch:
                            break

                # 窗口搜索(后向搜索)
                if backward:
                    cur_node = node
                    door = 0
                    if node.sons is not None:
                        while door < windowsize:
                            cur_node = cur_node.sons[0]
                            door += 1
                            back_smooth_list.append(cur_node)
                            if cur_node.sons is None:
                                break
                            if len(cur_node.sons) > 1 and branch:
                                break

            if mode != 0:
                branches = [node.data['pos']] + \
                           [x.data['pos'] for x in smooth_list[:mode * min(len(back_smooth_list), len(smooth_list))]] + \
                           [x.data['pos'] for x in back_smooth_list]
            else:
                branches = [node.data['pos']] + \
                           [x.data['pos'] for x in smooth_list] + \
                           [x.data['pos'] for x in back_smooth_list]
            node.data['pos'] = np.mean(branches, axis=0)

        return self


def data_to_swc(ids, parent, x, y, z, radius=None, types=None):
    n_number = len(ids)
    if radius is None:
        radius = [1.] * n_number
    if types is None:
        types = [3 if tx != -1 else 1 for tx in parent]
    df = pandas.DataFrame(
        data={
            'n': ids, 'parent': parent, 'x': x, 'type': types,
            'y': y, 'z': z, 'radius': radius
        },
        columns=['n', 'type', 'x', 'y', 'z', 'radius', 'parent']
    )
    return df


def load_neuron(file_path, scaling=1.0):
    swc = pandas.read_csv(
        file_path, delim_whitespace=True, comment='#',
        names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'],
        index_col=False
    )
    # swc = merge_soma(swc)
    nt = Tree().tree_from_swc(swc, scaling=scaling)
    return nt


def load_neurons(folder, verbose=False, return_reidx=False, scaling=1., return_filelist=False):
    neurons = []
    files = sorted([x for x in os.listdir(folder) if not os.path.isdir(x)])
    reidx = {}
    file_list = []
    for f in (files if not verbose else tqdm(files)):
        if not (f.endswith('.swc') or f.endswith('.SWC')):
            continue
        swc = pandas.read_csv(
            os.path.join(folder, f), delim_whitespace=True, comment='#',
            names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'],
            index_col=False
        )
        swc = merge_soma(swc)
        file_list.append(f)
        reidx[f] = len(neurons)
        neurons.append(Tree().tree_from_swc(swc, scaling=scaling))

    if return_reidx:
        if return_filelist:
            return (neurons, reidx, file_list)
        else:
            return (neurons, reidx)
    elif return_filelist:
        return (neurons, file_list)
    return neurons


def load_neurons_morphopy(folder, verbose=False, return_reidx=False):
    neurons = []
    files = [x for x in os.listdir(folder) if not os.path.isdir(x)]
    reidx = {}
    for f in (files if not verbose else tqdm(files)):
        if not f.endswith('.swc'):
            continue
        # print(os.path.join(folder, f))
        # exit()
        swc = pandas.read_csv(
            os.path.join(folder, f), delim_whitespace=True, comment='#',
            names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'],
            index_col=False
        )
        N = nt.NeuronTree(swc=swc)
        reidx[f] = len(neurons)
        neurons.append(N)
    return (neurons, reidx) if return_reidx else neurons


def load_neuron_morphopy(file_path):
    swc = pandas.read_csv(
        file_path, delim_whitespace=True, comment='#',
        names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'],
        index_col=False
    )
    neuron = nt.NeuronTree(swc=swc)
    return neuron

def merge_soma(swc):
    soma = swc[swc['type'] == 1]
    root_id = np.min(soma['n'].values)
    if soma.shape[0] > 1:
        if soma.shape[0] > 3:
            convex_hull = ConvexHull(soma[['x', 'y', 'z']].values, qhull_options='QJ')
            hull_points = convex_hull.points
            centroid = np.mean(hull_points, axis=0)
            rad = np.max(np.linalg.norm(hull_points - centroid, axis=1))
        else:
            centroid = np.mean(soma[['x', 'y', 'z']].values, axis=0)
            rad = np.mean(soma[['radius']].values)
        connection_locations = [row.n for k, row in swc.iterrows() if
                                row['parent'] in soma['n'].values and row['n'] not in soma['n'].values]
        connected_points = pandas.concat([swc[swc['n'] == n] for n in connection_locations])
        connected_points['parent'] = root_id
        swc.update(connected_points)

        # delete old soma points
        to_delete = [swc[swc['n'] == n].index[0] for n in soma['n'].values]
        swc = swc.drop(swc.index[to_delete])

        # add new soma point
        soma_dict = dict(zip(['n', 'type', 'x', 'y', 'z', 'radius', 'parent'],
                            [int(root_id), int(1), centroid[0], centroid[1], centroid[2], rad, int(-1)]))

        swc = swc.append(pandas.DataFrame(soma_dict, index=[0]))
        swc = swc.sort_index()
    return swc

def mat_to_swv(folder, save_folder):
    files = [x for x in os.listdir(folder) if not os.path.isdir(x)]
    if os.path.exists(save_folder) is False:
        os.mkdir(save_folder, )
    for file in tqdm(files, total=len(files)):
        if file == "infoRGC.mat":
            continue
        name = os.path.join(save_folder, file.split('.')[0] + '.swc')
        # if os.path.exists(name):
        #     continue
        data = scio.loadmat(os.path.join(folder, file))
        tree = data['tree'][0, 0]
        soma = data['somaCoord']
        x, y, z, edge, type_, redius, _ = tree
        edge = edge.tocoo()
        edge = dict(zip(edge.row, edge.col))
        tree = Tree()
        for i in range(x.shape[0]):
            data = {'pos': np.array([x[i, 0], y[i, 0], z[i, 0]]), 'radius': 0, 'type': 2}
            if i == 0:
                data['type'] = 1
                tree.nodes.append(Node(Idx=0, sons=None, father=None, data=data))
            else:
                tree.add_node(father_idx=edge[i], data=data)
        tree.cut_node()
        tree.write_to_swc(path=name)


def retype(target_neuron, generate_neuron):
    branches, offsets, old_dataset, layer, node, neuron_type = target_neuron.fetch_branch_seq(align=10, need_type=True)
    branches, offsets, new_dataset, layer, node, neuron_id = generate_neuron.fetch_branch_seq(align=10, need_id=True)
    for i, branch_id in enumerate(neuron_id):
        for idx in branch_id:
            generate_neuron.nodes[idx].data['type'] = neuron_type[i]
    return generate_neuron

def smooth(tree):
    tree.smooth(windowsize=30, cut_size=5, forward=True, backward=True, branch=True, mode=2)
    return tree


def resample(tree, mode=0):
    new_tree = tree.resample_tree(mode=mode)
    return new_tree


def smooth_swc(folder, save_path):
    files = [x for x in os.listdir(folder) if not os.path.isdir(x)]
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    max_dep = []
    for f in tqdm(files, total=len(files)):
        swc = pandas.read_csv(
            os.path.join(folder, f), delim_whitespace=True, comment='#',
            names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'],
            index_col=False
        )
        swc = merge_soma(swc)
        if os.path.exists(save_path + '/' + f):
            continue

        tree = resample(smooth(Tree().tree_from_swc(swc, scaling=1).cut_node()))
        branches, offsets, dataset, layer, node = tree.fetch_branch_seq()
        max_dep.append(np.max([len(i[0]) for i in dataset]))

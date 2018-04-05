# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:57:03 2018

@author: Trace
"""
import math

def calc_entropy(S):
    if 0 in S:
        return 0
    else:
        total = 0
        for val in S:
            total += -(val/sum(S))*math.log2(val/sum(S))
        return total
    
def calc_gain(S, A):
    total = 0
    for item in A:
        total += (sum(item)/sum(S))*calc_entropy(item)
    return calc_entropy(S) - total

def s_calc(data):
    s_att = {}
    for col in data.columns:
        if col != 'Play':
            s_att[col] = {}
            for val in set(data[col]):
                pos = data[data[col]==val]['Play'].sum()
                s_att[col][val] = [pos, data[data[col]==val]['Play'].count() - pos]
        else:
            pos = data['Play'].sum()
            s_tar = [pos, data['Play'].count() - pos]
    return s_tar, s_att

def calc_max_gain(S_tar, S_att):
    max_gain = 0
    for att in S_att.keys():
        gain = calc_gain(S_tar, S_att[att].values())
        if gain > max_gain:
            max_gain = gain
            attribute = att
    return attribute

class Node:

    def __init__(self, name, parent=None, value=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.value = value
        
    def set_value(self, value):
        self.value = value
        
    def set_parent(self, parent):
        self.parent = parent
        
    def set_children(self, child):
        self.children.append(child)

class ID3:
    
    def __init__(self, data):
        self.data = data
        
    def train(self):
        
        def id3(data):
            if data.iloc[:, -1].sum() == data.iloc[:, -1].count():
                return Node('Target', None, 1)
            elif data.iloc[:, -1].sum() == 0:
                return Node('Target', None, 0)
            else:
                s_tar, s_att = s_calc(data)
                attribute = calc_max_gain(s_tar, s_att)
                parent = Node(attribute)
                for key in s_att[attribute].keys():
                    if s_att[attribute][key][0] == 0:
                        parent.set_children(Node(key, parent, 0))
                    elif s_att[attribute][key][1] == 0:
                        parent.set_children(Node(key, parent, 1))
                    else:
                        child = Node(key, parent, None)
                        parent.set_children(child)
                        child.set_children(id3(data[data[parent.name]==child.name]))
                return parent
        
        self.tree = id3(self.data)
        
    def predict(self, test):
        
        def predict_aux(tree, test, path):
            if tree.value != None:
                return tree.value
            else:
                if tree.name in test.columns:
                    for child in tree.children:
                        if child.name == test[tree.name].values[0]:
                            path.append(child.name)
                            return predict_aux(child, test, path)
                else:
                    path.append(tree.children[0].name)
                    return predict_aux(tree.children[0], test, path)
                
        path = [self.tree.name]     
        return predict_aux(self.tree, test, path), path
    
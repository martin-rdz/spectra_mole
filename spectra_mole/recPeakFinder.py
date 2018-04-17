#! /usr/bin/env python3
# coding=utf-8
"""
Author: radenz@tropos.de

The ancient version of the peak finder for compatibility reasons...


"""
from . import helpers as h

#@profile
def detect_peak_recursive(array, thres, next_step):
    """
    peakfinder with recursion and tree output

    :param array:
    :param thres: initial noise threshold
    :param next_step: function eg lambda thres: thres*step
    :return:
    """

    pt = Tree((0,array.shape[0]), thres, root=True)

    peaks = h.detect_peak_simple(array, lthres=thres)
    [pt.insert((peak[0], peak[1]), thres) for peak in peaks]

    for i in range(1,20):
        thres = next_step(thres)
        peaks = h.detect_peak_simple(array, lthres=thres)
        #print('recursive peak threshold ', lin2z(thres), peaks)# if not silent else None
        [pt.insert((peak[0], peak[1]), thres) for peak in peaks]

    pt.concat()
    #print(pt)
    pt.extendedges()
    #print(pt)
    #print(pt.toflatList())

    return pt.toflatList(), pt


flatten = lambda l: sum(map(flatten, l), []) if isinstance(l,list) else [l]

class Tree():
    def __init__(self, bounds, noise_thres, root=False):
        self.bounds = bounds
        self.children = []
        self.root = root
        self.threshold = noise_thres

    def insert(self, new_bounds, current_thres):
        if new_bounds[0] < self.bounds[0] or new_bounds[1] > self.bounds[1]:
            raise ValueError("child out of parents bounds")
        # print(list(map(lambda x: x.bounds, self.children)))
        fitting_child = list(filter(
            lambda x: x.bounds[0] <= new_bounds[0] and x.bounds[1] >= new_bounds[1],
            self.children))
        # fitting_child = list(fitting_child)
        # recursive insert
        if len(fitting_child) == 1:
            fitting_child[0].insert(new_bounds, current_thres)
        # or insert here
        else:
            self.children.append(Tree(new_bounds, current_thres))
            # print('inserted bounds ', new_bounds)

    def concat(self):
        # print(self.bounds, map(lambda x: x.bounds, self.children))
        if not self.root:
            while len(self.children) == 1:
                # only one child in list
                new_children = self.children[0].children
                # pull it up a layer
                self.children = new_children
        # and apply it recursively
        [child.concat() for child in self.children]

    def extendedges(self):
        # if im the root myself do nothing
        if not self.root and self.children != []:
            # only at level two or so
            innerbounds = []
            for i in range(len(self.children) - 1):
                innerbounds.append(int(round((self.children[i].bounds[1] + self.children[i + 1].bounds[0]) / 2.)))

            new_bounds = [self.bounds[0]] + innerbounds + [self.bounds[1]]
            for child, new_bounds in zip(self.children, zip(new_bounds, new_bounds[1:])):
                child.bounds = new_bounds
        [child.extendedges() for child in self.children]

    def toflatList(self, l=[]):
        if self.bounds[0] != 0 and self.children == []:
            return self.bounds
        else:
            return flatten([t.toflatList() for t in self.children])

    def __str__(self, space=""):
        return "{}\n{}".format(space + str(self.bounds), ''.join([t.__str__("  " + space) for t in self.children]))
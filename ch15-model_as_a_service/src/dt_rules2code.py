# -*- coding: utf-8 -*-
# 张春强
# 《机器学习：软件工程方法与实现》 第15章-模型上线

import numpy as np

TREE_LEAF = sklearn.tree._tree.TREE_LEAF

class DtRules2Code:
    le = '<='
    gt = '>'

    def __init__(self, clf, feature_names):

        self.left = clf.tree_.children_left
        self.right = clf.tree_.children_right
        self.threshold = np.round(clf.tree_.threshold, 3)
        self.features = [feature_names[i] for i in clf.tree_.feature]
        self.value = clf.tree_.value
        self.samples = clf.tree_.n_node_samples

    def _print_tab(self, tabdepth):
        print('    ' * tabdepth, end='')

    def generate_python_code(self):
        def recurse(node, tabdepth=0):
            if (self.right[node] != TREE_LEAF or self.left[node] != TREE_LEAF):
                self._print_tab(tabdepth)
                print('if ' + self.features[node] + ' <= ' +
                      str(self.threshold[node]) + ':')
                if self.left[node] != TREE_LEAF:
                    recurse(self.left[node], tabdepth + 1)

                self._print_tab(tabdepth)
                print('else:')
                if self.right[node] != TREE_LEAF:
                    recurse(self.right[node], tabdepth + 1)
                self._print_tab(tabdepth)
                print('')
            else:
                self._print_tab(tabdepth)
                print('# samples:{},detail:{}'.format(self.samples[node],
                                                      self.value[node]))
                self._print_tab(tabdepth)
                print('return ' + str(np.argmax(self.value[node])))

        recurse(0)

    def generate_c_code(self):
        def recurse(node, tabdepth=0):
            if (self.threshold[node] != -2):
                self._print_tab(tabdepth)
                print("if ( " + self.features[node] + " <= " +
                      str(self.threshold[node]) + " ) {")
                if self.left[node] != TREE_LEAF:
                    recurse(self.left[node], tabdepth + 1)
                self._print_tab(tabdepth)
                print('} else {')
                if self.right[node] != TREE_LEAF:
                    recurse(self.right[node], tabdepth + 1)
                self._print_tab(tabdepth)
                print('}')
            else:
                self._print_tab(tabdepth)
                print('//samples:{},detail:{}'.format(self.samples[node],
                                                      self.value[node]))
                self._print_tab(tabdepth)
                print('return ' + str(np.argmax(self.value[node])), ';')

        recurse(0)

    def generate_sql_code(self, class_names=None):
        idx = np.argwhere(self.left == TREE_LEAF)[:, 0]

        def get_node_path(left, right, child, lineage=None):
            if lineage is None:
                lineage = [child]

            if child in left:
                parent = np.where(left == child)[0][0]
                split = 'l'
            else:
                parent = np.where(right == child)[0][0]
                split = 'r'
            lineage.append(
                (parent, split, self.threshold[parent], self.features[parent]))

            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return get_node_path(left, right, parent, lineage)

        print('CASE ')
        for j, child in enumerate(idx):
            clause = '  WHEN '
            for node in get_node_path(self.left, self.right, child):
                if not isinstance(node, tuple):
                    continue
                i = node
                if i[1] == 'l': sign = self.le
                else: sign = self.gt
                clause = clause + i[3] + sign + str(i[2]) + ' AND '

            clause = clause[:-4] + ' THEN ' + str(np.argmax(self.value[child]))
            print(clause)

        print('ELSE -1 END')
        
        
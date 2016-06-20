#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement Kalotay-Williams-Fabozzi model using a binomial tree


@author: ucaiado

Created on 06/20/2016
"""

# import libraries
from collections import defaultdict

'''
Begin help functions
'''

'''
End help functions
'''


class Node(object):
    '''
    A representation of a single Node in a binary tree. Store all nodes
    created
    '''
    def __init__(self, s_name):
        '''
        Instatiate a Node object. Save all parameter as attributes
        :param s_name: string. The name of the node. Is expected
            to be composed just by 'D'(down) and 'U'(up)
        '''
        # conta a qtde de subidas e descidas para
        # organizacao de nos posteriormente
        i_d = s_name.count('D') * -1
        i_u = s_name.count('U')
        i_len = len(s_name)
        if s_name == '_':
            i_len = 0
        self.i_step = i_len
        self.i_level = i_d+i_u
        self.node_idx = '{},{}'.format(i_len, i_d+i_u)
        # guarda nome e inicia branch
        self.name = str(s_name)
        # inicia parametros do no
        self.i_r = None
        self.i_time = None
        self.i_cupon = None
        self.i_value = None

    def get_childrens(self):
        '''
        Return the possible childrens of the node
        '''
        s_name = self.name
        if self.name == '_':
            s_name = ''
        return s_name + 'D',  s_name + 'U'

    def get_source(self):
        '''
        Return the source of the node
        '''
        s_name = self.name
        if len(s_name) == 1 and s_name != '_':
            return '_'
        elif self.name == '_':
            s_name = ''
        else:
            return s_name[:-1]

    def __str__(self):
        '''
        Return the name of the node
        '''
        return self.name

    def __repr__(self):
        '''
        Return the name of the node
        '''
        return self.name

    def __eq__(self, other):
        '''
        Return if a node has different node_idx from the other
        :param other: node object. Node to be compared
        '''
        if isinstance(other, str):
            i_aux = other.count('D')*-1 + other.count('U')
            i_len = len(other)
            if s_name == '_':
                i_len = 0
            s_aux = '{},{}'.format(i_len, i_aux)
            return self.node_idx == s_aux
        return self.node_idx == other.node_idx

    def __ne__(self, other):
        '''
        Return if a node has the same node_idx from the other
        :param other: node object. Node to be compared
        '''
        return not self.__eq__(other)

    def __hash__(self):
        '''
        Allow the node object be used as a key in a hash
        table
        '''
        return self.node_idx.__hash__()


class BinomialTree(object):
    '''
    A representation of a Binomial Tree
    '''
    def __init__(self, i_steps):
        '''
        Initiate a BinomialTree object. Save all parameter as
        attributes
        :param i_step: integer. Number of steps in the tree
        '''
        # python sets se comportam como hastable. O tempo
        # de procura eh O(1), e nao O(n) como uma lista
        self.set_of_nodes = set([])
        # inicia outras variaveis
        self.i_steps = i_steps
        self.d_step = defaultdict(list)
        self.d_level = defaultdict(list)
        # insere nodes
        node_root = Node('_')
        self.d_level[0] = [node_root]
        self.d_step[0] = [node_root]
        self.set_of_nodes.add(node_root)
        # constroi arvore
        self.set_branchs(i_steps)

    def set_branchs(self, i_steps):
        '''
        Create all the brachs of the binomial tree using the
        first node as root
        :param i_step: integer. Number of steps in the tree
        '''
        # constroi arvore
        for i_step in xrange(1, i_steps):
            for node in self.d_step[i_step-1]:
                s_down, s_up = node.get_childrens()
                self.add_node(i_step, s_down)
                self.add_node(i_step, s_up)

    def add_node(self, i_step, s_name):
        '''
        Include a new node in the tree. To reduce the number
        of nodes, a restriction called recombination condition
        is imposed on the algorithm. This make the binomial
        method more computationally tractable since the number
        of nodes at each step increases by only one node
        :param i_step: integer.
        :param s_name: string.
        '''
        node = Node(s_name)
        if node not in self.set_of_nodes:
            self.set_of_nodes.add(node)
            self.d_step[node.i_step].append(node)
            self.d_level[node.i_level].append(node)

    def __str__(self):
        '''
        Return ascii representation of the binomial tree,
        imposing a limite of 8 nodes to be printed out
        '''
        # limita passos plotados
        i_steps = min(8, self.i_steps)
        # itera dicionario de niveis e insere em string tabulada
        d_aux = {}
        for i_key in xrange(-1 * i_steps, i_steps + 1, 1):
            s_aux = ''
            for i_inner in xrange(0, i_steps + 1):
                valid_node = ''
                for node in self.d_level[i_key]:
                    if node.i_step == i_inner:
                        valid_node = node
                s_aux += '{}\t'.format(valid_node)
            d_aux[i_key] = s_aux[:-1] + '\n'
        # organiza chaves
        l = d_aux.keys()
        l.sort()
        # junta strings criadas em uma unica, de maneir aordenada
        s_rtn = ''
        for i_key in l:
            s_rtn += d_aux[i_key]
        if i_steps != self.i_steps:
            s_rtn += '\nPlotted {} from {} steps'.format(i_steps, self.i_steps)
        return s_rtn


class KWFTree(object):
    '''
    A representation of the Binomial Tree used in Kalotay-
    Williams-Fabozzi model
    '''
    def __init__(self, i_steps, l_short_rates, l_maturity):
        '''
        Initiate a BinomialTree object. Save all parameter as
        attributes
        :param i_step: integer. Number of steps in the tree
        '''
        # python sets se comportam como hastable. O tempo
        # de procura eh O(1), e nao O(n) como uma lista
        self.set_of_nodes = set([])
        self.set_of_nodes_new = set([])
        # inicia outras variaveis
        self.l_short_rates = l_short_rates
        self.l_maturity = l_maturity
        self.i_steps = i_steps
        self.d_step = defaultdict(Node)
        # insere nodes
        node_root = Node('_')
        self.d_step[0] = [node_root]
        self.set_of_nodes.add(node_root)
        # constroi arvore
        self.set_foward(i_steps)

    def set_foward(self, i_steps):
        '''
        Create all the brachs of the binomial tree using the
        first node as root
        :param i_step: integer. Number of steps in the tree
        '''
        # constroi arvore
        for i_step in xrange(1, i_steps):
            self.set_of_nodes_new = set([])
            for node in self.set_of_nodes:
                s_down, s_up = node.get_childrens()
                self.add_node(s_down)
                self.add_node(s_up)
            self.set_of_nodes = self.set_of_nodes_new.copy()

    def set_backward(self):
        '''
        Create all the brachs of the binomial tree the last
        nodes to recreate the tree
        '''
        # constroi arvore
        for i_step in xrange(len(self.set_of_nodes)-1, 0, -1):
            self.set_of_nodes_new = set([])
            for node in self.set_of_nodes:
                s_src = node.get_source()
                self.add_node(s_src)
            self.set_of_nodes = self.set_of_nodes_new.copy()

    def add_node(self, s_name):
        '''
        Include a new node in the tree. To reduce the number
        of nodes, a restriction called recombination condition
        is imposed on the algorithm. This make the binomial
        method more computationally tractable since the number
        of nodes at each step increases by only one node
        :param s_name: string.
        '''
        node = Node(s_name)
        if node not in self.set_of_nodes:
            self.set_of_nodes_new.add(node)
            if node.name.count('D') == node.i_step:
                self.d_step[node.i_step] = node

    def __str__(self):
        '''
        Create a BinomialTree to return ascii representation tree,
        imposing a limite of 8 nodes to be printed out
        '''
        # limita passos plotados
        i_steps = min(8, self.i_steps)
        # TODO: isso tah muito porco
        tree_aux = BinomialTree(i_steps)
        return str(tree_aux)

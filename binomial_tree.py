#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement Kalotay-Williams-Fabozzi model using a binomial tree
reference: http://www.kalotay.com/sites/default/files/private/FAJ93.pdf

@author: ucaiado

Created on 06/20/2016
"""

# import libraries
from collections import defaultdict
import math
import pandas as pd
from scipy import optimize

'''
Begin help functions
'''


class DIFFERENT_LENGTHS_ERROR(Exception):
    '''
    DIFFERENT_LENGTHS_ERROR is raised by the init method of the KWFTree class
    '''
    pass


class NOT_FITTED_ERROR(Exception):
    '''
    DIFFERENT_LENGTHS_ERROR is raised by the get_lattice method from KWFTree
    '''
    pass


def kwf_sde(f_init_rate, f_sigma, i_above_min, f_time):
    '''
    Return the value of the short rate given by the dynamic given in the paper
    :param f_time: float. time to maturity
    :param i_above_min: integer.
    '''
    theta_dt = 0.
    # sigma_w = f_sigma * (f_time**0.5) * (i_above_min * 1.)
    sigma_w = f_sigma * (i_above_min * 1.)
    return f_init_rate * math.exp(sigma_w)


'''
End help functions
'''


class Node(object):
    '''
    A representation of a single Node in a binary tree
    '''
    def __init__(self, s_name, f_prob=0.5):
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
        self.f_r = 0.
        self.f_time = 0.
        self.f_cupon = 0.
        self.f_value = 0.
        self.f_prob = 0.5

    def set_values(self, f_cupon, f_value, f_r=None, f_time=None):
        '''
        Set values of the node
        '''
        if f_r:
            self.f_r = f_r
        if f_time:
            self.f_time = f_time
        if self.name != '_':
            self.f_cupon = f_cupon
        self.f_value = f_value

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
        Return the possible childrens of the node
        '''
        s_name = self.name
        if len(s_name) == 1 and s_name != '_':
            return '_'
        elif self.name == '_':
            s_name = ''
        else:
            return s_name[:-1]

    def copy(self):
        '''
        Return a copy of the node
        '''
        # copia todos os atributos
        node = Node(self.name)
        node.i_step = self.i_step
        node.i_level = self.i_level
        node.node_idx = self.node_idx
        node.f_cupon = self.f_cupon
        node.f_value = self.f_value
        node.f_r = self.f_r
        node.f_time = self.f_time
        node.f_cupon = self.f_cupon
        node.f_prob = self.f_prob
        return node

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
        Initiate a BinomialTree object. Save all parameter as attributes
        :param i_step: integer. Number of steps in the tree
        '''
        # inicia outras variaveis
        self.i_steps = i_steps
        self.d_step = defaultdict(list)
        self.d_level = defaultdict(list)
        # python sets se comportam como hastable. O tempo
        # de procura eh O(1), e nao O(n) como uma lista
        self.set_of_nodes = set([])
        self.d_nodes = defaultdict(float)

    def add_node(self, s_name, f_time):
        '''
        Include a new node in the tree. To reduce the number of nodes, a
        restriction called recombination condition is imposed on the algorithm.
        This make the binomial method more computationally tractable since the
        number of nodes at each step increases by only one node
        :param s_name: string.
        :param f_time: float.
        '''
        node = Node(s_name)
        if node not in self.set_of_nodes:
            node.f_time = f_time
            self.set_of_nodes.add(node)
            self.d_nodes[node] = node
            self.d_step[node.i_step].append(node)
            self.d_level[node.i_level].append(node)

    def _go_foward(self, i_steps):
        '''
        Create all the brachs of the binomial tree using the first node as root
        :param i_step: integer. Number of steps in the tree
        '''
        # insere primeiro node
        node_root = Node('_')
        self.d_level[0] = [node_root]
        self.d_step[0] = [node_root]
        self.set_of_nodes.add(node_root)
        self.d_nodes[node_root] = node_root
        # constroi arvore
        for i_step in xrange(1, i_steps):
            for node in self.d_step[i_step-1]:
                s_down, s_up = node.get_childrens()
                self.add_node(s_down, i_step)
                self.add_node(s_up, i_step)

    def __getitem__(self, s_idx):
        '''
        Allow direct access to the nodes of the object
        '''
        if isinstance(s_idx, str):
            s_idx = Node(s_idx)
        return self.d_nodes[s_idx]

    def __str__(self):
        '''
        Return ascii representation of the binomial tree, imposing a limite of
        eight nodes to be printed out
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


class KWFTree(BinomialTree):
    '''
    A representation of a Binomial Tree used by Kalotay-Williams-Fabozzi model
    '''
    def __init__(self, l_short_rates, l_maturities):
        '''
        Initiate a KWFTree object. Save all parameter as attributes
        :param l_short_rates: list. The short rates to be fitted
        :param l_maturities: list. The maturities of the rates
        '''
        # checa se listas tem mesmo tamanho
        if len(l_short_rates) != len(l_maturities):
            raise DIFFERENT_LENGTHS_ERROR
        super(KWFTree, self).__init__(len(l_short_rates))
        self.l_maturities = l_maturities
        self.l_short_rates = l_short_rates
        self.already_fitted = False

        # insere nodes
        node_root = Node('_')
        node_root.set_values(f_r=l_short_rates[0],
                             f_time=l_maturities[0],
                             f_cupon=0,
                             f_value=None)
        self.d_level[0] = [node_root]
        self.d_step[0] = [node_root]
        self.set_of_nodes.add(node_root)
        self.d_nodes[node_root] = node_root
        # constroi arvore
        self._go_foward(self.i_steps)

    def fit_foward_curve(self, f_sigma, f_faceval=100., func_rate=kwf_sde):
        '''
        Fit the foward curve using the short term rates
        :param f_sigma: float. Dispersion of the short rate
        :*param f_faceval. float. Face value of the hypothetical bond
        :*param func_rate: function obj. The dynamics of the short rate
        '''
        self.f_face_value = f_faceval
        self.f_sigma = f_sigma
        self.func_rate = func_rate
        for i_step in xrange(1, self.i_steps):
            self._go_backward(i_step)
        self.already_fitted = True

    def get_lattice(self):
        '''
        Return the interest rate lattice produced by the tree simulation
        '''
        if not self.already_fitted:
            raise NOT_FITTED_ERROR
        d_rtn = {}
        # cria ultimos nos sem recombinacao
        l_nodes = []
        for node in self.d_step[self.i_steps-2]:
            # refaz todos os nodes do ultimo passo
            s_d, s_u = node.get_childrens()
            node_down = self[s_d].copy()
            node_down.name = s_d
            node_up = self[s_u].copy()
            node_up.name = s_u
            l_nodes.append(node_down)
            l_nodes.append(node_up)
        # itera para recriar trelica
        for idx, node in enumerate(l_nodes):
            s_name = node.name
            s_key = idx
            d_rtn[s_key] = []
            for idx in xrange(len(s_name)+1):
                d_rtn[s_key].append(float(self[s_name[:idx]].f_r))
        # monta dataframe
        df = pd.DataFrame(d_rtn).T
        df = df + 1
        df = df.cumprod(axis=1)
        df = df.apply(lambda x: x**(1./(x.name+1))) - 1
        df = df.T
        df.index = self.l_maturities
        return df

    def get_description(self):
        '''
        Print a description of each node in the last state
        '''
        s_aux = '     {:3}{:10}{:10}{:10}{:10}'
        print s_aux.format('', 'cupon', 'valor', 'taxa', 'prazo')
        for i_step in self.d_step.keys():
            for node in self.d_step[i_step]:
                f_val = float(node.f_value)
                if not f_val:
                    f_val = 0
                f_val1 = float(node.f_r)
                if not f_val1:
                    f_val1 = 0
                s_aux = '{:3}{:10.2f}{:10.3f}{:10.3f}{:10.4f}'
                print s_aux.format(node.name,
                                   node.f_cupon,
                                   f_val,
                                   f_val1 * 100,
                                   node.f_time)

    def _get_short_rate(self, f_init_rate, node):
        '''
        Get the value of the short rate based on the short rate dynamic setted
        :param f_init_rate: float. the short rate setted to the lower node
        :param node: Node object. Current Node
        '''
        i_above_min = node.i_level - (node.i_step * -1)
        f_rtn = self.func_rate(f_init_rate,
                               self.f_sigma,
                               i_above_min,
                               node.f_time)
        return f_rtn

    def _get_time_step(self, i_step):
        '''
        Get the length of the time step
        :param i_step: integer. current step
        '''
        f_time = self.l_maturities[i_step]
        if i_step == 0:
            f_time_0 = 0.
        else:
            f_time_0 = self.l_maturities[i_step-1]
        return f_time-f_time_0

    def _go_foward(self, i_steps):
        '''
        Create all the brachs of the binomial tree using the
        first node as root
        :param i_step: integer. Number of steps in the tree
        '''
        # constroi arvore
        l_time = [0.] + self.l_maturities
        for i_step in xrange(1, i_steps+1):
            # cria nos do step
            for node in self.d_step[i_step-1]:
                f_time = l_time[i_step] - l_time[i_step-1]
                s_down, s_up = node.get_childrens()
                self.add_node(s_down, f_time)
                self.add_node(s_up, f_time)

    def _go_backward(self, i_step):
        '''
        Create all the brachs of the binomial tree using the
        first node as root
        :param i_step: integer. Number of steps in the tree
        '''
        # preenche valores no passo posterior
        f_initial_rate = self.l_short_rates[i_step]
        f_face_value = self.f_face_value

        for node in self.d_step[i_step+1]:
            f_cupon = f_face_value * ((1+f_initial_rate)**node.f_time-1)
            node.set_values(f_cupon=f_cupon,
                            f_value=f_face_value)
        # itera taxas para fazer root ficar com valor de zero
        res = optimize.minimize(self._set_all_values,
                                f_initial_rate,
                                args=(f_initial_rate, i_step))

    def _set_all_values(self, f_rate, f_initial_rate, i_step, b_print=False):
        '''
        Return the squared diference between the estimated value of a
        hypothetical bond and the desired value. As this process is a
        martingale, the current value should be equal to the face value of the
        bond
        :param f_rate: float. rate to discount the hypothetical bond
        :param f_cupon: float. Cupon of the hypothetical bond
        :param i_step: integer. Step to start iteration.
        '''
        # preenche taxas do passo atual
        if b_print:
            print f_rate
        f_face_value = self.f_face_value
        for node in self.d_step[i_step]:
            node.f_r = self._get_short_rate(f_rate, node)
        # itera toda a arvore. Eh esperado que nos anteriores jah
        # tenham a taxa setada
        for i_aux_step in xrange(i_step, -1, -1):
            for node in self.d_step[i_aux_step]:
                # calcula cupon usando taxa atual
                f_cupon = f_face_value * ((1+f_initial_rate)**node.f_time-1)
                # calcula valor presente pegabdo valores dos childrens
                s_down, s_up = node.get_childrens()
                node_down = self[s_down]
                node_up = self[s_up]
                f_aux = (node_down.f_value + node_down.f_cupon)
                f_aux *= node_down.f_prob
                f_aux += (node_up.f_value + node_up.f_cupon) * node_up.f_prob
                f_aux /= (1+node.f_r)**(node_down.f_time)
                node.set_values(f_cupon=f_cupon,
                                f_value=f_aux)
        return (self['_'].f_value - self.f_face_value)**2

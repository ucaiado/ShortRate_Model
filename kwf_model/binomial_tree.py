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
import matplotlib.pylab as plt
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


class TO_BIG_TO_CREATE_ERROR(Exception):
    '''
    DIFFERENT_LENGTHS_ERROR is raised by the get_lattice method from KWFTree
    '''
    pass


def kwf_sde_adj(f_init_rate, f_sigma, i_above_min, f_time, i_step):
    '''
    Return the value of the short rate given by the dynamic given in the paper
    :param f_sigma: float. Foward one-year rate volatility af all times
    :param f_time: float. time until this node
    :param i_above_min: integer. number of nodes  above the lower node
    :param i_step: integer. The step number
    '''
    theta_dt = 0.
    # sigma_w = f_sigma * (f_time**0.5) * (i_above_min * 1.)
    f_sigma1 = f_sigma * f_time / (i_step * 1.)
    sigma_w = f_sigma1 * (i_above_min * 1.)  # equivalente a 2 * sigma
    # como a initial rate pode ser negativa, calculo o incemento
    f_incr = abs(f_init_rate * (math.exp(sigma_w)-1))
    return f_init_rate + f_incr, f_sigma1


def kwf_sde(f_init_rate, f_sigma, i_above_min, f_time, i_step):
    '''
    Return the value of the short rate given by the dynamic given in the paper
    :param f_sigma: float. Foward one-year rate volatility af all times
    :param f_time: float. time between the last node and this node
    :param i_above_min: integer. number of nodes  above the lower node
    '''
    theta_dt = 0.
    # sigma_w = f_sigma * (f_time**0.5) * (i_above_min * 1.)
    sigma_w = f_sigma * (i_above_min * 1.)
    # como a initial rate pode ser negativa, calculo o incemento
    f_incr = abs(f_init_rate * (math.exp(sigma_w)-1))
    return f_init_rate + f_incr, f_sigma


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
        # inicia variaveis para precificacao de instrumentos
        self.f_cupon_precify = 0.
        self.f_value_precify = 0.
        # inciia variavel para guardar sigma
        self.f_sigma = 0.

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

    def set_values_to_precify(self, f_value, f_cupon=None):
        '''
        Set values of the node to be used in precification of instruments
        :param f_value: float. Current instrument value in the node
        :*param f_cupon: float. copon value in the node
        '''
        if not isinstance(f_cupon, type(None)):
            if self.name != '_':
                self.f_cupon_precify = f_cupon
        self.f_value_precify = f_value

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
        for i_step in xrange(1, i_steps+1):
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
        l.sort(reverse=True)
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

    def fit_foward_curve(self, f_sigma, f_faceval=100., func_rate=kwf_sde,
                         b_use_pu=False):
        '''
        Fit the foward curve using the short term rates
        :param f_sigma: float. Foward one-year rate volatility af all times
        :*param f_faceval. float. Face value of the hypothetical bond
        :*param func_rate: function obj. The dynamics of the short rate
        :*param b_use_pu: boolean. If should use the pu in optimization step
        '''
        self.f_face_value = f_faceval
        self.l_pu = []
        for f_rate, f_mat in zip(self.l_short_rates, self.l_maturities):
            self.l_pu.append(f_faceval/(1+f_rate)**f_mat)
        self.f_sigma = f_sigma
        self.func_rate = func_rate
        for i_step in xrange(1, self.i_steps):
            f_pu = None
            if b_use_pu:
                f_pu = self.l_pu[i_step]
            self._go_backward(i_step, f_pu)
        self.already_fitted = True

    def get_fowards_dynamics(self):
        '''
        Return all paths of foward rate generated by the binomial tree, without
        considering the recombing tree approach. It will create a structure
        with 2^i_steps
        '''
        # lanca erro se ainda nao houver taxas geradas para cada no
        if not self.already_fitted:
            raise NOT_FITTED_ERROR
        # lanca erro se a matriz aser gerada for maior que 500,000 linhas
        if (2**self.i_steps) > 50*10e4:
            raise TO_BIG_TO_CREATE_ERROR
        d_rtn = {}
        # recria toda estrutura novamente sem recombinacao
        d_nodes_new = defaultdict(list)
        d_nodes_new[0].append(self.d_step[0][0])
        for idx in xrange(1, self.i_steps):
            for node in d_nodes_new[idx-1]:
                s_d, s_u = node.get_childrens()
                node_down = self[s_d].copy()
                node_down.name = s_d
                node_up = self[s_u].copy()
                node_up.name = s_u
                d_nodes_new[idx].append(node_down)
                d_nodes_new[idx].append(node_up)
        l_nodes = d_nodes_new[idx]
        # itera para recriar trelica
        for idx, node in enumerate(l_nodes):
            s_name = node.name
            s_key = idx
            d_rtn[s_key] = []
            for idx in xrange(len(s_name)+1):
                d_rtn[s_key].append(float(self[s_name[:idx]].f_r))
        return pd.DataFrame(d_rtn).T

    def get_lattice(self):
        '''
        Return the interest rate lattice produced by the tree simulation
        '''
        # monta dataframe
        df = self.get_fowards_dynamics()
        df = df + 1
        df = df.cumprod(axis=1)
        df = df.apply(lambda x: x**(1./(x.name+1))) - 1
        df = df.T
        df.index = self.l_maturities
        return df

    def get_description(self, i_limit=10, b_precify=False, b_index=False):
        '''
        Print a description of each node in the last state
        :param i_limit: integer. Limit to print the results
        :param b_precify: boolean. Return the values in the last precification
        :param b_index: boolean. Return the index instead of the name
        '''
        s_aux = '     {:12}{:10}{:10}{:10}{:10}{:10}{:10}'
        print s_aux.format('', 'cupon', 'valor', 'taxa', 'prazo',
                           'venc', 'sigma')
        i_rows = 0
        for i_step in self.d_step.keys():
            for node in self.d_step[i_step]:
                f_val = float(node.f_value)
                if not f_val:
                    f_val = 0
                f_val1 = float(node.f_r)
                if not f_val1:
                    f_val1 = 0
                s_aux = '{:12}{:10.2f}{:10.3f}{:10.3f}{:10.4f}{:10.4f}{:10.4f}'
                i_rows += 1
                f_venc = self.l_maturities[node.i_step-1]
                if node.name == '_':
                    f_venc = 0.
                f_cupon = node.f_cupon
                if b_precify:
                    f_cupon = node.f_cupon_precify
                    f_val = float(node.f_value_precify)
                s_name = node.name
                if b_index:
                    s_name = node.node_idx
                print s_aux.format(s_name,
                                   f_cupon,
                                   f_val,
                                   f_val1 * 100,
                                   node.f_time,
                                   f_venc,
                                   node.f_sigma * 100)
                if i_rows > i_limit:
                    print '...'
                    return

    def plot_curves(self):
        '''
        Plot the curve approximated and the original curve
        '''
        df = self.get_lattice()
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1.set_xlabel('Prazos em anos')
        ax2.set_xlabel('Prazos em anos')
        ax1.set_ylabel('Taxas')
        ax1.set_title('curva media gerada\npela arvore')
        ax2.set_title('curva de marcado\n')
        df.T.mean().plot(ax=ax1)
        df_aux = pd.DataFrame(self.l_short_rates, self.l_maturities)
        df_aux.plot(legend=False, ax=ax2)
        f.tight_layout()
        return f

    def plot_lattice(self):
        '''
        Plot the lattice generated by the tree
        '''
        df1 = self.get_lattice()
        ax1 = df1.plot(legend=False)
        ax1.set_xlabel('Prazos em anos')
        ax1.set_ylabel('Taxas')
        ax1.set_title(u'Curvas Geradas pela Árvore\n', fontsize=16)
        return ax1

    def plot_hist(self):
        '''
        Return a histogram of the last step of the tree
        '''
        df = self.get_lattice()
        ax1 = df.tail(1).T.hist(bins=17)
        ax1[0][0].set_title(u'Distribuição de Valores do Último Passo\n',
                            fontsize=16)
        return ax1

    def _get_cupon(self, f_face_value, f_initial_rate, f_time):
        '''
        Get the cupon value to the parameters passed
        :param f_face_value: float.
        :param f_initial_rate: float.
        :param f_time: float
        '''
        f_cupon = f_face_value * ((1+f_initial_rate)**f_time-1)
        return f_cupon

    def _get_short_rate(self, f_init_rate, node):
        '''
        Get the value of the short rate based on the short rate dynamic setted
        :param f_init_rate: float. the short rate setted to the lower node
        :param node: Node object. Current Node
        '''
        i_above_min = node.i_level - (node.i_step * -1)
        f_rtn, f_sig = self.func_rate(f_init_rate,
                                      self.f_sigma,
                                      i_above_min,
                                      self.l_maturities[node.i_step-1],
                                      node.i_step)
        node.f_sigma = f_sig
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

    def _go_backward(self, i_step, f_pu=None):
        '''
        Create all the brachs of the binomial tree using the
        first node as root
        :param i_step: integer. Number of steps in the tree
        :param f_pu: float. the face value to be used in optimization
        '''
        # preenche valores no passo posterior
        f_initial_rate = self.l_short_rates[i_step]
        f_face_value = self.f_face_value

        for node in self.d_step[i_step+1]:
            f_cupon = self._get_cupon(f_face_value,
                                      f_initial_rate,
                                      node.f_time)
            node.set_values(f_cupon=f_cupon,
                            f_value=f_face_value)
        # itera taxas para fazer root ficar com valor de zero
        res = optimize.leastsq(self._set_all_values,
                               x0=-0.05,
                               xtol=1e-7,
                               args=(f_initial_rate, i_step, f_pu))

    def _set_all_values(self, f_delta, f_initial_rate, i_step, f_pu=None):
        '''
        Return the squared diference between the estimated value of a
        hypothetical bond and the desired value. As this process is a
        martingale, the current value should be equal to the face value of the
        bond
        :param f_rate: float. rate to discount the hypothetical bond
        :param f_cupon: float. Cupon of the hypothetical bond
        :param i_step: integer. Step to start iteration.
        :*param f_pu: float. Value to be used in optimization
        '''
        # preenche taxas do passo atual
        f_optim = self.f_face_value
        if f_pu:
            f_optim = f_pu
        f_rate = f_initial_rate + f_delta
        f_face_value = self.f_face_value
        for node in self.d_step[i_step]:
            node.f_r = self._get_short_rate(f_rate, node)
        # itera toda a arvore. Eh esperado que nos anteriores jah
        # tenham a taxa setada
        for i_aux_step in xrange(i_step, -1, -1):
            for node in self.d_step[i_aux_step]:
                # calcula cupon usando taxa atual
                f_cupon = self._get_cupon(f_face_value,
                                          f_initial_rate,
                                          node.f_time)
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
        return (self['_'].f_value - f_optim)**2


class KWFTreePU(KWFTree):
    '''
    A representation of a Binomial Tree used by Kalotay-Williams-Fabozzi model
    using th ePU of the contracts to fit the curve
    '''
    def __init__(self, l_short_rates, l_maturities):
        '''
        Initiate a KWFTreePU object. Save all parameter as attributes
        :param l_short_rates: list. The short rates to be fitted
        :param l_maturities: list. The maturities of the rates
        '''
        super(KWFTreePU, self).__init__(l_short_rates, l_maturities)

    def _get_cupon(self, f_face_value, f_initial_rate, f_time):
        '''
        Get the cupon value to the parameters passed
        :param f_face_value: float.
        :param f_initial_rate: float.
        :param f_time: float
        '''
        f_cupon = 0.
        return f_cupon

    def fit_foward_curve(self, f_sigma, f_faceval=100., func_rate=kwf_sde_adj):
        '''
        Fit the foward curve using the short term rates
        :param f_sigma: float. Foward one-year rate volatility af all times
        :*param f_faceval. float. Face value of the hypothetical bond
        :*param func_rate: function obj. The dynamics of the short rate
        '''
        b_use_pu = True
        self.f_face_value = f_faceval
        self.l_pu = []
        for f_rate, f_mat in zip(self.l_short_rates, self.l_maturities):
            self.l_pu.append(f_faceval/(1+f_rate)**f_mat)
        self.f_sigma = f_sigma
        self.func_rate = func_rate
        for i_step in xrange(1, self.i_steps):
            f_pu = None
            if b_use_pu:
                f_pu = self.l_pu[i_step]
            self._go_backward(i_step, f_pu)
        self.already_fitted = True

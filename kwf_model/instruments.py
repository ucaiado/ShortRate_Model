#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement the precifications methods of different bonds and derivatives of
interest rate based on binomial tree

@author: ucaiado

Created on 06/22/2016
"""

# import libraries
from collections import defaultdict
import math
import matplotlib.pylab as plt
import pandas as pd
from scipy import optimize
from binomial_tree import NOT_FITTED_ERROR

'''
Begin help functions
'''


class INCORRECT_SETUP_ERROR(Exception):
    '''
    DIFFERENT_LENGTHS_ERROR is raised by the init method of the KWFTree class
    '''
    pass

'''
End help functions
'''


class Instrument(object):
    '''
    A Instrument representation
    '''
    def __init__(self, f_face_value, f_cupon):
        '''
        Initiate a Instrument object. Save all parameters as attributes
        :param f_face_value: float. The face value of the Instrument
        :param f_cupon: float. The cupon paid at each month by the instrument
        '''
        self.f_face_value = f_face_value
        self.f_cupon = f_cupon

    def _get_terminal_value(self):
        '''
        Return the terminal value to the bond. This method should be changed
        to different instrauments
        '''
        raise NotImplemented

    def _get_value_on_the_node(self, f_value, f_time):
        '''
        Return the value of the node
        '''
        raise NotImplemented

    def _get_discount_factor(self, node, node_ahead):
        '''
        Return the appropriate discount factor of the current step
        :param node: Node object. The current node
        :param node_ahead: Node object. the node in the next step
        '''
        raise NotImplemented

    def _get_appropriate_maturity(self, node):
        '''
        Return the the time of the node based on the previous values
        :param node: Node object. The current node
        '''
        raise NotImplemented

    def _get_range_of_values(self, tree_fitted, i_steps):
        '''
        Return a range of the possible values to the instrument
        :param tree_fitted: BinomialTree object. A tree already fitted
        :param i_steps: integer. the step of the matutiry of the bond
        '''
        # checa se a arvore ja tem as fowars jah setadas
        if not tree_fitted.already_fitted:
            raise NOT_FITTED_ERROR
        if i_steps <= 0 or i_steps > tree_fitted.i_steps:
            raise INCORRECT_SETUP_ERROR
        # preenche valores no ultimo passo com valor de face
        df_value = self._get_terminal_value()
        df_forwards = tree_fitted.get_fowards_dynamics()
        df_forwards += 1
        df_forwards = df_forwards.ix[:, :i_steps-1]
        for idx, row in df_forwards.T.ix[::-1, :].iterrows():
            node = tree_fitted.d_step[idx+1][0]
            f_time = self._get_appropriate_maturity(node)
            f_maturity = tree_fitted.l_maturities[node.i_step-1]
            f_this_cupon, f_val = self._get_value_on_the_node(0., f_maturity)
            df_value = (df_value + f_this_cupon)/(row**(f_time))

        return df_value

    def _get_current_value(self, tree_fitted, i_steps):
        '''
        Calculate the price to the instrument going backward in the tree passed
        :param tree_fitted: BinomialTree object. A tree already fitted
        :param i_steps: integer. the step of the matutiry of the bond
        '''
        # checa se a arvore ja tem as fowars jah setadas
        if not tree_fitted.already_fitted:
            raise NOT_FITTED_ERROR
        if i_steps <= 0 or i_steps > tree_fitted.i_steps:
            raise INCORRECT_SETUP_ERROR
        # preenche valores no ultimo passo com valor de face
        f_face_value = self._get_terminal_value()
        for node in tree_fitted.d_step[i_steps]:
            # checa quais valores de cupon e face value devem ser usados
            f_maturity = tree_fitted.l_maturities[node.i_step-1]
            f_cupon, f_val = self._get_value_on_the_node(self.f_face_value,
                                                         f_maturity)
            node.set_values_to_precify(f_cupon=f_cupon,
                                       f_value=f_val)

        # itera toda a arvore de tras para frente
        for i_aux_step in xrange(i_steps-1, -1, -1):
            for node in tree_fitted.d_step[i_aux_step]:
                # calcula valor presente pegando valores dos childrens
                s_down, s_up = node.get_childrens()
                node_down = tree_fitted[s_down]
                node_up = tree_fitted[s_up]
                f_aux = (node_down.f_value_precify + node_down.f_cupon_precify)
                f_aux *= node_down.f_prob
                f_aux2 = (node_up.f_value_precify + node_up.f_cupon_precify)
                f_aux2 *= node_up.f_prob
                f_aux += f_aux2
                f_aux /= self._get_discount_factor(node, node_down)
                # f_aux /= (1+node.f_r)**(node_down.f_time)
                f_maturity = tree_fitted.l_maturities[node.i_step-1]
                f_this_cupon, f_val = self._get_value_on_the_node(f_aux,
                                                                  f_maturity)
                node.set_values_to_precify(f_cupon=f_this_cupon,
                                           f_value=f_val)
        return float(tree_fitted['_'].f_value_precify)

    def get_range_of_values(self, tree_fitted, i_steps):
        '''
        Return a range of the possible values to the instrument. It calls the
        _get_range_of_values() method.
        :param tree_fitted: BinomialTree object. A tree already fitted
        :param i_steps: integer. the step of the matutiry of the bond
        '''
        return self._get_range_of_values(tree_fitted, i_steps)

    def get_current_value(self, tree_fitted, i_steps):
        '''
        Calculate the price to the instrument going backward in the tree passed
        It call _get_current_value() method
        :param tree_fitted: BinomialTree object. A tree already fitted
        :param i_steps: integer. the step of the matutiry of the bond
        '''
        return self._get_current_value(tree_fitted, i_steps)


class Bond(Instrument):
    '''
    A Bond representation
    '''
    def __init__(self, f_face_value, f_cupon=0):
        '''
        Initiate a Bond object. Save all parameters as attributes
        :param f_face_value: float. The face value of the Instrument
        :*param f_cupon: float. The cupon paid at each month by the instrument
        '''
        super(Bond, self).__init__(f_face_value, f_cupon)

    def _get_terminal_value(self):
        '''
        Return the terminal value to the bond. This method should be changed
        to different instrauments
        '''
        return self.f_face_value

    def _get_value_on_the_node(self, f_value, f_time):
        '''
        Return the cupon and value of the node
        '''
        return self.f_cupon, f_value

    def _get_discount_factor(self, node, node_ahead):
        '''
        Return the appropriate discount factor of the current step
        :param node: Node object. The current node
        :param node_ahead: Node object. the node in the next step
        '''
        return (1+node.f_r)**(node_ahead.f_time)

    def _get_appropriate_maturity(self, node):
        '''
        Return the the time of the node based on the previous values
        :param node: Node object. The current node
        '''
        return node.f_time


class BondBetween(Instrument):
    '''
    A Bond representation that presents a maturity between two nodes of the
    tree used to precify it. The maturiry should less or equal of the last tree
    step
    '''
    def __init__(self, f_face_value, f_maturity, f_cupon=0):
        '''
        Initiate a BondBetween object. Save all parameters as attributes
        :param f_face_value: float. The face value of the Instrument
        :param f_maturity: float. the maturiry of the bond expressed in years
        :*param f_cupon: float. The cupon paid at each month by the instrument
        '''
        # inicia objeto
        self.f_maturity = f_maturity
        super(BondBetween, self).__init__(f_face_value, f_cupon)
        # encontra step da arvore imediatamente inferior ao vencimento do bond
        self.node_step = None
        self.node_maturity = None

    def _get_terminal_value(self):
        '''
        Return the terminal value to the bond. This method should be changed
        to different instrauments
        '''
        return self.f_face_value

    def _get_value_on_the_node(self, f_value, f_time):
        '''
        Return the cupon and value of the node
        '''
        return self.f_cupon, f_value

    def _get_discount_factor(self, node, node_ahead):
        '''
        Return the appropriate discount factor of the current step
        :param node: Node object. The current node
        :param node_ahead: Node object. the node in the next step
        '''
        if node_ahead.i_step >= self.node_step:
            return (1+node.f_r)**(self.f_time)
        return (1+node.f_r)**(node_ahead.f_time)

    def _get_appropriate_maturity(self, node):
        '''
        Return the the time of the node based on the previous values
        :param node: Node object. The current node
        '''
        if node.i_step >= self.node_step:
            return self.f_time
        return node.f_time

    def get_range_of_values(self, tree_fitted):
        '''
        Return a range of the possible values to the instrument. It calls the
        _get_range_of_values() method.
        :param tree_fitted: BinomialTree object. A tree already fitted
        :param i_steps: integer. the step of the matutiry of the bond
        '''
        l_mat = [0] + tree_fitted.l_maturities
        for idx, f_aux in enumerate(l_mat):
            if f_aux >= self.f_maturity:
                break
        self.node_step = idx
        self.f_time = self.f_maturity - l_mat[idx-1]
        return self._get_range_of_values(tree_fitted, self.node_step)

    def get_current_value(self, tree_fitted):
        '''
        Calculate the price to the instrument going backward in the tree passed
        It call _get_current_value() method
        :param tree_fitted: BinomialTree object. A tree already fitted
        :param i_steps: integer. the step of the matutiry of the bond
        '''
        l_mat = [0] + tree_fitted.l_maturities
        for idx, f_aux in enumerate(l_mat):
            if f_aux >= self.f_maturity:
                break
        self.node_step = idx
        self.f_time = self.f_maturity - l_mat[idx-1]
        return self._get_current_value(tree_fitted, self.node_step)


class NTN_F(BondBetween):
    '''
    A NTN-F representation. This Bond pays semestral cupons of 10 percent per
    year and has a face value of 1,000
    '''
    def __init__(self, f_maturity, l_du_cupons):
        '''
        Initiate a NTN_F object. Save all parameters as attributes
        :param f_maturity: float. the maturiry of the bond expressed in years
        :param l_du_cupons: list. business days for the payment of each cupon
        '''
        # inicia objeto
        self.l_du_cupons = l_du_cupons
        self.f_maturity = f_maturity
        f_cupon = ((1.+0.1)**0.5-1.)*1000.
        super(NTN_F, self).__init__(1000., f_maturity, f_cupon)
        # encontra step da arvore imediatamente inferior ao vencimento do bond
        self.node_step = None
        self.node_maturity = None

    def _get_value_on_the_node(self, f_value, f_maturity):
        '''
        Return the cupon and value of the node. Pay semestral cupons
        '''
        # if abs((f_time/0.5)-int((f_time/0.5)))<1e-4:
        if f_maturity in self.l_du_cupons:
            # print f_maturity*252.
            return self.f_cupon, f_value
        return 0., f_value

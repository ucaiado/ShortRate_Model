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

    def _get_value_on_the_node(self, f_value):
        '''
        Return the value of the node
        '''
        raise NotImplemented

    def get_current_value(self, tree_fitted, i_steps):
        '''
        Calculate the price to the instrument going backward in the tree passed
        :param tree_fitted: BinomialTree object. A tree with the foward rates
            already fitted
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
            node.set_values_to_precify(f_cupon=self.f_cupon,
                                       f_value=self.f_face_value)

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
                f_aux /= (1+node.f_r)**(node_down.f_time)
                f_this_cupon, f_val = self._get_value_on_the_node(f_aux)
                node.set_values_to_precify(f_cupon=f_this_cupon,
                                           f_value=f_val)
        return float(tree_fitted['_'].f_value_precify)


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

    def _get_value_on_the_node(self, f_value):
        '''
        Return the cupon and value of the node
        '''
        return self.f_cupon, f_value

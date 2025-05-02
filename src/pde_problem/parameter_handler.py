#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:49:46 2018
@author: Niccolò Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

import numpy as np
import random
import os

import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class ParameterHandler:
    """Class to handle the parameters involved in parameter-dependent steady and unsteady problems
    """

    def __init__(self):
        """Initializing the parameter handler with default values
        """

        self.M_param_min = np.zeros(0)
        self.M_param_max = np.zeros(0)
        self.M_param = np.zeros(0)
        self.M_num_parameters = 0

        return

    def assign_parameters_bounds(self, _param_min, _param_max):
        """Method to assign the bounding values (min and max) to the parameters involved in the problem

        :param _param_min: minimum values of the parameters
        :type _param_min: numpy.ndarray
        :param _param_max: maximum values of the parameters
        :type _param_max: numpy.ndarray
        """
        self.M_param_min = _param_min
        self.M_param_max = _param_max
        self.M_param = np.zeros(_param_min.shape)
        self.M_num_parameters = _param_min.shape[0]
        return

    def assign_parameters(self, _param):
        """Method to assign the parameter value, provided that the input has the right shape

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        """
        assert self.M_num_parameters == _param.shape[0]
        self.M_param = _param
        return

    def rescale_parameters(self, _param):
        """Method to rescale a parameter from [0;1] to the proper bounding interval

        :param _param: normalized value of the parameter
        :type _param: numpy.ndarray
        :return: rescaled value of the parameter
        :rtype: numpy.ndarray
        """
        return self.M_param_min + _param * (self.M_param_max - self.M_param_min)

    def normalize_parameters(self, _param):
        """Method to rescale a parameter from the proper bounding interval to [0;1]

        :param _param: original value of the parameter
        :type _param: numpy.ndarray
        :return: normalized value of the parameter
        :rtype: numpy.ndarray
        """
        return (_param - self.M_param_min) / (self.M_param_max - self.M_param_min)

    def print_parameters(self):
        """Method to print the parameter value and its shape
        """
        print("Number of parameters : %d " % self.M_num_parameters)
        print(f"The current parameter is: {self.M_param}")
        return

    def generate_parameter(self, prob=None, seed=42):
        """Method to generate, randomly, the parameter. The random sampling is performed in [0;1], obeying to a given
        discrete probability distribution 'prob'. If such distribution is passed as None, a uniform sampling is
        performed. The parameter is lately rescaled in the proper interval.

        :param prob: discrete probability distribution used for the random parameter sampling
        :type prob: list or tuple or numpy.ndarray
        :param seed: seed for the random number generation. Defaults to 42
        :type seed: int
        """

        assert self.M_num_parameters > 0

        random.seed(seed)
        np.random.seed(seed)

        if prob is not None:
            if type(prob) is list or type(prob) is tuple:
                dim = len(prob)
            elif type(prob) is np.ndarray and len(prob.shape) == 1:
                dim = prob.shape[0]
            else:
                logger.critical("'prob' must be either a list, a tuple or a 1D numpy.ndarray")
                raise TypeError
            pool = np.arange(dim)

        for iP in range(self.M_num_parameters):
            if prob is None:
                pRandom = float(random.randint(0, 10000) / 10000)
            else:
                pRandom = float(np.random.choice(pool, p=prob) / dim)

            self.M_param[iP] = self.M_param_min[iP] + pRandom * (self.M_param_max[iP] - self.M_param_min[iP])

        return

    @property
    def param(self):
        """Getter method, to get the parameter value

        :return: parameter value
        :rtype: numpy.ndarray
        """
        return self.M_param

    @property
    def num_parameters(self):
        """Getter method, to get the number of parameters, i.e. the shape of the current parameter value

        :return: number of parameters
        :rtype: int
        """
        return self.M_num_parameters

    @property
    def param_min(self):
        """Getter method, to get the minimum values of the parameters

        :return: minimum values of the parameters
        :rtype: numpy.ndarray
        """
        return self.M_param_min

    @property
    def param_max(self):
        """Getter method, to get the maximum values of the parameters

        :return: maximum values of the parameters
        :rtype: numpy.ndarray
        """
        return self.M_param_max

    @property
    def param_range(self):
        """Getter method, to get the ranges of the parameter values

        :return: range of the parameter values
        :rtype: numpy.ndarray
        """
        return self.M_param_max - self.M_param_min


__all__ = [
    "ParameterHandler"
]



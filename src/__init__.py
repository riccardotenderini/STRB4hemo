#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The *src* module contains the implementation of the core methods of the library, i.e. the methods which allow to
interface with the chosen third party library for the FOM approximation of the parametrized PDE at hand, to handle the
affine decomposition (either real or approximated via DEIM/M-DEIM algorithms) of the problem with respect to its
characteristic parameters, to construct the Dimensionality Reduced framework or to solve the problem at hand in the c
ontext of such Dimensionality Reduced framework. All these tasks are handled by different classes, organized into
different submodules that are linked hereafter.
"""


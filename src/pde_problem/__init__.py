#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The *PDE_Problem* submodule contains the implementation of classes that allow to handle the FOM approximation of any
paramerized PDE, either steady or time-dependent. In particular one class, called *ParameterHandler* is responsible for
the handling of the characteristic parameters of the problem at hand, implementing methods to normalize/rescale the
parameter values and to generate new random parameter values within a certain range. The other two classes are the
*FomProblem* classes and configure as abstract classes that merge in a unique interface all the elements characterizing
a parametrized FOM problem, i.e. the assembler/solver methods of the *ExternalEngine* class, the parameter handling
methods of the *ParameterHandler* class and the definition of the specifics of the FOM problem (space/time
discretization parameters, type of solver...), which is compulsory to provide the desired FOM approximation of the PDE.
This task is achieved via three class attributes, which are instances of *ExternalEngine* (or, better,
*MatlabExternalEngine* in our case) and *ParameterHandler* for the first two goals; the FOM specifics are instead stored
via a dictionary class attribute. Those are anyway abstract classes because, not being related to a specific
parametrized PDE problem, they do not specify how the values of the characteristic parameters must be combined together
with the ones of the affine components, in order to build the parameter-dependent FOM-approximated operators. Such task
is ideally performed by the *define_theta_functions()* class method, which anyway simply raises a *SystemError* in this
case. All the implemented PDE problems configure as classes inheriting from *FomProblem* classes and "concretizing"
these via the implementation of a *define_theta_functions()* method, able to compute the coefficients of the affine
decompositions of all the problem operators from the values of the characteristic parameters. Finally, the *FomProblem*
classes are two, since there is an "original" one, called *FOMProblem*, that relates to steady problems, and a second
one, called *FOMProblemUnsteady* and inheriting from the first one, that instead manages time-dependent parametrized
PDEs.
"""


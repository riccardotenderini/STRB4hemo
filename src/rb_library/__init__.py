#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The *RB_Library* submodule configures as the core submodule of the *src* module of `PyORB` library; indeed it
contains the implementation of the classes which ultimately allow to handle the PDE problem at hand in a dimensionality
reduced framework, both during the offline phase (i.e. snapshots computation and Reduced Space construction) and the
online phase (i.e. solution of the problem for a new parameter value, taking advantage of its affine decomposability).
In particular, this submodule contains the implementation of two main categories of classes:


    * **AffineDecompositionHandler classes**: these classes are responsible for handling the affine decomposition
      of the PDE problem at hand, both at the FOM approximation level and at the Dimensionality Reduced one. At FOM
      level, this is simply achieved by interfacing with the *ExternalEngine* method that allows to build the affine
      components for each operator involved in the considered PDE problem, which is in turn possible via the wrapper
      method *retrieve_fom_affine_components* of the *FOMProblem* classes. At dimensionality reduced level, instead, the
      construction of the affine components may be more complex; for the standard RB method (used for both the steady
      problems and the unsteady ones handled via multistep methods) it consists simply in projecting the FOM affine
      components onto the Reduced Space, via a Reduced Basis in space to be provided as input argument. For the ST-LSPG
      projection approach on unsteady problems, instead, the expression of the reduced affine components is different,
      being them reduced both along the space and the time dimension; thus, their computation requires the
      implementation of slightly more sofisticated algorithms. Additionally, the *AffineDecompositionHandler* classes
      feature methods both to import the desired FOM/RB affine components from files and to save the computed FOM/RB
      affine components to files. Such classes are then organized into a hierarchy. The original base class is called
      *AffineDecompositionHandler* and handles only steady parametrized problems, thus for instance not considering the
      presence of the mass matrix operator, being it never involved in the steady PDE problems implemented in the
      project. The derived *AffineDecompositionHandlerUnsteady* class, instead, manages the affine components for
      generic parametrized time-dependent problems, thus involving also the mass matrix in the list of considered
      operators and managing the fact that the right-hand side forcing term may vary along time. Finally, the
      *AffineDecompositionHandlerSpaceTime* and *AffineDecompositionHandlerMultistep* classes inherit from
      *AffineDecompositionHandlerUnsteady* and manage the reduced affine components arising while solving time-dependent
      parametrized problems with the ST-LSPG projection approach and with linear multistep methods respectively. As a
      remark, it is worth recalling that in the context of this project we have taken into account only the thermal
      block problem, which is known to be affinely parametrized with respect to its characteristic parameters; anyway
      `PyORB` allows to handle also non-affinely parametrized problems, by constructing an affinely parametrized
      approximation of those via DEIM/M-DEIM algorithms. Such passage is handled by the *DEIM* and *MDEIM* classes,
      implemented in the *RB_Library* submodule; anyway, not being involved in the current project, they won't be
      described any further.
    * **RbManager classes**: these classes are the ultimate responsible for the efficient resolution of parametrized PDE
      problems, providing the implementation of all the fundamental methods related to the construction and testing of
      a Dimensionality Reduced framework. This is achieved by storing as class attributes both an instance of a
      *FOMProblem* class, to manage the original FOM approximation of the problem at hand, and an instance of a
      *AffineDecompositionHandler* class, which instead allows to deal with its affine decomposability with respect to
      the characteristic parameters. Specifically, for the offline phase they allow to either compute or import from
      files the FOM snapshots (obtained via the chosen *ExternalEngine*, by computing the FOM solution to the PDE
      problem at hand for multiple values of its characteristic parameters), the Reduced Basis (in both space and time
      for the ST-LSPG approach) and the projection of the affine components onto the dimensionality reduced space (via a
      class attribute being an instance of one of the *AffineDecompositionHandler* classes). Regarding the online phase,
      instead, those of the \emph{RbManager} classes which configure as "concrete" feature one method, called
      *solve_reduced_problem*, that, given a parameter value, assembles the reduced problem (taking advantage of its
      affine decomposability) according to the implemented solver, and computes its solution. Notice that some of the
      classes are abstract, since their *solve_reduced_problem* method raises a *SystemError* if invoked; specifically,
      this applies to those "general" classes that are not linked to a specific solver method, i.e. *RbManagerUnsteady*
      and *RbManagerMultistep*. Moreover, the *RbManager* classes contain methods that allow to assess the performances
      of the implemented Reduced Solvers, by comparing the solutions computed with such solvers with the ones obtained
      directly from the FOM problem, both in terms of errors (in usual vectorial norms) and execution times. Also the
      *RbManager* classes family is orgainized into a hierarchy, which resembles the one of the
      *AffineDecompositionHandler* classes. The base parent class is \emph{RbManager}, which handles the RB
      approximation of steady parametrized problems and for this features class attributes being instances of
      *AffineDecompositionHandler* and *FomProblem*. The *RbManagerUnsteady* class is an abstract class (i.e. it does
      not have a "valid" solver method) that inherits from *RbManager* and features all the methods common to the
      Dimensionality Reduction approaches for unsteady parametrized PDE problems that have been implemented, like the
      computation and storage of time-dependent snapshots or the handling of forcing terms that vary along time. To
      achieve this, such class features as attributes instances of \emph{AffineDecompositionHandlerUnsteady} and
      *FomProblemUnsteady*. These first two classes are the only ones in the whole *RbManager* family that feature
      testing methods to assess the performances of the implemented reduced solvers; indeed, all methods for unsteady
      parametrized PDEs are tested via a unique *test_rb_solver()* method, implemented in the *RbManagerUnsteady*
      class. Two classes derive then from *RbManagerUnsteady*. On the one side there is *RbManagerSpaceTime*, which
      handles the resolution of time-dependent PDE problems using a ST-LSPG projection approach, thus gaining
      dimensionality reduction in both space and time; such class features a class attribute being an instance of
      *AffineDecompositionHandlerSpaceTime*. On the other side, instead, there is *RbManagerMultistep*, which
      configures anyway as an abstract class, not being specifically referred to any implemented multistep solver;
      instead, it implements methods common to all Reduced multistep solvers, featuring an instance of
      *AffineDecompositionHandlerMultistep* as class attribute. Finally, the three classes *RbManagerMultistepTheta*,
      *RBManagerMultistepAM* and *RbManagerMultistepBDF* all inherit from *RbManagerMultistep* and "concretize" such
      class, by implementing a valid solver method, that makes use of the one-step Theta-method, the Adams-Moulton
      method and the Backward Differentiation Formulas method respectively.

In addition to these two big families of classes, the *RB_Library submodule of `PyORB` also contains the implementation
of the *ProperOrthogonalDecomposition* class; such class is actually a callable object (i.e. it is equipped with a
*__call__()* method) that, given as input a tensor and a threshold, it computes the Proper Orthogonal Decomposition
(POD) of the tensor with the chosen threshold. An instance of such class is stored as class attribute in the *RbManager*
class and, thus, in any class belonging to the *RbManager* hierarchy.
"""

  
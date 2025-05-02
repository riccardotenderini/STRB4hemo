#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:08:34 2021
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os

import src.rb_library.rb_manager.space_time.rb_manager_space_time as rbmst

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSTRB(rbmst.RbManagerSpaceTime):
    """Class which handles the assembling and the resolution of unsteady parametrized PDE problems, performing
   dimensionality reduction in both space and time dimension via the employment of the ST-RB approach.
   The Space-Time Reduced Basis is  constructed as the outer product of a space basis and a time basis and it is built
   by performing two PODs (one over the space dimension and the other one over the time dimension) on the tensors
   assembled by computing different FOM solutions, for different values of the characteristic parameters. Such solutions
   are furthermore computed interfacing to another third-party library, which could be either the 'feamat' Matlab
   library (https:\\github.com\lucapegolotti\feamat\tree\pyorb_wrappers_parabolic) or the 'LifeV' C++ library
   (https:\\bitbucket.org\lifev-dev\lifev-release\wiki\Home) in the context of this project.
   It inherits from :class:`~rb_manager_unsteady.RbManagerSpaceTime`
   """

    def __init__(self, _fom_problem, _affine_decomposition):
        """Initialization of the RbManagerSpaceTime class

        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        :param _affine_decomposition: AffineDecompositionHandlerSpaceTime object, used to handle the affine
           decomposition of the unsteady parametric FOM problem at hand with respect to the characteristic parameters,
           according to the ST-LSPG projection approach. If None, it is not initialized. Defaults to None.
        :type _affine_decomposition: AffineDecompositionHandlerSpaceTime or NoneType
        """

        super().__init__(_fom_problem, _affine_decomposition)

        self.reduction_method = "ST-RB"

        return

    def build_rb_affine_decompositions(self, operators=None):
        """Method which constructs the RB affine components for the operators passed in input, which can be either
        'Mat', for the affine components of the left-hand side matrix, or 'f', for the ones of the right-hand side
        vector, arising from the final residual minimization according to the ST-LSPG projection approach.
        If no operator is passed, the RB affine arrays are constructed both the operators. If the matrices M*phi_i and
        A_q*phi_i have not been constructed, they are assembled here, since they are needed by the
        ``build_rb_affine_decompositions`` method of
        :class:`~affine_decomposition_space_time.AffineDecompositionHandlerSpaceTime` class. The computed RB affine
        components are stored in the class attribute which is an instance of
        :class:`~affine_decomposition_space_time.AffineDecompositionHandlerSpaceTime` class.

        :param operators: operators for which the RB affine components have to be constructed. Admissible values are
          'Mat' for the left-hand side matrix and 'f' for the right-hand side vector.
        :type operators: set{str} or NoneType
        """

        if operators is None:
            operators = {'Mat', 'f'}

        self.M_affine_decomposition.build_rb_affine_decompositions(self.M_basis_space, self.M_basis_time,
                                                                   self.M_fom_problem,
                                                                   operators=operators)
        return

    def build_rb_approximation(self, _ns, _tol_space=1e-4, _tol_time=1e-4, prob=None):
        """Method which allows to build the dimensionality reduced, according to the ST-RB approach, the
        left-hand side matrix and the right-hand side vector, characterizing the final linear system to be solved.
        In particular, the method is basically divided into several steps and at each step it is possible either to
        import the desired quantities from ``.txt`` files (provided that a valid path to such files is given in input)
        or to construct the same quantities from scratch.
        Such steps are:

            * importing or construction of the snapshots and of the corresponding parameters. If the number of the
              desired snapshots exceeds the number of the already stored ones, additional snapshots are anyway computed.
              The number of desired snapshots is specified in input.
            * importing or construction of the Reduced Basis in both Space and Time
            * importing or construction of the dimensionality reduced affine components fro the left-hand side matrix
              and the right-hand side vector of the final linear system to be solved

        Notice that, while the importing of the SpaceTime basis and of the dimensionality reduced affine components is
        compulsory for the online execution of the method, the one of the snapshots it is not and it is also memory
        demanding; because of this, the method supports the possibility of importing all the quantities but the
        snapshots and proceeds to the building of new snapshots only if some of the other quantities have not been
        successfully imported. Finally, if a quantity is constructed from scratch and the class attribute flag
        'self.M_save_offline_structures' is set to True, then such quantity is saved at a ``.txt`` file; the saving path
        is known to the class if the class method :func:`~rb_manager_space_time.RbManagerSpaceTime.save_offline_structures`
        has been previously called via a proper input argument.

        :param _ns: number of "standard" snapshots
        :type _ns: int
        :param _tol_space: tolerance for the relative energy of the already scanned singular values with respect to the
          energy of the full set of singular values, used for the POD in space on the "standard" snapshots tensor. Energy
          is interpreted as the squared l2-norm of the singular values. The actual tolerance used in the algorithm is
          'tolerance = 1.0 - _tol', thus '_tol' is intended to be close to 0. Defaults to 1e-4
        :type _tol_space: float
        :param _tol_time: tolerance for the relative energy of the already scanned singular values with respect to the
          energy of the full set of singular values, used for the POD in time on the "standard" snapshots tensor. Energy
          is interpreted as the squared l2-norm of the singular values. The actual tolerance used in the algorithm is
          'tolerance = 1.0 - _tol', thus '_tol' is intended to be close to 0. Defaults to 1e-4
        :type _tol_time: float
        :param prob: discrete probability distribution used for the random parameter sampling
        :type prob: list or tuple or numpy.ndarray
        """

        self.reset_reduced_structures()
        self.reset_rb_approximation()

        logger.info(f"Building ST-RB approximation with {_ns} snapshots and tolerances of {_tol_space:.0e} in space and "
                    f"of {_tol_time:.0e} in time")

        if self.M_import_snapshots:
            logger.info('Importing stored snapshots')
            import_success_snapshots = self.import_snapshots_matrix(_ns)
        else:
            import_success_snapshots = False

        if self.M_import_snapshots:
            logger.info('Importing stored parameters')
            import_success_parameters = self.import_snapshots_parameters(_ns)
        else:
            import_success_parameters = False

        if self.M_import_offline_structures:
            logger.info('Importing space and time basis matrices\n')
            import_success_basis = self.import_snapshots_basis()
        else:
            import_success_basis = False
            
        if import_success_snapshots and self.M_ns < _ns:
            logger.warning(f"We miss some snapshots! I have only {self.M_ns} in memory and "
                           f"I would need to compute {_ns - self.M_ns} more!")

        if (not import_success_snapshots or not import_success_parameters) and not import_success_basis:
            if _ns - self.M_ns > 0:
                logger.info("Snapshots importing has failed! We need to construct them from scratch and to "
                            "assemble all the offline structures!")
                self.build_snapshots(_ns - self.M_ns, prob=prob)
        elif import_success_parameters and import_success_snapshots and self.M_ns < _ns:
            logger.info(f"We miss some snapshots! I have only {self.M_ns} in memory and "
                        f"I need to compute {_ns - self.M_ns} more and to assemble all the offline structures.")
            self.build_snapshots(_ns - self.M_ns, prob=prob)
            import_success_basis = False  # I have to recompute the basis if I build more snapshots

        if not import_success_basis:
            logger.info("Basis importing failed. We need to construct it via POD")
            self.build_ST_basis(_tol_space=_tol_space, _tol_time=_tol_time)
            import_success_basis = True

        logger.info('Building RB affine decomposition')
        if self.M_import_offline_structures and import_success_basis:
            logger.info('Importing RB affine components')
            import_failures_set = self.import_rb_affine_components()
            if import_failures_set:
                logger.info('Building reduced structures whose import has failed')
                self.build_rb_affine_decompositions(operators=import_failures_set)

            logger.info("Importing the generalized coordinates")
            import_success_gen_coords = self.import_generalized_coordinates()
            if not import_success_gen_coords:
                logger.info("Building the generalized coordinates")
                self.get_generalized_coordinates()

        else:
            logger.info('Building all RB affine components')
            self.build_rb_affine_decompositions(operators={'Mat', 'f', 'u0'})

            logger.info("Building the generalized coordinates")
            self.get_generalized_coordinates()

        if self.M_save_offline_structures:
            self.save_rb_affine_decomposition()

        return

    def build_reduced_problem(self, _param):
        """Method which allows to assemble the final linear system that, according to the ST-RB approach,
        allows to compute the desired solution in efficient way; this is achieved by exploiting dimensionality
        reduction in both Space and Time. In particular this method, given a parameter value, assembles the
        dimensionality reduced left-hand side matrix and right-hand side vector, by linearly combining the
        parameter-dependent theta functions with the dimensionality reduced affine components.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        """

        self.M_An = np.zeros((self.M_N, self.M_N))
        self.M_fn = np.zeros(self.M_N)

        dt = self.dt

        theta_a = self.M_fom_problem.get_full_theta_a(_param)
        for iQa in range(self.M_used_Qa):
            self.M_An += theta_a[iQa] * dt * self.get_rb_affine_matrix(iQa)

        for iQm in range(2 * self.M_used_Qm):
            self.M_An += self.get_rb_affine_matrix(self.M_used_Qa + iQm)

        theta_f = self.M_fom_problem.get_full_theta_f(_param)
        for iQf in range(self.M_used_Qf):
            self.M_fn += theta_f[iQf] * dt * self.get_rb_affine_vector(iQf)

        self.M_fn += self.get_rb_initial_condition()

        return

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The *TPL_Managers* (Third-Party Libraries Managers) submodule is responsible for the interaction with the third-party
libraries (thus either *feamat* for 2D problems or *LifeV* for 3D problems) in order to handle the FOM approximation of
the PDE at hand.

.. note:: In the context of this project, we have considered only the 2D thermal block problem (both steady and
  unsteady), thus we have only made use of the *feamat* *Matlab* library; because of this, no description of the
  external engine responsible for the interface with the C++ *LifeV* library (which is anyway implemented in the repo)
  is provided.

"""


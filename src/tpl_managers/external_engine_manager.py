#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:59:45 2018
@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

from src.tpl_managers import matlab_external_engine as matlab_ext
# from src.tpl_managers import cpp_external_engine as cpp_ext


class ExternalEngineManager:
    """ Class that manages the initialization, starting and quitting of the external engine that has been selected for
    the resolution of the FOM problems
    """

    def __init__(self, _engine_type, _library_path):
        """ Initialization of the ExternalEngineManager class. If the engine type is not in the recognized ones, it
        raises an Exception.

        :param _engine_type: identificative string of the external engine (either 'matlab' or 'cpp' in this project)
        :type _engine_type: str
        :param _library_path: path to the library used to solve the FOM problems
        :type _library_path: str
        """

        self.M_engine_type = _engine_type
        self.M_library_path = _library_path

        if _engine_type == 'matlab':
            self.M_external_engine = matlab_ext.MatlabExternalEngine(_library_path)
        # elif _engine_type == 'cpp':
        #     self.M_external_engine = cpp_ext.CppExternalEngine(_library_path)
        else:
            self.M_external_engine = None
            raise Exception(f"Unrecognized external engine type {_engine_type}")
        return

    def get_external_engine(self):
        """ Getter method which returns the external engine

        :return: the external engine
        :rtype: ExternalEngine
        """

        return self.M_external_engine

    def start_engine(self):
        """ Method which starts the selected external engine
        """

        self.M_external_engine.start_engine()
        return

    def quit_engine(self):
        """ Method which quits the selected external engine
        """

        self.M_external_engine.quit_engine()
        return


__all__ = [
    "ExternalEngineManager"
]

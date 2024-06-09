#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
"""

from ctypes import c_double, c_int, c_bool, c_void_p, Structure, CDLL

# loading the library
try:
    LIBSPOT = CDLL('./libspot.so')
except OSError as e:
    print(e)
    print("The library libspot seems missing... To get it see https://asiffer.github.io/libspot/download/")

# Some basic C structure needed for the interface


class SpotConfig(Structure):
    """
    SpotConfig (C structure, inherit ctypes.Structure)
    This structure gathers the configuration of a Spot object (aims to build
    similar instance)
    """
    _fields_ = [('q', c_double),
                ('bounded', c_bool),
                ('max_excess', c_int),
                ('alert', c_bool),
                ('up', c_bool),
                ('down', c_bool),
                ('n_init', c_int),
                ('level', c_double)]
    _fmt_ = """
            ---- SpotConfig ----
            --------------------
                     q  {:8.2e}
               bounded  {:8}
            max_excess  {:8d}
                 alert  {:8}
                    up  {:8}
                  down  {:8}
                n_init  {:8d}
                 level  {:8.3f}"""

    def __repr__(self):
        out = ''
        header = '{0} {1} {0}\n'.format("----", "SpotConfig")
        width = len(header) - 1
        width_2 = width // 2
        pattern = '{:-^' + str(width) + '}\n'
        out += header
        out += pattern.format("")
        base_pattern = '{:>' + str(width_2) + '}'
        for name, ctype in self._fields_:
            if ctype is c_double:
                pattern = base_pattern + "  {:.4f}\n"
            elif ctype is c_int:
                pattern = base_pattern + "  {:d}\n"
            else:
                pattern = base_pattern + "  {}\n"
            out += pattern.format(name, self.__getattribute__(name))
        return out

    def to_dict(self):
        return {'q': self.q,
                'bounded': self.bounded,
                'max_excess': self.max_excess,
                'alert': self.alert,
                'up': self.up,
                'down': self.down,
                'n_init': self.n_init,
                'level': self.level}

    def __iter__(self):
        yield 'q', self.q
        yield 'bounded', self.bounded
        yield 'max_excess', self.max_excess
        yield 'alert', self.alert
        yield 'up', self.up
        yield 'down', self.down
        yield 'n_init', self.n_init
        yield 'level', self.level


class SpotStatus(Structure):
    """
    SpotStatus (C structure, inherit ctypes.Structure)
    This structure summarizes the status of the Spot object around the decision
    area. It gives the number of excesses, the number of alarms, the number of
    normal observations, the values of the thresholds etc.
    """
    _fields_ = [('n', c_int),
                ('ex_up', c_int),
                ('ex_down', c_int),
                ('Nt_up', c_int),
                ('Nt_down', c_int),
                ('al_up', c_int),
                ('al_down', c_int),
                ('t_up', c_double),
                ('t_down', c_double),
                ('z_up', c_double),
                ('z_down', c_double)]

    def __repr__(self):
        out = ''
        header = '{0} {1} {0}\n'.format("----", "SpotStatus")
        width = len(header) - 1
        width_2 = width // 2
        pattern = '{:-^' + str(width) + '}\n'
        out += header
        out += pattern.format("")
        base_pattern = '{:>' + str(width_2) + '}'
        for name, ctype in self._fields_:
            if ctype is c_double:
                pattern = base_pattern + "  {:.4f}\n"
            elif ctype is c_int:
                pattern = base_pattern + "  {:d}\n"
            else:
                pattern = base_pattern + "  {}\n"
            out += pattern.format(name, self.__getattribute__(name))
        return out

    def to_dict(self):
        return {'n': self.n,
                'ex_up': self.ex_up,
                'ex_down': self.ex_down,
                'Nt_up': self.Nt_up,
                'Nt_down': self.Nt_down,
                'al_up': self.al_up,
                'al_down': self.al_down,
                't_up': self.t_up,
                't_down': self.t_down,
                'z_up': self.z_up,
                'z_down': self.z_down}

    def __iter__(self):
        yield 'n', self.n,
        yield 'ex_up', self.ex_up
        yield 'ex_down', self.ex_down
        yield 'Nt_up', self.Nt_up
        yield 'Nt_down', self.Nt_down
        yield 'al_up', self.al_up
        yield 'al_down', self.al_down
        yield 't_up', self.t_up
        yield 't_down', self.t_down
        yield 'z_up', self.z_up
        yield 'z_down', self.z_down


# Define the input/output of the C functions (interface.cpp)
# Spot object creation
LIBSPOT.Spot_new.argtypes = [c_double, c_int, c_double, c_bool, c_bool,
                             c_bool, c_bool, c_int]
LIBSPOT.Spot_new.restype = c_void_p

# Spot destructor
LIBSPOT.Spot_delete.argtypes = [c_void_p]

# Spot step method
LIBSPOT.Spot_step.argtypes = [c_void_p, c_double]
LIBSPOT.Spot_step.restype = c_int

# Spot getters
LIBSPOT.Spot_getUpperThreshold.restype = c_double
LIBSPOT.Spot_getLowerThreshold.restype = c_double
LIBSPOT.Spot_getUpper_t.restype = c_double
LIBSPOT.Spot_getLower_t.restype = c_double
LIBSPOT.Spot_up_probability.argtypes = [c_void_p, c_double]
LIBSPOT.Spot_up_probability.restype = c_double
LIBSPOT.Spot_down_probability.argtypes = [c_void_p, c_double]
LIBSPOT.Spot_down_probability.restype = c_double

# Spot status export
LIBSPOT.Spot_status.argtypes = [c_void_p]
LIBSPOT.Spot_status.restype = SpotStatus

# Spot config export
LIBSPOT.Spot_config.argtypes = [c_void_p]
LIBSPOT.Spot_config.restype = SpotConfig

# Spot setter
LIBSPOT.Spot_set_q.argtypes = [c_void_p, c_double]


# DSPOT (deprecated)
# LIBSPOT.DSpot_new.argtypes = [c_int, c_double, c_int, c_double, c_bool, c_bool, c_bool, c_bool, c_int]
# LIBSPOT.DSpot_new.restype = c_void_p
LIBSPOT.DSpot_new_light.argtypes = [
    c_int,
    c_int,
    c_double,
    c_bool,
    c_bool,
    c_bool,
    c_bool,
    c_int]
LIBSPOT.DSpot_new_light.restype = c_void_p
LIBSPOT.DSpot_set_q.argtypes = [c_void_p, c_double]

# DSpot destructor
LIBSPOT.DSpot_delete.argtypes = [c_void_p]

# DSpot step method
LIBSPOT.DSpot_step.argtypes = [c_void_p, c_double]
LIBSPOT.DSpot_step.restype = c_int

# DSpot getters
LIBSPOT.DSpot_getUpperThreshold.restype = c_double
LIBSPOT.DSpot_getLowerThreshold.restype = c_double
LIBSPOT.DSpot_getUpper_t.restype = c_double
LIBSPOT.DSpot_getLower_t.restype = c_double
LIBSPOT.DSpot_getDrift.restype = c_double

# DSpot status export
LIBSPOT.DSpot_status.argtypes = [c_void_p]
LIBSPOT.DSpot_status.restype = SpotStatus


# Spot object creation
class Spot(object):
    """Embed the Spot algorithm to flag outliers in streaming data

    This object computes probabilistic upper and lower bounds (as desired)
    according to the incoming data and the main parameter q

    Notes
    -----
    The following attributes are members of the underlying C++ class. They are
    not all directly accessible through this Python class. You can use the
    ``status`` or the ``config`` methods to access it.

    Attributes
    ----------

    q : float
        The main parameter ( P(X>z_q) < q ) : probability of abnormal events
        (between 0 and 1, but close to 0)
    n_init : int
        Number of initial observations to perform calibration
    level : float
        Quantile level to discriminate the excesses from the normal data
        (between 0 and 1, but close to 1)
    up : bool
        Compute upper threshold
    down : bool
        Compute lower threshold
    alert : bool
        If true, the algorithm triggers alarms (the outlier is not taking into
        account in the model)
    bounded : bool
        Bounded mode. If true, the number of stored will be bounded by
        max_excess
    max_excess : int
        Maximum number of storable excesses (for bounded mode)

    n : int
        Number of normal observations (not the alarms)
    Nt_up : int
        Number of up excesses
    Nt_down : int
        Number of down excesses
    al_up : int
        Number of up alarms
    al_down : int
        Number of down alarms
    init_batch : vector<double> (C++)
        Initial batch (for calibration)
    t_up : float
        Transitional up threshold
    t_down : float
        Transitional down threshold
    z_up : float
        Up alert thresholds
    z_down : float
        Down alert thresholds
    upper_bound : GPDfit (C++ class)
        Object to compute the upper threshold
    lower_bound : GPDfit (C++ class)
        Object to compute the lower threshold

    """

    def __init__(self, q=1e-4, n_init=2000, level=0.99, up=True, down=True,
                 alert=True, bounded=True, max_excess=200):
        """
        Full parametrizable constructor

        Parameters
        ----------
        q : float
            Probability of abnormal events (between 0 and 1, but close to 1)
        n_init : int
            Number of initial observations to perform calibration
        level : float
            Quantile level to discriminate the excesses from the normal data
            (between 0 and 1, but close to 1)
        up : bool
            Compute upper threshold
        down : bool
            Compute lower threshold
        alert : bool
            Trigger alert
        bounded : bool
            Bounded mode
        max_excess : int
            Maximum number of storable excesses (for bounded mode)

        Returns
        -------
        Spot object

        Example
        -------
        >>> S = Spot(q = 1e-4, n_init = 1000, level = 0.98)

        """
        self.spot_ptr = LIBSPOT.Spot_new(
            q, n_init, level, up, down, alert, bounded, max_excess)

    def step(self, data):
        """Spot iteration

        Parameters
        ----------
        data : float
            input data

        Returns
        -------
        The nature of the input data
            0 Normal data
            1 Up alert
            -1 Down alert
            2 Up excess
            -2 Down excess
            3 to initial batch
            4 calibration step

        Example
        -------
        >>> import numpy as np
        >>> S = Spot(q = 1e-4, n_init = 1000, level = 0.98)
        >>> for x in np.random.normal(0,1,100000):
                S.step(x)
        """
        return LIBSPOT.Spot_step(self.spot_ptr, data)

    def status(self):
        """Get the internal state of the Spot instance
        """
        return LIBSPOT.Spot_status(self.spot_ptr)

    def config(self):
        """Initial config of the algorithm
        """
        return LIBSPOT.Spot_config(self.spot_ptr)

    def get_upper_threshold(self):
        """Get the current upper threshold
        """
        return LIBSPOT.Spot_getUpperThreshold(self.spot_ptr)

    def get_lower_threshold(self):
        """Get the current lower threshold
        """
        return LIBSPOT.Spot_getLowerThreshold(self.spot_ptr)

    def get_upper_t(self):
        """Get the upper excess quantile
        """
        return LIBSPOT.Spot_getUpper_t(self.spot_ptr)

    def get_lower_t(self):
        """Get the lower excess quantile
        """
        return LIBSPOT.Spot_getLower_t(self.spot_ptr)

    def up_probability(self, z):
        """Give the probability to observe things higher than a value
        """
        return LIBSPOT.Spot_up_probability(self.spot_ptr, z)

    def down_probability(self, z):
        """Give the probability to observe things lower than a value
        """
        return LIBSPOT.Spot_down_probability(self.spot_ptr, z)

    def __del__(self):
        """Call C destructor"""
        LIBSPOT.Spot_delete(self.spot_ptr)


# Spot object creation
class DSpot(object):
    """Embed the Spot algorithm to flag outliers in streaming data

    This object computes probabilistic upper and lower bounds (as desired)
    according to the incoming data and the main parameter q

    Notes
    -----
    The following attributes are members of the underlying C++ class. They are
    not all directly accessible through this Python class. You can use the
    ``status`` or the ``config`` methods to access it.

    Attributes
    ----------
    d : int
        Depth of the moving average
    q : float
        The main parameter ( P(X>z_q) < q ) : probability of abnormal events
        (between 0 and 1, but close to 1)
    n_init : int
        Number of initial observations to perform calibration
    level : float
        Quantile level to discriminate the excesses from the normal data
        (between 0 and 1, but close to 1)
    up : bool
        Compute upper threshold
    down : bool
        Compute lower threshold
    alert : bool
        If true, the algorithm triggers alarms (the outlier is not taking into
        account in the model)
    bounded : bool
        Bounded mode. If true, the number of stored will be bounded by
        max_excess
    max_excess : int
        Maximum number of storable excesses (for bounded mode)

    n : int
        Number of normal observations (not the alarms)
    Nt_up : int
        Number of up excesses
    Nt_down : int
        Number of down excesses
    al_up : int
        Number of up alarms
    al_down : int
        Number of down alarms
        init_batch : vector<double> (C++)
        Initial batch (for calibration)
    t_up : float
        Transitional up threshold
    t_down : float
        Transitional down threshold
    z_up : float
        Up alert thresholds
    z_down : float
        Down alert thresholds
    upper_bound : GPDfit (C++ class)
        Object to compute the upper threshold
    lower_bound : GPDfit (C++ class)
        Object to compute the lower threshold

    """

    def __init__(
            self,
            d=10,
            q=1e-4,
            n_init=2000,
            level=0.99,
            up=True,
            down=True,
            alert=True,
            bounded=True,
            max_excess=200):
        """
        Full parametrizable constructor

        Parameters
        ----------
        d : int
            Depth of the moving average
        q : float
            Probability of abnormal events (between 0 and 1, but close to 1)
        n_init : int
            Number of initial observations to perform calibration
        level : float
            Quantile level to discriminate the excesses from the normal data
            (between 0 and 1, but close to 1)
        up : bool
            Compute upper threshold
        down : bool
            Compute lower threshold
        alert : bool
            Trigger alert
        bounded : bool
            Bounded mode
        max_excess : int
            Maximum number of storable excesses (for bounded mode)

        Returns
        -------
        Spot object

        Example
        -------
        >>> S = DSpot(d = 50, q = 1e-4, n_init = 1000, level = 0.98)

        """
        # self.dspot_ptr = LIBSPOT.DSpot_new(
        #     d, q, n_init, level, up, down, alert, bounded, max_excess)
        self.dspot_ptr = LIBSPOT.DSpot_new_light(
            d, n_init, level, up, down, alert, bounded, max_excess)
        LIBSPOT.DSpot_set_q(self.dspot_ptr, q)

    def step(self, data):
        """Spot iteration

        Parameters
        ----------
        data : float
            input data

        Returns
        -------
        The nature of the input data
            0 Normal data
            1 Up alert
            -1 Down alert
            2 Up excess
            -2 Down excess
            3 to initial batch
            4 calibration step

        Example
        -------
        # >>> import numpy as np
        # >>> S = DSpot(d = 50, q = 1e-4, n_init = 1000, level = 0.98)
        # >>> for x in np.random.normal(0,1,100000):
                S.step(x)
        """
        return LIBSPOT.DSpot_step(self.dspot_ptr, data)

    def status(self):
        """Get the internal state of the Spot instance
        """
        return LIBSPOT.DSpot_status(self.dspot_ptr)

    def config(self):
        """Initial config of the algorithm
        """
        return LIBSPOT.Spot_config(self.dspot_ptr)

    def get_upper_threshold(self):
        """Get the current upper threshold
        """
        return LIBSPOT.DSpot_getUpperThreshold(self.dspot_ptr)

    def get_lower_threshold(self):
        """Get the current lower threshold
        """
        return LIBSPOT.DSpot_getLowerThreshold(self.dspot_ptr)

    def get_upper_t(self):
        """Get the upper excess quantile
        """
        return LIBSPOT.DSpot_getUpper_t(self.dspot_ptr)

    def get_lower_t(self):
        """Get the lower excess quantile
        """
        return LIBSPOT.DSpot_getLower_t(self.dspot_ptr)

    def get_drift(self):
        """Get the current drift
        """
        return LIBSPOT.DSpot_getDrift(self.dspot_ptr)

    def __del__(self):
        """Call C destructor"""
        LIBSPOT.DSpot_delete(self.dspot_ptr)

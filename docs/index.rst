.. Muldoon documentation master file, created by
   sphinx-quickstart on Thu Aug 19 10:14:16 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Muldoon's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Changelog:
++++++++++

**1.3.5 (2022 Feb 24)**

*  Fixed bug - pressure_timeseries_injection_recovery used wrong object

**1.3.4 (2022 Feb 24)**

*  Fixed bug - detrend kept adding data_trend over and over again

**1.3.3 (2022 Feb 16)**

*  Moved popts and uncs keywords to super class
*  Moved retrieve_vortices to super class

**1.3.2 (2022 Feb 16)**

*  Fixed bug - popts/uncs were added in the wrong place

**1.3.1 (2022 Feb 16)**

*  Added popts/uncs to PressureTimeseries as optional keywords

**1.3.0 (2022 Jan 26)**

*  Allow read_Perseverance_ATS_data to return all ATS time series

**1.2.2 (2022 Jan 10)**

* Added the ability to read LMST times as well as LTST times to read_data 

**1.2.1 (2022 Jan 10)**

* Fixed bug in make_seconds_since_midnight

**1.2.0 (2022 Jan 8)**

* Added the beginnings of a TemperatureTimeseries class to met_timeseries

**1.1.1 (2022 Jan 7)**

* Fixed utils.read_data to incorporate sub-second sampling

**1.1.0 (2022 Jan 7)**

* Changed "read_Perseverance_MEDA_data" to "read_Perseverance_PS_data"
* Added read_Perseverance_ATS_data
* Added an example ATS data file

**1.0.1 (2022 Jan 6)**

* Fixed bug in requirements.txt

**1.0.0 (2022 Jan 6)**

* Modified MetTimeseries to move all the functionality specific to pressure
  time-series to a new class PressureTimeseries - This change clobbers
  backwards functionality!

**0.5.3 (2021 Aug 30)**

* Fixed bug in read_data

**0.4.3 (2021 Aug 30)**

* Made helper functions in read_data into externally accessible functions

**0.4.2 (2021 Aug 28)**

* Fixed axis label on make_conditioned_data_plot

**0.4.1 (2021 Aug 27)**

* Fixed units on Gamma in injection_recovery

**0.4.0 (2021 Aug 27)**

* Added some parameters to met_timeseries
* Added injection/recovery test to utils

**0.3.6 (2021 Aug 27)**

* Removed bug which allowed pressure_trend to include None

**0.3.5 (2021 Aug 25)**

* Extended maximum allowed Gamma in _determine_bounds

**0.3.4 (2021 Aug 24)**

* Fixed _which_primary_sol

**0.3.3 (2021 Aug 24)**

* Fixed setup.py

**0.3.2 (2021 Aug 24)**

* Fixed setup.py

**0.3.1 (2021 Aug 24)**

* Added statsmodels to required package list

**0.3.0 (2021 Aug 23)**

* Added plot_vortex and make figures

**0.2.1 (2021 Aug 22)**

* Removed bug in make_conditioned_data_figure

**0.2.0 (2021 Aug 22)**

* Added some capabilities to deal with gaps in data

**0.1.0 (2021 Aug 22)**

* Added read_data module

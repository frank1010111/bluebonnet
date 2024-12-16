---
title:
  "Bluebonnet: Scaling solutions for production analysis from unconventional oil
  and gas wells"
tags:
  - Python
  - hydraulic fracturing
  - production analysis
  - production forecasting
  - multiphase flow
authors:
  - name: Frank Male
    orcid: 0000-0002-3402-5578
    corresponding: true
    affiliation: "1, 2"
  - name: Michael P. Marder
    affiliation: 2
  - name: Leopoldo M. Ruiz-Maraggi
    affiliation: 2
  - name: Larry W. Lake
    affiliation: 2
affiliations:
  - name: Pennsylvania State University, University Park, PA, USA
    index: 1
  - name: University of Texas at Austin, TX, USA
    index: 2
date: 24 October 2022
bibliography: paper.bib
---

# Summary

Unconventional oil and gas wells are only productive due to extensive hydraulic
fracturing treatments. Therefore, the character of their production over time is
greatly influenced by engineering decisions. However, it can be difficult to
separate the engineering decisions from the effects due to fluid properties.
Also, during production these wells might be producing oil, gas, and water
simultaneously, with each phase interacting with the others. Numerical tools are
necessary to fully capture the effects of fluid properties on production.

`Bluebonnet` is a Python package that uses dimensionally scaled solutions of a
pressure diffusivity equation to analyze, history-match, and forecast production
of tight-oil and shale gas wells. `Bluebonnet` has been developed to help
researchers and petroleum engineers analyzing production data from
unconventional (shale gas and tight oil) wells. It provides the user with a set
of tools to evaluate production performance of tight-oil and shale gas wells.
These tools provide the following functionality:

1. `bluebonnet.fluids`: pressure-volume-temperature properties for oil, water,
   and gas phases.
2. `bluebonnet.flow`: physics-based production curves and hydrocarbon recovery
   factors.
3. `bluebonnet.forecast`: fits and forecasts for unconventional production.

<!-- prettier-ignore-->
The `fluids` submodule estimates the formation volume factors, solubility
ratios, and viscosity for the oil, water and gas phases given the reservoir
temperature, oil API gravity, gas specific gravity, and initial gas/oil ratio.

The `flow` submodule solves the pressure diffusivity equation to provide
estimates of the hydrocarbon production over time and the hydrocarbon recovery
factors. This module allows the user to estimate production for shale gas wells
using scaled solutions of the single-phase real gas diffusivity equation
[@patzek2013; @male2015application]. In addition, this module simulates
production for tight-oil and gas condensate wells using a two-phase scaled
solution of the pressure diffusivity equation [@ruizmaraggi2022twophase]. The
`flow` submodule also allows users to capture production variations due to
changes in bottomhole pressure.

The `forecast` submodule performs history matches and forecasts the production
of unconventional wells using the scaling solutions present in the `flow`
module. The `forecast` submodule also allows users to history-match and forecast
production of wells subject to variable bottomhole pressure conditions using a
modification of the approach developed by @ruizmaraggi2022pressure.

<!-- prettier-ignore -->
# Statement of need

`Bluebonnet` is a Python package using petroleum engineering methods to perform
production analysis of hydrofractured wells. Parts of this code were first
developed to assist in determining U.S. shale gas reserves [@patzek2013;
@male2019assessing].

There are no free open-source tools that use physics-based scaled flow solutions
of the diffusivity equation to perform decline-curve and rate-transient analysis
for unconventional reservoirs like `bluebonnet`. The goal for producing this
software package is to provide researchers and reservoir engineers with a free
and open source tool suitable to analyze production from unconventional (tight
oil and shale gas) reservoirs.

The present library can be used for the following tasks:

1. Estimate fluid properties of reservoir fluids.
2. Build type curves and recovery factors for shale gas and tight-oil
   reservoirs.
3. History-match and forecast the production of shale gas and tight-oil wells.
4. Perform Rate-transient analysis (rate-time-pressure) of unconventional
   reservoirs.

# Acknowledgements

This library would not exist without Tad Patzek introducing several of the
authors to the problem of unconventional production forecasting and kindly
providing code samples of the pressure diffusivity equation. We thank ExxonMobil
for funding this project with the grant "Heterogeneity and Unconventional
Production" (PI: Michael Marder). Valuable comments and criticism came from
discussions with Gary Hunter, Emre Turkoz, Zaheera Jabeen, and Deniz Ertas.

This project relies on the following open-source Python packages: NumPy
[@numpy2011; @numpy2020], SciPy [@scipy2020], matplotlib [@matplotlib2007], and
pandas [@pandas2010].

The authors would like to thank the Society of Petroleum Engineers (SPE) for
providing open access to production data from unconventional wells through the
SPE Data Repository, Data Set 1 [@spedata] used to illustrate the application of
this package.

# References

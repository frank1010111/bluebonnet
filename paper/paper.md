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
affiliations:
  - name: Pennsylvania State University, University Park, PA, USA
    index: 1
  - name: University of Texas at Austin, TX, USA
    index: 2
date: 20 September 2022
bibliography: paper.bib
---

# Summary

Unconventional oil and gas wells produce only due to extensive hydraulic
fracturing treatments. Therefore, the character of their production over time is
greatly influenced by engineering decisions. However, it can be difficult to
separate the engineering decisions from the effects from fluid properties. Also,
during production these wells might be producing oil, gas, and water
simultaneously, with each phase interacting with the others. Numerical tools are
necessary to fully capture the effects of fluid properties on production.

`Bluebonnet` is a Python package that uses scaling solutions of the pressure
diffusivity equation to analyze, history-match, and forecast production of
tight-oil and shale gas wells. `Bluebonnet` development aims to help researchers
and petroleum engineers analyzing production data from unconventional (shale gas
and tight oil) wells. It provides the user with a collection of tools to
evaluate production performance of tight-oil and shale gas wells. These tools
are:

1. `fluids` calculates pressure-volume-temperatura data for oil, water, and gas
   phases.
2. `flow` for building physics-based production curves and estimating
   hydrocarbon recovery factors.
3. `forecast` for fitting and forecasting unconventional production.

The `fluids` subpackage estimates the formation volume factors, solubility
ratios, and viscosity for the oil, water and gas phases given the reservoir
temperature, oil API gravity, gas specific gravity, and initial gas/oil ratio.
**Fig. 1** illustrates the plots of the: **(a)** formation volume factors and
**(b)** viscosities for the oil, gas, and water phases using the `fluids`
subpackage.

**Fig. 1**—Plots of the: **(a)** formation volume factors and **(b)**
viscosities for the oil, gas, and water phases using the `fluids` subpackage.

The `flow` subpackage solves the pressure diffusivity equation to provide
estimates of the hydrocarbon production with time and the hydrocarbon recovery
factors. This module allows the user to estimate production for shale gas wells
using the scaled solutions of the single-phase real gas diffusivity equation
[@patzek2013; @male2015application]. In addition, this module simulates
production for tight-oil and gas condensate wells using a two-phase scaled
solution of the pressure diffusivity equation [@ruizmaraggi2022twophase].

**Fig. 2** shows the gas recovery factors for single-phase ideal gas, real gas,
and multiphase scaled flow solutions using the `flow` subpackage.

**Fig. 2**—Plots of the gas recovery factors for ideal gas, real gas, and
multiphase flow solutions of the pressure diffusivity equation using the `flow`
subpackage.

The `flow` subpackage also allows to history-match and forecast production of
wells subject to variable bottomhole pressure conditions using a modification of
the approach developed by @ruizmaraggi2022pressure. **Fig.3** illustrates the
**(a)** history-match of the gas well #20 from the SPE data repository, dataset
1 (Society of Petroleum Engineers 2021) subject to variable bottomhole flowing
pressure conditions **(b)**.

**Fig. 3**—Plots the **(a)** history-match of the gas well # 20 from the SPE
data repository, dataset 1 (Society of Petroleum Engineers 2021) subject to
variable bottomhole flowing pressure conditions **(b)**.

The `forecast` subpackage performs history matches and forecasts the production
of unconventional wells using the scaling solutions present in the `flow`
module. **Fig.4** illustrates the history-match of a gas well using the
single-phase real gas flow solution.

**Fig. 4**—History-match of a shale gas well (blue dotted curve) using the
single-phase real gas flow solution (solid red curve).

# Statement of need

`Bluebonnet` is a petroleum engineering focused Python package for production
analysis of hydrofractured wells. Parts of this code were first developed to
assist in determining U.S. shale gas reserves (Patzek, Male, and Marder 2013).

There are no free open-source tools that use physics-based scaled flow solutions
of the diffusivity equation to perform decline-curve and rate-transient analysis
for unconventional reservoirs like `bluebonnet`. The main goal of producing this
software package is to provide researches and reservoir engineers with a free
tool suitable to analyze production from unconventional (tight oil and shale
gas) reservoirs.

The present library can be used for the following tasks:

1. Estimate fluid properties of reservoir fluids.
2. Build type curves and recovery factors for shale gas and tight-oil
   reservoirs.
3. History-match and forecast the production of shale gas and tight-oil wells.
4. Rate-transient (rate-time-pressure) analysis of unconventional reservoirs.

<!--
# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }
-->

# Acknowledgements

This library would never exist without Tad Patzek introducing several of the
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
SPE Data Repository, Data Set 1 (Society of Petroleum Engineers 2021) used to
illustrate the application of the `bluebonnet` package.

# References
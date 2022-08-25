---
title:
  "Bluebonnet: Scaling solutions for production analysis from unconventional oil
  and gas wells"
tags:
  - Python
  - hydraulic fracturing
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
date: 25 August 2022
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

# Statement of need

`Bluebonnet` is a petroleum engineering focused Python package for production
analysis of hydrofractured wells. Parts of this code were first developed to
assist in determining U.S. shale gas reserves [@Patzek:2013].

`Bluebonnet` is meant for researchers and petroleum engineers to analyze
unconventional wells.

<!--
Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

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
providing code samples of pressure diffusion. We thank ExxonMobil for funding
this project with the grant "Heterogeneity and Unconventional Production" (PI:
Michael Marder) . Valuable comments and criticism came from discussions with
Gary Hunter, Emre Turkoz, Zaheera Jabeen, and Deniz Ertas.

# References

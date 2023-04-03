# bluebonnet

Scaling solutions for production analysis from unconventional oil and gas wells.

<p align="center">
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
<a href="https://codecov.io/gh/frank1010111/bluebonnet" >
<a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="BSD License"></a>
<a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit powered"></a>
</p>
<p align="center">
<a href="https://bluebonnet.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/bluebonnet/badge/?version=latest" alt="Documentation"></a>
<a href="https://github.com/frank1010111/bluebonnet/actions/workflows/tests.yml/"> <img src="https://github.com/frank1010111/bluebonnet/actions/workflows/tests.yml/badge.svg" alt="tests">
</a>
 <img src="https://codecov.io/gh/frank1010111/bluebonnet/branch/main/graph/badge.svg?token=2I28WS7LYQ"/>
 </a>
<a href="https://pypi.org/project/bluebonnet">
 <img src="https://img.shields.io/pypi/dm/bluebonnet">
</a>
</p>
<p align="center">
<a href="https://joss.theoj.org/papers/4837f716d7ac1273629fb7d3f8b4ca10"><img src="https://joss.theoj.org/papers/4837f716d7ac1273629fb7d3f8b4ca10/status.svg"></a>
</p>

![bluebonnets in bloom](https://github.com/frank1010111/bluebonnet/raw/main/docs/_static/bluebonnets.jpg)

## Installation

Run the command

```bash
pip install bluebonnet
```

This package works for Python versions 3.8-3.11. Dependencies are automatically
installed by `pip`. They include standards of the Python scientific stack,
including `lmfit`, `matplotlib`, `numpy`, `pandas`, and `scipy`.

## Usage

`bluebonnet` has a collection of tools for performing reservoir simulation in
tight oil and shale gas reservoirs. The main tools are:

1. `fluids` for modeling PVT and viscosity of oil, water, and gas;
2. `flow` for building physics-based production curves; and
3. `forecast` for fitting and forecasting unconventional production.

Examples can be found in
[the documentation](https://bluebonnet.readthedocs.io/en/latest).

## Contributing

Interested in contributing? Check out the
[contributing guidelines](https://bluebonnet.readthedocs.io/en/latest/contributing.html)
to get started. Please note that this project is released with a Code of
Conduct. By contributing to this project, you agree to abide by its terms.

### Contributor Hall of Fame

Michael Marder

## License

`bluebonnet` was created by Frank Male. It is licensed under the terms of the
BSD 3-Clause license.

## Credits

This work was funded in part by an ExxonMobil grant to the University of Texas
at Austin, with Michael Marder as PI and Larry Lake as co-PI. The Physics-based
scaling curve was developed for shale gas reservoirs by Patzek et al. (2013). It
was extended to tight oil by Male (2019). It was extended to two-phase by Ruiz
Maraggi et al. (2020). It was extended to include variable fracture face
pressure by Ruiz Maraggi et al. (2021). In the future, it might be extended
further.  
`bluebonnet` was created with
[`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the
`py-pkgs-cookiecutter`
[template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## Bibliography

Papers developing or using this approach include:

1. Patzek, T. W., Male, F. and Marder, M., 2013. "Gas production in the Barnett
   Shale obeys a simple scaling theory," Proceedings of the National Academy of
   Science. https://doi.org/10.1073/pnas.1313380110
1. Patzek, T. W., Male, F. and Marder, M., 2014. "A simple model of gas
   production from hydrofractured horizontal wells in shales," AAPG Bulletin v.
   98, no. 12. https://doi.org/10.1306/03241412125
1. Male, F., Islam, A.W., Patzek, T.W., Ikonnikova, S.A., Browning, J.R., and
   Marder, M.P., 2015. "Analysis of gas production from hydraulically fractured
   wells in the Haynesville shale using scaling methods." Journal of
   Unconventional Oil and Gas Resources.
   https://doi.org/10.1016/j.juogr.2015.03.001
1. Male, F., 2015. Application of a one dimensional nonlinear model to flow in
   hydrofractured shale gas wells using scaling solutions (Doctoral
   dissertation). https://repositories.lib.utexas.edu/handle/2152/46706
1. Eftekhari, B., Marder, M. and Patzek, T.W., 2018. Field data provide
   estimates of effective permeability, fracture spacing, well drainage area and
   incremental production in gas shales. Journal of Natural Gas Science and
   Engineering, 56, pp.141-151. https://doi.org/10.1016/j.jngse.2018.05.027
1. Male, F. 2019, "Assessing impact of uncertainties in decline curve analysis
   through hindcasting." Journal of Petroleum Science and Engineering, 172,
   340-348. https://doi.org/10.1016/j.petrol.2018.09.072
1. Male, F. 2019, "Using a segregated flow model to forecast production of oil,
   gas, and water in shale oil wells." Journal of Petroleum Science and
   Engineering, 180, 48-61. https://doi.org/10.1016/j.petrol.2019.05.010
1. Patzek, T.W., Saputra, W., Kirati, W. and Marder, M., 2019. "Generalized
   extreme value statistics, physical scaling, and forecasts of gas production
   in the Barnett shale." Energy & fuels, 33(12), pp.12154-12169.
   https://doi.org/10.1021/acs.energyfuels.9b01385
1. Ruiz Maraggi, L.M., Lake, L.W. and Walsh, M.P., 2020. "A Two-Phase Non-Linear
   One-Dimensional Flow Model for Reserves Estimation in Tight Oil and Gas
   Condensate Reservoirs Using Scaling Principles." In SPE Latin American and
   Caribbean Petroleum Engineering Conference. OnePetro.
   https://doi.org/10.2118/199032-MS
1. Ruiz Maraggi, L.M., Lake, L.W. and Walsh, M.P., 2020. "A Bayesian Framework
   for Addressing the Uncertainty in Production Forecasts of Tight Oil
   Reservoirs Using a Physics-Based Two-Phase Flow Model." In SPE/AAPG/SEG Latin
   America Unconventional Resources Technology Conference. OnePetro.
   https://doi.org/10.15530/urtec-2020-10480
1. Maraggi, L.M.R., Lake, L.W. and Walsh, M.P., 2021. Deconvolution of
   Time-Varying Bottomhole Pressure Improves Rate-Time Models History Matches
   and Forecasts of Tight-Oil Wells Production. In SPE/AAPG/SEG Unconventional
   Resources Technology Conference. OnePetro.
1. Ruiz Maraggi, L.M., Lake, L.W., and Walsh. M.P., 2022 "Rate-Pseudopressure
   Deconvolution Enhances Rate-Time Models Production History Matches and
   Forecasts of Shale Gas Wells." Paper presented at the SPE Canadian Energy
   Technology Conference, Calgary, Alberta, Canada, March 2022. doi:
   https://doi.org/10.2118/208967-MS
1. Ruiz Maraggi, L.M., Lake, L.W. and Walsh, M.P., 2022. Deconvolution Overcomes
   the Limitations of Rate Normalization and Material Balance Time in
   Rate-Transient Analysis of Unconventional Reservoirs. In SPE Canadian Energy
   Technology Conference. OnePetro.
1. Male, F., Duncan, I.J., 2022, "The Paradox of Increasing Initial Oil
   Production but Faster Decline Rates in Fracking the Bakken Shale:
   Implications for Long Term Productivity of Tight Oil Plays," Journal of
   Petroleum Science and Engineering,
   https://doi.org/10.1016/j.petrol.2021.109406
1. Ruiz Maraggi, L.M., 2022. Production analysis and forecasting of shale
   reservoirs using simple mechanistic and statistical modeling (Doctoral
   dissertation). http://dx.doi.org/10.26153/tsw/42112

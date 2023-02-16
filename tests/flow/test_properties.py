"""Define a suite a tests for the flowproperties module."""
from __future__ import annotations

from copy import copy

import numpy as np
import pandas as pd
import pytest
from bluebonnet.flow.flowproperties import (
    FlowPropertiesTwoPhase,
    RelPermParams,
    relative_permeabilities,
    relative_permeabilities_twophase,
    rescale_pseudopressure,
)

pr = 8_000.0
Sw = 0.1
columns_renamer_gas = {
    "P": "pressure",
    "Z-Factor": "z-factor",
    "Cg": "compressibility",
    "Viscosity": "viscosity",
    "Density": "density",
}
columns_renamer_oil = {
    "P": "pressure",
    "Z-Factor": "z-factor",
    "Co": "compressibility",
    "Oil_Viscosity": "viscosity",
    "Oil_Density": "density",
}
pvt_gas = pd.read_csv("tests/data/pvt_gas.csv").rename(columns=columns_renamer_gas)
pvt_oil = pd.read_csv("tests/data/pvt_oil.csv").rename(columns=columns_renamer_oil)


@pytest.fixture()
def relperm_params():
    return RelPermParams(
        n_o=1,
        n_g=1,
        n_w=1,
        S_or=0,
        S_gc=0,
        S_wc=0.1,
        k_ro_max=1,
        k_rw_max=1,
        k_rg_max=1,
    )


@pytest.fixture()
def saturations_test():
    return pd.DataFrame(
        {
            "So": np.linspace(0, 1 - Sw, 50),
            "Sw": np.full(50, Sw),
            "Sg": np.linspace(1 - Sw, 0, 50),
        }
    )


@pytest.fixture()
def df_pvt():
    # get pvt tables
    pvt_oil = pd.read_csv("tests/data/pvt_oil.csv")
    pvt_water = pd.read_csv("tests/data/pvt_water.csv").rename(
        columns={"T": "temperature", "P": "pressure", "Viscosity": "mu_w"}
    )
    rename_cols = {
        "T": "temperature",
        "P": "pressure",
        "Oil_Viscosity": "mu_o",
        "Gas_Viscosity": "mu_g",
        "Rso": "Rs",
    }
    df_pvt = (
        pvt_water.drop(columns=["temperature"])
        .merge(
            pvt_oil.rename(columns=rename_cols),
            on="pressure",
        )
        .assign(Rv=0)
    )
    # calculate So, Sg assuming no mobile water
    df_pvt["So"] = (1 - Sw) / (
        (df_pvt["Rs"].max() - df_pvt["Rs"]) * df_pvt["Bg"] / df_pvt["Bo"] / 5.61458 + 1
    )
    return df_pvt


# # TODO: Figure out how to test FlowProperties
# @pytest.mark.parametrize("pvt_table", [pvt_gas, pvt_oil])
# def test_flow_properties(pvt_table):
#     alpha = 1 / (pvt_table["viscosity"] * pvt_table["density"])
#     fluid = FlowProperties(pvt_table, pr)
#     assert np.allclose(alpha, fluid.pvt_props["alpha"])


def test_relative_perms(relperm_params):
    """Relative permeability testing."""
    df_kr = relative_permeabilities_twophase(relperm_params)
    for fluid in ["o", "g"]:
        assert (
            df_kr.iloc[df_kr[f"S{fluid}"].argmax()][f"kr{fluid}"]
            > df_kr.iloc[df_kr[f"S{fluid}"].argmin()][f"kr{fluid}"]
        ), "relative permeability should be higher at higher saturation"
    assert df_kr["krw"].max() < 1e-3, "Water should be immobile below S_wr"


def test_relperm_bounds_errors(relperm_params, saturations_test):
    saturations = copy(saturations_test)
    saturations += 1
    with pytest.raises(ValueError, match="Saturations must sum to 1"):
        relative_permeabilities(saturations.to_records(index=False), relperm_params)
    # test parameter bounds
    relperm_wrong = relperm_params._replace(n_o=8)
    with pytest.raises(ValueError, match="Exponents .* less than"):
        relative_permeabilities(saturations_test.to_records(index=False), relperm_wrong)
    relperm_wrong = relperm_params._replace(n_w=0)
    with pytest.raises(ValueError, match="Exponents .* at least"):
        relative_permeabilities(saturations_test.to_records(index=False), relperm_wrong)
    relperm_wrong = relperm_params._replace(S_gc=-1)
    with pytest.raises(ValueError, match="saturation.* at least"):
        relative_permeabilities(saturations_test.to_records(index=False), relperm_wrong)
    relperm_wrong = relperm_params._replace(S_or=1.1)
    with pytest.raises(ValueError, match="saturation.* less than"):
        relative_permeabilities(saturations_test.to_records(index=False), relperm_wrong)
    relperm_wrong = relperm_params._replace(k_ro_max=1.1)
    with pytest.raises(ValueError, match="relative permeability.* less than"):
        relative_permeabilities(saturations_test.to_records(index=False), relperm_wrong)


def test_relative_perm_twophase(relperm_params):
    with pytest.raises(ValueError, match="Water saturation is above residual"):
        relative_permeabilities_twophase(relperm_params, Sw=0.8)
    df_kr_test = relative_permeabilities_twophase(relperm_params, Sw)
    for col in ("So", "Sw", "Sg", "kro", "krw", "krg"):
        assert col in df_kr_test.columns


def test_multiphase_flowproperties(relperm_params, df_pvt):
    p_frac = 1000
    p_res = pr
    phi = 0.1
    # scale pseudopressure
    df_pvt = rescale_pseudopressure(df_pvt, p_frac, p_res)
    df_kr = relative_permeabilities_twophase(relperm_params)
    reference_densities = {
        "rho_o0": 141.5 / (45 + 131.5),
        "rho_g0": 1.03e-3,
        "rho_w0": 1,
    }
    flow_props = FlowPropertiesTwoPhase.from_table(
        df_pvt, df_kr, reference_densities, phi, Sw, p_res
    )
    assert flow_props is not None, "Smoke testing FlowPropertiesTwoPhase"
    for fluid in "ogw":
        k_rel = flow_props.kr[f"kr{fluid}"]([0, 0.2, 0.4])
        assert np.max(k_rel) <= 1.0, "k_rel should be less than or equal to 1"
    # TODO: could use some more tests

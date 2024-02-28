# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas
import numpy
from matplotlib import pyplot
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from scipy.stats import gaussian_kde

# %% [markdown]
# ## Aim
#
# Estimate the number of households that are off the gas grid by urban - rural setting.
#
# ### Data
# [P002: Number of households by postcode, 2021](https://www.nomisweb.co.uk/sources/census_2021_pc)
# [xoserve off gas grid postcodes](https://www.xoserve.com/help-centre/supply-points-metering/supply-point-administration-spa/#off-gas-postcode-register5)
# [ons postcode directory](https://geoportal.statistics.gov.uk/)

# %%
# Load full ons postcode directory
with TemporaryDirectory() as tmp:
    with ZipFile("../../inputs/data/ONSPD_FEB_2024_UK.zip") as zf:
        zf.extractall(tmp)
        onspd = (
            pandas.read_csv(
                f"{tmp}/Data/ONSPD_FEB_2024_UK.csv",
                usecols=["pcd", "oa11", "oa21", "ru11ind"],
                dtype={"pcd": str, "oa11": str, "oa21": str, "ru11ind": str},
            )
            .assign(pcstrip=lambda df: df["pcd"].str.replace("\s+", "", regex=True))
            .loc[lambda df: df["oa11"].str[0].isin(["E", "W"])]
        )

# %%
# Load postcode household counts
# NB only postcode that have at least 1 household (in 2021 Census)
hh_counts = pandas.read_csv("../../inputs/data/pcd_p002.csv").assign(
    pcstrip=lambda df: df["Postcode"].str.replace("\s+", "", regex=True)
)

# %%
# Load off gas grid postcodes
off_gas = (
    pandas.read_excel(
        "../../inputs/data/off_gas_postcode_register_oct_2023_v3_2.xlsx", sheet_name=1
    )
    .assign(pcstrip=lambda df: df["Post Code"].str.replace("\s+", "", regex=True))[
        "pcstrip"
    ]
    .to_list()
)

# %%
# Merge data - misses 3 postcodes against hh_counts
hh_counts = hh_counts.merge(onspd, how="inner", on="pcstrip").assign(
    off_gas=lambda df: df["pcstrip"].isin(off_gas),
    off_gas_hh=lambda df: df["off_gas"] * df["Count"],
)

# %%
# This says only 10% of households are off gas. Feels low.
# However, xoserve measure is postcodes not recorded in UK Link, the register of gas infrastructure.
# So likely to overestimate gas connections vs. a household measure.
hh_counts["off_gas_hh"].sum() / hh_counts["Count"].sum() * 100

# %%
ruc11 = {
    "A1": "Urban: Major Conurbation",
    "B1": "Urban: Minor Conurbation",
    "C1": "Urban: City and Town",
    "C2": "Urban: City and Town in a Sparse Setting",
    "D1": "Rural: Town and Fringe",
    "D2": "Rural: Town and Fringe in a Sparse Setting",
    "E1": "Rural: Village",
    "E2": "Rural: Village in a Sparse Setting",
    "F1": "Rural: Hamlets and Isolated Dwellings",
    "F2": "Rural: Hamlets and Isolated Dwellings in a Sparse Setting",
}

(
    hh_counts.groupby("ru11ind", as_index=False)
    .agg({"Count": "sum", "off_gas_hh": "sum"})
    .assign(
        off_gas_prop=lambda df: df["off_gas_hh"] / df["Count"] * 100,
        ruc_name=lambda df: df["ru11ind"].map(ruc11),
    )
)

# %%
ruc11 = {
    "A1": "Urban",
    "B1": "Urban",
    "C1": "Urban",
    "C2": "Urban",
    "D1": "Rural",
    "D2": "Rural",
    "E1": "Rural",
    "E2": "Rural",
    "F1": "Rural",
    "F2": "Rural",
}

(
    hh_counts.assign(urban_rural=lambda df: df["ru11ind"].map(ruc11))
    .groupby("urban_rural", as_index=False)
    .agg({"Count": "sum", "off_gas_hh": "sum"})
    .assign(off_gas_prop=lambda df: df["off_gas_hh"] / df["Count"] * 100)
)

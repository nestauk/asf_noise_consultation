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
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from matplotlib import pyplot

# %% [markdown]
# ## Aim
#
# Look at urban-rural rates of heat pump adoption.
#
# Because this is noise-related, we'll limit the focus to ASHP.
#
# We'll also limit the focus to England and Wales (where we observe a value under RUC11) in line with the other analysis and due to Scotland's different rural-urban classification.
#
# For simplicity, I'll use Census 2011 household counts as the denominator for urban-rural categories.
#
# ### Data

# %%
# MCS data
mcs = pandas.read_parquet(
    "s3://asf-daps/lakehouse/processed/mcs/mcs_installations_231023-0.parquet"
).assign(pcstrip=lambda df: df["postcode"].str.replace(" ", ""))

# %%
# ONSPD
with TemporaryDirectory() as tmp:
    with ZipFile("../../inputs/data/ONSPD_FEB_2024_UK.zip") as zf:
        zf.extractall(tmp)
        onspd = pandas.read_csv(
            f"{tmp}/Data/ONSPD_FEB_2024_UK.csv",
            usecols=["pcd", "ru11ind"],
            dtype={"pcd": "str", "ru11ind": "str"},
        ).assign(pcstrip=lambda df: df["pcd"].str.replace(" ", ""))

# %%
# Merge ONSPD data
mcs = mcs.merge(onspd, how="left", on="pcstrip")

# %%
# Apply inclusion criteria
mcs = mcs.loc[
    lambda df: (df["tech_type"] == "Air Source Heat Pump")
    & df["ru11ind"].isin(["A1", "B1", "C1", "C2", "D1", "D2", "E1", "E2", "F1", "F2"]),
    :,
]

# %%
# Add RUC11 lookup
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

mcs = mcs.assign(
    ru11_name=lambda df: df["ru11ind"].map(ruc11),
    ru11_2cat=lambda df: df["ru11ind"].apply(
        lambda x: (
            "Urban"
            if x
            in [
                "A1",
                "B1",
                "C1",
                "C2",
            ]
            else "Rural"
        )
    ),
)

# %%
# Urban Rural Denominators
# Using Census 2011 QS418EW - Number of Dwellings
dwellings = pandas.read_csv(
    "../../inputs/data/1030629437103289.csv", skiprows=7, skipfooter=5, engine="python"
).rename(columns={"2011": "dwellings"})

# %%
# Read RUC2011 OA data sheet, fix non-breaking spaces in categories, add country.
ruc11_oa = pandas.read_excel(
    "../../inputs/data/Rural_Urban_Classification_2011_lookup_tables_for_small_area_geographies.ods",
    sheet_name="OA11",
    skiprows=2,
    dtype={
        "Output Area 2011 Code": str,
        "Rural Urban Classification 2011 code": pandas.CategoricalDtype(
            categories=[
                "A1",
                "A2",
                "C1\xa0\xa0",
                "C2",
                "D1\xa0",
                "D2",
                "E1",
                "E2",
                "F1",
                "F2",
            ]
        ),
        "Rural Urban Classification 2011 (10 fold)": pandas.CategoricalDtype(
            categories=[
                "Urban major conurbation\xa0",
                "Urban minor conurbation",
                "Urban city and town\xa0\xa0",
                "Urban city and town in a sparse setting",
                "Rural town and fringe\xa0",
                "Rural town and fringe\xa0in a sparse setting\xa0",
                "Rural village",
                "Rural village\xa0in a sparse setting",
                "Rural hamlets and isolated dwellings",
                "Rural hamlets and isolated dwellings in a sparse setting",
            ]
        ),
        "Rural Urban Classification 2011 (2 fold)": pandas.CategoricalDtype(
            categories=["Urban", "Rural"]
        ),
    },
).assign(
    **{
        "Rural Urban Classification 2011 (10 fold)": lambda df: df[
            "Rural Urban Classification 2011 (10 fold)"
        ].cat.rename_categories(
            {
                "Urban major conurbation\xa0": "Urban major conurbation",
                "Urban city and town\xa0\xa0": "Urban city and town",
                "Rural town and fringe\xa0": "Rural town and fringe",
                "Rural village\xa0in a sparse setting": "Rural village in a sparse setting",
                "Rural town and fringe\xa0in a sparse setting\xa0": "Rural town and fringe in a sparse setting",
            }
        ),
        "Rural Urban Classification 2011 code": lambda df: df[
            "Rural Urban Classification 2011 code"
        ].cat.rename_categories({"A2": "B1", "D1\xa0": "D1", "C1\xa0\xa0": "C1"}),
        "Country": lambda df: df["Output Area 2011 Code"]
        .str[0]
        .map({"E": "England", "W": "Wales"})
        .astype("category"),
    }
)

# %%
ruc11_oa = ruc11_oa.merge(
    dwellings, how="left", left_on="Output Area 2011 Code", right_on="2011 output area"
)

# %%
# Create dwelling denominators
ruc11_dwellings = ruc11_oa.groupby(
    [
        "Rural Urban Classification 2011 code",
        "Rural Urban Classification 2011 (10 fold)",
    ],
    observed=True,
    as_index=False,
)["dwellings"].sum()

# %%
# Create dwelling denominators
broad_ruc11_dwellings = ruc11_oa.groupby(
    "Rural Urban Classification 2011 (2 fold)", observed=True, as_index=False
)["dwellings"].sum()

# %% [markdown]
# ### Analysis

# %%
broad_rates = (
    mcs.groupby(["commission_year", "ru11_2cat"], as_index=False)["ru11ind"]
    .count()
    .merge(
        broad_ruc11_dwellings,
        how="left",
        left_on="ru11_2cat",
        right_on="Rural Urban Classification 2011 (2 fold)",
    )
    .assign(install_rate=lambda df: df["ru11ind"] / df["dwellings"] * 10_000)
)

# %%
broad_rates.pivot(index="ru11_2cat", columns="commission_year", values="install_rate")

# %%
f, ax = pyplot.subplots(figsize=(9, 4))

for setting in ["Urban", "Rural"]:
    ax.plot(
        broad_rates.loc[lambda df: df["ru11_2cat"] == setting, "commission_year"],
        broad_rates.loc[lambda df: df["ru11_2cat"] == setting, "install_rate"],
        label=setting,
    )

ax.set_xlabel("Year")
ax.set_ylabel("Heat Pump Installation Rate\n(per 10,000 households)")
ax.legend()

# %%
f, ax = pyplot.subplots(figsize=(9, 4))

for setting in ["Urban", "Rural"]:
    temp = (
        mcs.loc[lambda df: df["ru11_2cat"] == setting, ["commission_date"]]
        .assign(count=1)
        .set_index("commission_date")
        .sort_index()
        .rolling("365D")
        .sum()
        / broad_ruc11_dwellings.loc[
            lambda df: df["Rural Urban Classification 2011 (2 fold)"] == setting,
            "dwellings",
        ].values[0]
        * 10000
    )
    ax.plot(temp.index, temp, label=setting)
ax.set_xlabel("Year")
ax.set_ylabel("Heat Pump Installation Rate\n(per 10,000 households)")
ax.legend()

# %%
full_rates = (
    mcs.groupby(["commission_year", "ru11ind", "ru11_name"], as_index=False)[
        "ru11_2cat"
    ]
    .count()
    .merge(
        ruc11_dwellings,
        how="left",
        left_on="ru11ind",
        right_on="Rural Urban Classification 2011 code",
    )
    .assign(install_rate=lambda df: df["ru11_2cat"] / df["dwellings"] * 10_000)
)

# %%
full_rates.pivot(index="ru11_name", columns="commission_year", values="install_rate")

# %%
f, ax = pyplot.subplots(figsize=(9, 4))

colours = {
    "Urban: Major Conurbation": "#08519c",
    "Urban: Minor Conurbation": "#3182bd",
    "Urban: City and Town": "#6baed6",
    "Urban: City and Town in a Sparse Setting": "#bdd7e7",
    "Rural: Town and Fringe": "#8c2d04",
    "Rural: Town and Fringe in a Sparse Setting": "#d94801",
    "Rural: Village": "#f16913",
    "Rural: Village in a Sparse Setting": "#fd8d3c",
    "Rural: Hamlets and Isolated Dwellings": "#fdae6b",
    "Rural: Hamlets and Isolated Dwellings in a Sparse Setting": "#fdd0a2",
}

for setting in colours.keys():
    ax.plot(
        full_rates.loc[lambda df: df["ru11_name"] == setting, "commission_year"],
        full_rates.loc[lambda df: df["ru11_name"] == setting, "install_rate"],
        color=colours[setting],
        label=setting,
    )

ax.set_xlabel("Year")
ax.set_ylabel("Heat Pump Installation Rate\n(per 10,000 households)")
ax.legend(fontsize=8)

# %%
ruc11_dwellings

# %%
f, ax = pyplot.subplots(figsize=(9, 5))

colours = {
    "Urban: Major Conurbation": "#08519c",
    "Urban: Minor Conurbation": "#3182bd",
    "Urban: City and Town": "#6baed6",
    "Urban: City and Town in a Sparse Setting": "#bdd7e7",
    "Rural: Town and Fringe": "#8c2d04",
    "Rural: Town and Fringe in a Sparse Setting": "#d94801",
    "Rural: Village": "#f16913",
    "Rural: Village in a Sparse Setting": "#fd8d3c",
    "Rural: Hamlets and Isolated Dwellings": "#fdae6b",
    "Rural: Hamlets and Isolated Dwellings in a Sparse Setting": "#fdd0a2",
}

codes = {
    "Urban: Major Conurbation": "A1",
    "Urban: Minor Conurbation": "B1",
    "Urban: City and Town": "C1",
    "Urban: City and Town in a Sparse Setting": "C2",
    "Rural: Town and Fringe": "D1",
    "Rural: Town and Fringe in a Sparse Setting": "D2",
    "Rural: Village": "E1",
    "Rural: Village in a Sparse Setting": "E2",
    "Rural: Hamlets and Isolated Dwellings": "F1",
    "Rural: Hamlets and Isolated Dwellings in a Sparse Setting": "F2",
}

for setting in colours.keys():
    temp = (
        mcs.loc[lambda df: df["ru11_name"] == setting, ["commission_date"]]
        .assign(count=1)
        .set_index("commission_date")
        .sort_index()
        .rolling("365D")
        .sum()
        / ruc11_dwellings.loc[
            lambda df: df["Rural Urban Classification 2011 code"] == codes[setting],
            "dwellings",
        ].values[0]
        * 10000
    )
    ax.plot(temp.index, temp, label=setting, color=colours[setting])

ax.set_xlabel("Year")
ax.set_ylabel("Heat Pump Installation Rate\n(per 10,000 households)")
ax.legend(fontsize=9)

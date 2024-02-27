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
import geopandas
from matplotlib import pyplot
import numpy
from scipy.stats import gaussian_kde

# %% [markdown]
# ## Aim of Notebook
#
# Create a profile of urban and rural areas according to the current rural urban classification 2011. We want to understand:
# * The number of households that fall within each rural-urban category.
# * The density of households within each rural-urban category.
# * The distribution of property types within each rural-urban category.
# * Access to gardens within each rural-urban category.
#
# ## Data
# [Rural Urban Classification 2011 lookup tables for small area geographies (OA 2011; MSOA 2011)](https://www.gov.uk/government/statistics/2011-rural-urban-classification-lookup-tables-for-all-geographies).
# [Output Areas (2011) to Output Areas (2021) to Local Authority District (2022) Lookup in England and Wales (Version 2)](https://geoportal.statistics.gov.uk/datasets/ons::output-areas-2011-to-output-areas-2021-to-local-authority-district-2022-lookup-in-england-and-wales-version-2-1/about)
# [TS041 - Number of Households (OA 2021)](https://www.nomisweb.co.uk/query/construct/summary.asp?mode=construct&version=0&dataset=2059)
# [TS044 - Accommodation type (OA 2021)](https://www.nomisweb.co.uk/query/construct/summary.asp?mode=construct&version=0&dataset=2062)
# [Access to garden space, Great Britain, April 2020](https://www.ons.gov.uk/economy/environmentalaccounts/datasets/accesstogardensandpublicgreenspaceingreatbritain)

# %%
# Read RUC2011 OA data sheet, fix non-breaking spaces in categories, add country.
ruc11_oa = pandas.read_excel(
    "../../inputs/data/Rural_Urban_Classification_2011_lookup_tables_for_small_area_geographies.ods",
    sheet_name="OA11",
    skiprows=2,
    dtype={
        "Output Area 2011 Code": str,
        "Rural Urban Classification 2011 code": pandas.CategoricalDtype(
            categories=["A1", "A2", "C1", "C2", "D1", "D2", "E1", "E2", "F1", "F2"]
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
        "Country": lambda df: df["Output Area 2011 Code"]
        .str[0]
        .map({"E": "England", "W": "Wales"})
        .astype("category"),
    }
)

# %%
# Read RUC2011 OA data sheet, fix non-breaking spaces in categories, add country.
ruc11_msoa = pandas.read_excel(
    "../../inputs/data/Rural_Urban_Classification_2011_lookup_tables_for_small_area_geographies.ods",
    sheet_name="MSOA11",
    skiprows=2,
    dtype={
        "Middle Super Output Area 2011 Code": str,
        "Middle Super Output Area 2011 Name": str,
        "Rural Urban Classification 2011 code": pandas.CategoricalDtype(
            categories=[
                "A1",
                "B1",
                "C1\xa0\xa0",
                "C2\xa0",
                "D1\xa0",
                "D2\xa0",
                "E1",
                "E2",
            ]
        ),
        "Rural Urban Classification 2011 (10 fold)": pandas.CategoricalDtype(
            categories=[
                "Urban major conurbation\xa0",
                "Urban city and town\xa0\xa0",
                "Rural town and fringe\xa0",
                "Urban minor conurbation",
                "Rural village and dispersed",
                "Rural village and dispersed in a sparse setting",
                "Rural town and fringe\xa0in a sparse setting\xa0",
                "Urban city and town in a sparse setting\xa0",
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
                "Rural town and fringe\xa0": "Rural town and fringe",
                "Urban city and town\xa0\xa0": "Urban city and town",
                "Rural town and fringe\xa0in a sparse setting\xa0": "Rural town and fringe in a sparse setting",
                "Urban city and town in a sparse setting\xa0": "Urban city and town in a sparse setting",
            }
        ),
        "Rural Urban Classification 2011 code": lambda df: df[
            "Rural Urban Classification 2011 code"
        ].cat.rename_categories(
            {"C1\xa0\xa0": "C1", "C2\xa0": "C2", "D1\xa0": "D1", "D2\xa0": "D2"}
        ),
        "Country": lambda df: df["Middle Super Output Area 2011 Code"]
        .str[0]
        .map({"E": "England", "W": "Wales"})
        .astype("category"),
    }
)

# %%
# OA2011 to OA 2021 lookup
oa_bestfit = pandas.read_csv(
    "../../inputs/data/Output_Areas_(2011)_to_Output_Areas_(2021)_to_Local_Authority_District_(2022)_Lookup_in_England_and_Wales_(Version_2).csv",
    usecols=["OA11CD", "OA21CD"],
    dtype={"OA11CD": str, "OA21CD": str},
)

# %%
# Household numbers
hhs = pandas.read_csv(
    "../../inputs/data/3224998461328484.csv", skiprows=6, skipfooter=6, engine="python"
).rename(columns={"2021 output area": "OA21CD", "2021": "Households"})

# %%
# accomodation type
accom_type = pandas.read_csv(
    "../../inputs/data/32300121820536688.csv", skiprows=6, skipfooter=6, engine="python"
).rename(columns={"2021 output area": "OA21CD"})

# %%
# Access to garden space
# NB can't figure out what 'Flat: Private outdoor space count' actually represents.
# Although not stated, we infer these are 2011 MSOAs based on how many there are.
gardens = pandas.read_excel(
    "../../inputs/data/osprivateoutdoorspacereferencetables.xlsx",
    sheet_name="MSOA gardens",
    skiprows=1,
    skipfooter=3,
)
# Set a sensible header without typos.
gardens.columns = [
    "Country code",
    "Country name",
    "Region code",
    "Region name",
    "LAD code",
    "LAD name",
    "MSOA code",
    "MSOA name",
    "House: Address count",
    "House: Address with private outdoor space count",
    "House: Private outdoor space total area (m2)",
    "House: Percentage of addresses with private outdoor space",
    "House: Average size of private outdoor space (m2)",
    "House: Median size of private outdoor space (m2)",
    "Flat: Address count",
    "Flat: Address with private outdoor space count",
    "Flat: Private outdoor space total area (m2)",
    "Flat: Private outdoor space count",
    "Flat: Percentage of addresses with private outdoor space",
    "Flat: Average size of private outdoor space (m2)",
    "Flat: Average number of flats sharing a garden",
    "Total: Address count",
    "Total: Address with private outdoor space count",
    "Total: Private outdoor space total area (m2)",
    "Total: Percentage of addresses with private outdoor space",
    "Total: Average size of private outdoor space (m2)",
]

# %%
# Merge OA 2021 lookup and data to RUC 2011
ruc11_oa = (
    ruc11_oa.merge(
        oa_bestfit, how="left", left_on="Output Area 2011 Code", right_on="OA11CD"
    )
    .merge(hhs, how="left", on="OA21CD")
    .merge(accom_type, how="left", on="OA21CD")
)

# %%
# Merge gardens data to RUC 2011 (MSOA)
ruc11_msoa = ruc11_msoa.merge(
    gardens,
    how="left",
    left_on="Middle Super Output Area 2011 Code",
    right_on="MSOA code",
)

# %% [markdown]
# ## Summary Statistics
#
# ### Top-Level Summary of Households

# %%
ruc11_oa_broad_summary = pandas.concat(
    [
        ruc11_oa.groupby(
            ["Country", "Rural Urban Classification 2011 (2 fold)"],
            observed=True,
            as_index=False,
        )
        .agg({"Output Area 2011 Code": "count", "Households": "sum"})
        .rename(columns={"Output Area 2011 Code": "OA11_Count"}),
        ruc11_oa.groupby(
            "Rural Urban Classification 2011 (2 fold)", observed=True, as_index=False
        )
        .agg({"Output Area 2011 Code": "count", "Households": "sum"})
        .rename(columns={"Output Area 2011 Code": "OA11_Count"})
        .assign(Country="Total"),
    ],
    ignore_index=True,
)

# %%
hh_totals = {
    "England": ruc11_oa_broad_summary.loc[
        lambda df: df["Country"] == "England", "Households"
    ].sum(),
    "Wales": ruc11_oa_broad_summary.loc[
        lambda df: df["Country"] == "Wales", "Households"
    ].sum(),
    "Total": ruc11_oa_broad_summary.loc[
        lambda df: df["Country"] == "Total", "Households"
    ].sum(),
}

ruc11_oa_broad_summary = ruc11_oa_broad_summary.assign(
    hh_prop=lambda df: df.apply(
        lambda x: round(x["Households"] / hh_totals[x["Country"]] * 100, 1), axis=1
    )
)

ruc11_oa_broad_summary

# %% [markdown]
# ### Detailed Summary of Households

# %%
ruc11_oa_summary = pandas.concat(
    [
        ruc11_oa.groupby(
            [
                "Country",
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        )
        .agg({"Output Area 2011 Code": "count", "Households": "sum"})
        .rename(columns={"Output Area 2011 Code": "OA11_Count"}),
        ruc11_oa.groupby(
            [
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        )
        .agg({"Output Area 2011 Code": "count", "Households": "sum"})
        .rename(columns={"Output Area 2011 Code": "OA11_Count"})
        .assign(Country="Total"),
    ],
    ignore_index=True,
)

# %%
hh_totals = {
    "England": ruc11_oa_summary.loc[
        lambda df: df["Country"] == "England", "Households"
    ].sum(),
    "Wales": ruc11_oa_summary.loc[
        lambda df: df["Country"] == "Wales", "Households"
    ].sum(),
    "Total": ruc11_oa_summary.loc[
        lambda df: df["Country"] == "Total", "Households"
    ].sum(),
}

ruc11_oa_summary = ruc11_oa_summary.assign(
    hh_prop=lambda df: df.apply(
        lambda x: round(x["Households"] / hh_totals[x["Country"]] * 100, 1), axis=1
    )
)
ruc11_oa_summary

# %% [markdown]
# ### Accomodation Type
# Unit of measure is households

# %%
ruc11_oa_broad_accom_type_summary = pandas.concat(
    [
        ruc11_oa.groupby(
            ["Country", "Rural Urban Classification 2011 (2 fold)"],
            observed=True,
            as_index=False,
        )
        .agg(
            {
                "Detached": "sum",
                "Semi-detached": "sum",
                "Terraced": "sum",
                "In a purpose-built block of flats or tenement": "sum",
                "Part of a converted or shared house, including bedsits": "sum",
                "Part of another converted building, for example, former school, church or warehouse": "sum",
                "In a commercial building, for example, in an office building, hotel or over a shop": "sum",
            }
        )
        .rename(columns={"Output Area 2011 Code": "OA11_Count"}),
        ruc11_oa.groupby(
            "Rural Urban Classification 2011 (2 fold)", observed=True, as_index=False
        )
        .agg(
            {
                "Detached": "sum",
                "Semi-detached": "sum",
                "Terraced": "sum",
                "In a purpose-built block of flats or tenement": "sum",
                "Part of a converted or shared house, including bedsits": "sum",
                "Part of another converted building, for example, former school, church or warehouse": "sum",
                "In a commercial building, for example, in an office building, hotel or over a shop": "sum",
            }
        )
        .rename(columns={"Output Area 2011 Code": "OA11_Count"})
        .assign(Country="Total"),
    ],
    ignore_index=True,
)


# %%
# This function correctly creates lambdas.
# Without it only the final one created in the loop is used.
def makeFunc(x):
    return lambda df: eval(x)


accom = [
    "Detached",
    "Semi-detached",
    "Terraced",
    "In a purpose-built block of flats or tenement",
    "Part of a converted or shared house, including bedsits",
    "Part of another converted building, for example, former school, church or warehouse",
    "In a commercial building, for example, in an office building, hotel or over a shop",
]

accom_totals = {
    "England": {
        key: ruc11_oa_broad_accom_type_summary.loc[
            lambda df: df["Country"] == "England", key
        ].sum()
        for key in accom
    },
    "Wales": {
        key: ruc11_oa_broad_accom_type_summary.loc[
            lambda df: df["Country"] == "Wales", key
        ].sum()
        for key in accom
    },
    "Total": {
        key: ruc11_oa_broad_accom_type_summary.loc[
            lambda df: df["Country"] == "Total", key
        ].sum()
        for key in accom
    },
}

accom_props = {
    f"{key}_prop": makeFunc(
        f"df.apply(lambda x: round(x['{key}'] / accom_totals[x['Country']]['{key}'] * 100, 1), axis=1)"
    )
    for key in accom
}

ruc11_oa_broad_accom_type_summary = ruc11_oa_broad_accom_type_summary.assign(
    **accom_props
)
ruc11_oa_broad_accom_type_summary

# %%
ruc11_oa_accom_type_summary = pandas.concat(
    [
        ruc11_oa.groupby(
            [
                "Country",
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            {
                "Detached": "sum",
                "Semi-detached": "sum",
                "Terraced": "sum",
                "In a purpose-built block of flats or tenement": "sum",
                "Part of a converted or shared house, including bedsits": "sum",
                "Part of another converted building, for example, former school, church or warehouse": "sum",
                "In a commercial building, for example, in an office building, hotel or over a shop": "sum",
            }
        )
        .rename(columns={"Output Area 2011 Code": "OA11_Count"}),
        ruc11_oa.groupby(
            [
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            {
                "Detached": "sum",
                "Semi-detached": "sum",
                "Terraced": "sum",
                "In a purpose-built block of flats or tenement": "sum",
                "Part of a converted or shared house, including bedsits": "sum",
                "Part of another converted building, for example, former school, church or warehouse": "sum",
                "In a commercial building, for example, in an office building, hotel or over a shop": "sum",
            }
        )
        .rename(columns={"Output Area 2011 Code": "OA11_Count"})
        .assign(Country="Total"),
    ],
    ignore_index=True,
)


# %%
# This function correctly creates lambdas.
# Without it only the final one created in the loop is used.
def makeFunc(x):
    return lambda df: eval(x)


accom = [
    "Detached",
    "Semi-detached",
    "Terraced",
    "In a purpose-built block of flats or tenement",
    "Part of a converted or shared house, including bedsits",
    "Part of another converted building, for example, former school, church or warehouse",
    "In a commercial building, for example, in an office building, hotel or over a shop",
]

accom_totals = {
    "England": {
        key: ruc11_oa_accom_type_summary.loc[
            lambda df: df["Country"] == "England", key
        ].sum()
        for key in accom
    },
    "Wales": {
        key: ruc11_oa_accom_type_summary.loc[
            lambda df: df["Country"] == "Wales", key
        ].sum()
        for key in accom
    },
    "Total": {
        key: ruc11_oa_accom_type_summary.loc[
            lambda df: df["Country"] == "Total", key
        ].sum()
        for key in accom
    },
}

accom_props = {
    f"{key}_prop": makeFunc(
        f"df.apply(lambda x: round(x['{key}'] / accom_totals[x['Country']]['{key}'] * 100, 1), axis=1)"
    )
    for key in accom
}

ruc11_oa_accom_type_summary = ruc11_oa_accom_type_summary.assign(**accom_props)
ruc11_oa_accom_type_summary

# %% [markdown]
# ## Gardens
# ### All addresses

# %%
broad_gardens_total = pandas.concat(
    [
        ruc11_msoa.groupby(
            ["Country", "Rural Urban Classification 2011 (2 fold)"],
            observed=True,
            as_index=False,
        ).agg(
            {
                "Total: Percentage of addresses with private outdoor space": "mean",
                "Total: Average size of private outdoor space (m2)": "mean",
            }
        ),
        ruc11_msoa.groupby(
            "Rural Urban Classification 2011 (2 fold)", observed=True, as_index=False
        )
        .agg(
            {
                "Total: Percentage of addresses with private outdoor space": "mean",
                "Total: Average size of private outdoor space (m2)": "mean",
            }
        )
        .assign(Country="Total"),
    ],
    ignore_index=True,
)
broad_gardens_total

# %%
gardens_total = pandas.concat(
    [
        ruc11_msoa.groupby(
            [
                "Country",
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 code",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        ).agg(
            {
                "Total: Percentage of addresses with private outdoor space": "mean",
                "Total: Average size of private outdoor space (m2)": "mean",
            }
        ),
        ruc11_msoa.groupby(
            [
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 code",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            {
                "Total: Percentage of addresses with private outdoor space": "mean",
                "Total: Average size of private outdoor space (m2)": "mean",
            }
        )
        .assign(Country="Total"),
    ],
    ignore_index=True,
)

gardens_total

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

modes = {}
for setting in ["Urban", "Rural"]:
    z = gaussian_kde(
        ruc11_msoa.loc[
            lambda df: df["Rural Urban Classification 2011 (2 fold)"] == setting,
            "Total: Percentage of addresses with private outdoor space",
        ]
    ).evaluate(numpy.linspace(0, 1, 1001))
    ax.plot(numpy.linspace(0, 1, 1001), z, label=setting)
    modes[setting] = numpy.linspace(0, 1, 1001)[numpy.argmax(z)]

ax.set_xticks(numpy.linspace(0, 1, 11))
ax.legend()
ax.grid()
ax.set_ylabel("Density of Addresses")
ax.set_xlabel("Proportion of addresses with private outdoor space")

# %%
modes

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

colours = {
    "Urban major conurbation": "#08519c",
    "Urban minor conurbation": "#3182bd",
    "Urban city and town": "#6baed6",
    "Urban city and town in a sparse setting": "#bdd7e7",
    "Rural town and fringe": "#8c2d04",
    "Rural town and fringe in a sparse setting": "#d94801",
    "Rural village and dispersed": "#f16913",
    "Rural village and dispersed in a sparse setting": "#fd8d3c",
}

modes = {}
for setting in colours.keys():
    z = gaussian_kde(
        ruc11_msoa.loc[
            lambda df: df["Rural Urban Classification 2011 (10 fold)"] == setting,
            "Total: Percentage of addresses with private outdoor space",
        ]
    ).evaluate(numpy.linspace(0, 1, 1001))
    ax.plot(numpy.linspace(0, 1, 1001), z, label=setting, color=colours[setting])
    modes[setting] = numpy.linspace(0, 1, 1001)[numpy.argmax(z)]

ax.set_xticks(numpy.linspace(0, 1, 11))
ax.legend()
ax.grid()
ax.set_ylabel("Density of Addresses")
ax.set_xlabel("Proportion of addresses with private outdoor space")

# %%
lookup = {
    "Urban major conurbation": "Urban",
    "Urban minor conurbation": "Urban",
    "Urban city and town": "Urban",
    "Urban city and town in a sparse setting": "Urban",
    "Rural town and fringe": "Rural town and fringe",
    "Rural town and fringe in a sparse setting": "Rural town and fringe in a sparse setting",
    "Rural village and dispersed": "Rural village and dispersed",
    "Rural village and dispersed in a sparse setting": "Rural village and dispersed in a sparse setting",
}

ruc11_msoa["ruc_condense"] = ruc11_msoa[
    "Rural Urban Classification 2011 (10 fold)"
].map(lookup)

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

colours = {
    "Urban": "#3182bd",
    "Rural town and fringe": "#8c2d04",
    "Rural town and fringe in a sparse setting": "#d94801",
    "Rural village and dispersed": "#f16913",
    "Rural village and dispersed in a sparse setting": "#fd8d3c",
}

modes = {}
for setting in colours.keys():
    z = gaussian_kde(
        ruc11_msoa.loc[
            lambda df: df["ruc_condense"] == setting,
            "Total: Percentage of addresses with private outdoor space",
        ]
    ).evaluate(numpy.linspace(0, 1, 1001))
    ax.plot(numpy.linspace(0, 1, 1001), z, label=setting, color=colours[setting])
    modes[setting] = numpy.linspace(0, 1, 1001)[numpy.argmax(z)]

ax.set_xticks(numpy.linspace(0, 1, 11))
ax.legend()
ax.grid()
ax.set_ylabel("Density of Addresses")
ax.set_xlabel("Proportion of addresses with private outdoor space")

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

modes = {}
for setting in ["Urban", "Rural"]:
    z = gaussian_kde(
        ruc11_msoa.loc[
            lambda df: df["Rural Urban Classification 2011 (2 fold)"] == setting,
            "Total: Average size of private outdoor space (m2)",
        ]
    ).evaluate(numpy.linspace(0, 2_500, 1001))
    ax.plot(numpy.linspace(0, 2_500, 1001), z, label=setting)
    modes[setting] = numpy.linspace(0, 2_500, 1001)[numpy.argmax(z)]

ax.set_xticks(numpy.linspace(0, 2_500, 11))
ax.legend()
ax.grid()
ax.set_ylabel("Density of Addresses")
ax.set_xlabel("Average size of private outdoor space ($\mathrm{m}^{2}$)")

# %%
modes

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

colours = {
    "Urban major conurbation": "#08519c",
    "Urban minor conurbation": "#3182bd",
    "Urban city and town": "#6baed6",
    "Urban city and town in a sparse setting": "#bdd7e7",
    "Rural town and fringe": "#8c2d04",
    "Rural town and fringe in a sparse setting": "#d94801",
    "Rural village and dispersed": "#f16913",
    "Rural village and dispersed in a sparse setting": "#fd8d3c",
}

modes = {}
for setting in colours.keys():
    z = gaussian_kde(
        ruc11_msoa.loc[
            lambda df: df["Rural Urban Classification 2011 (10 fold)"] == setting,
            "Total: Average size of private outdoor space (m2)",
        ]
    ).evaluate(numpy.linspace(0, 2_500, 1001))
    ax.plot(numpy.linspace(0, 2_500, 1001), z, label=setting, color=colours[setting])
    modes[setting] = numpy.linspace(0, 2_500, 1001)[numpy.argmax(z)]

ax.set_xticks(numpy.linspace(0, 2_500, 11))
ax.legend()
ax.grid()
ax.set_ylabel("Density of Addresses")
ax.set_xlabel("Average size of private outdoor space ($\mathrm{m}^{2}$)")

# %% [markdown]
# ### Houses

# %%
broad_gardens_house = pandas.concat(
    [
        ruc11_msoa.groupby(
            ["Country", "Rural Urban Classification 2011 (2 fold)"],
            observed=True,
            as_index=False,
        ).agg(
            {
                "House: Percentage of addresses with private outdoor space": "mean",
                "House: Average size of private outdoor space (m2)": "mean",
                "House: Median size of private outdoor space (m2)": "mean",
            }
        ),
        ruc11_msoa.groupby(
            "Rural Urban Classification 2011 (2 fold)", observed=True, as_index=False
        )
        .agg(
            {
                "House: Percentage of addresses with private outdoor space": "mean",
                "House: Average size of private outdoor space (m2)": "mean",
                "House: Median size of private outdoor space (m2)": "mean",
            }
        )
        .assign(Country="Total"),
    ],
    ignore_index=True,
)
broad_gardens_house

# %%
gardens_house = pandas.concat(
    [
        ruc11_msoa.groupby(
            [
                "Country",
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 code",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        ).agg(
            {
                "House: Percentage of addresses with private outdoor space": "mean",
                "House: Average size of private outdoor space (m2)": "mean",
                "House: Median size of private outdoor space (m2)": "mean",
            }
        ),
        ruc11_msoa.groupby(
            [
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 code",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            {
                "House: Percentage of addresses with private outdoor space": "mean",
                "House: Average size of private outdoor space (m2)": "mean",
                "House: Median size of private outdoor space (m2)": "mean",
            }
        )
        .assign(Country="Total"),
    ],
    ignore_index=True,
)
gardens_house

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

for setting in ["Urban", "Rural"]:
    z = gaussian_kde(
        ruc11_msoa.loc[
            lambda df: df["Rural Urban Classification 2011 (2 fold)"] == setting,
            "House: Percentage of addresses with private outdoor space",
        ]
    ).evaluate(numpy.linspace(0, 1, 1001))
    ax.plot(numpy.linspace(0, 1, 1001), z, label=setting)

ax.set_xticks(numpy.linspace(0, 1, 11))
ax.legend()
ax.grid()
ax.set_ylabel("Density of Addresses")
ax.set_xlabel("Proportion of house addresses with private outdoor space")

# %%
f, ax = pyplot.subplots(figsize=(8, 5))

for setting in ["Urban", "Rural"]:
    z = gaussian_kde(
        ruc11_msoa.loc[
            lambda df: df["Rural Urban Classification 2011 (2 fold)"] == setting,
            "House: Average size of private outdoor space (m2)",
        ]
    ).evaluate(numpy.linspace(0, 2_500, 1001))
    ax.plot(numpy.linspace(0, 2_500, 1001), z, label=setting)

ax.set_xticks(numpy.linspace(0, 2_500, 11))
ax.legend()
ax.grid()
ax.set_ylabel("Density of Addresses")
ax.set_xlabel("Average size of private outdoor space ($\mathrm{m}^{2}$)")

# %% [markdown]
# ### Flats

# %%
broad_gardens_flat = pandas.concat(
    [
        ruc11_msoa.groupby(
            ["Country", "Rural Urban Classification 2011 (2 fold)"],
            observed=True,
            as_index=False,
        ).agg(
            {
                "Flat: Percentage of addresses with private outdoor space": "mean",
                "Flat: Average size of private outdoor space (m2)": "mean",
                "Flat: Average number of flats sharing a garden": "mean",
            }
        ),
        ruc11_msoa.groupby(
            "Rural Urban Classification 2011 (2 fold)", observed=True, as_index=False
        )
        .agg(
            {
                "Flat: Percentage of addresses with private outdoor space": "mean",
                "Flat: Average size of private outdoor space (m2)": "mean",
                "Flat: Average number of flats sharing a garden": "mean",
            }
        )
        .assign(Country="Total"),
    ],
    ignore_index=True,
)
broad_gardens_flat

# %%
gardens_flat = pandas.concat(
    [
        ruc11_msoa.groupby(
            [
                "Country",
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 code",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        ).agg(
            {
                "Flat: Percentage of addresses with private outdoor space": "mean",
                "Flat: Average size of private outdoor space (m2)": "mean",
                "Flat: Average number of flats sharing a garden": "mean",
            }
        ),
        ruc11_msoa.groupby(
            [
                "Rural Urban Classification 2011 (2 fold)",
                "Rural Urban Classification 2011 code",
                "Rural Urban Classification 2011 (10 fold)",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            {
                "Flat: Percentage of addresses with private outdoor space": "mean",
                "Flat: Average size of private outdoor space (m2)": "mean",
                "Flat: Average number of flats sharing a garden": "mean",
            }
        )
        .assign(Country="Total"),
    ],
    ignore_index=True,
)
gardens_flat

# %% [markdown]
# ### Save Summaries to Output Spreadsheet

# %%
with pandas.ExcelWriter(
    "../../outputs/data/20240219_rural_urban_summary.xlsx"
) as writer:
    ruc11_oa_broad_summary.to_excel(
        writer, sheet_name="Broad Summary OA21 Household", index=False
    )
    ruc11_oa_summary.to_excel(
        writer, sheet_name="Full Summary OA21 Household", index=False
    )
    ruc11_oa_broad_accom_type_summary.to_excel(
        writer, sheet_name="Broad Summary OA21 Accom. Type", index=False
    )
    ruc11_oa_accom_type_summary.to_excel(
        writer, sheet_name="Full Summary OA21 Accom. Type", index=False
    )
    broad_gardens_total.to_excel(
        writer, sheet_name="Broad Summary MSOA11 Gardens", index=False
    )
    gardens_total.to_excel(
        writer, sheet_name="Full Summary MSOA11 Gardens", index=False
    )
    broad_gardens_house.to_excel(
        writer, sheet_name="Broad Houses MSOA11 Gardens", index=False
    )
    gardens_house.to_excel(writer, sheet_name="Full Houses MSOA11 Gardens", index=False)
    broad_gardens_flat.to_excel(
        writer, sheet_name="Broad Flats MSOA11 Gardens", index=False
    )
    gardens_flat.to_excel(writer, sheet_name="Full Flats MSOA11 Gardens", index=False)

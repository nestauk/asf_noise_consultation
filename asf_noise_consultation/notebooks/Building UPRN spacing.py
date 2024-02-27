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
from dask import dataframe as dd
import dask_geopandas
from pyogrio import read_info, read_dataframe
import pandas
import pickle
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from matplotlib import pyplot
import numpy

# %% [markdown]
# ## Aim
#
# Identify building UPRNS and calculate average nearest neighbour distances within 2011 OAs.
#
# Using [OS Open Map Local](https://www.ordnancesurvey.co.uk/products/os-open-map-local) to identify building UPRNs and [OS Open UPRN](https://www.ordnancesurvey.co.uk/products/os-open-uprn) as the source of UPRNS.
#
# Building feature codes are: 15014 - Building 15016 - Glasshouse 15018 - Air Transport 15019 - Education 15020 - Medical Care 15021 - Road Transport 15022 - Water Transport 15023 - Emergency Service 15024 - Cultural Facility 15025 - Religious Buildings 15026 - Retail 15027 - Sports Or Exercise Facility 15028 - Attraction And Leisure
#
# We'll exclude the codes that ary unlikely to occur in mixed use with residential, namely: glasshouse, air transport, education, road transport, water transport, emergency service, religious buildings, medical care.

# %%
filepath = "../../../asf_kensa_suitability_prototype/inputs/data/os_openmap_local/opmplc_gb.gpkg"
uprn_path = "../../../asf_kensa_suitability_prototype/inputs/data/uprn/osopenuprn_202401_csv/osopenuprn_202312.csv"

# %%
# Load uprns from file
uprn_df = dd.read_csv(uprn_path)

# %%
# Convert dask dataframe to geodataframe - defaults to 31 partitions from uprn_df.
# Now we have an initial set of UPRNs to filter down to building uprns
uprn_gdf = dask_geopandas.from_dask_dataframe(
    uprn_df,
    geometry=dask_geopandas.points_from_xy(uprn_df, "X_COORDINATE", "Y_COORDINATE"),
)
uprn_gdf = uprn_gdf.set_crs("epsg:27700")

# %%
# Iterate over buildings in chunks and create a new dataset
# of just those UPRNs that fall within a building polygon.

building_uprns = []

for skip in range(0, 18_000_000, 500_000):
    buildings = read_dataframe(
        filepath,
        layer="building",
        skip_features=skip,
        max_features=500_000,
        where="feature_code in (15014, 15024, 15026, 15027, 15028)",
    )

    buildings = dask_geopandas.from_geopandas(buildings, npartitions=10)

    building_uprns.append(
        dask_geopandas.sjoin(
            uprn_gdf, buildings, how="inner", predicate="within"
        ).compute()["UPRN"]
    )
    print(skip + 500_000)
    del buildings

building_uprns = pandas.concat(building_uprns, ignore_index=True)

# save for later use
building_uprns.to_csv("./building_uprns.csv")

# %%
# Load building uprns if needed.
uprns = pandas.read_csv("./building_uprns.csv", index_col=0)

# %%
# Filter original uprn dataset to just the building uprns
uprn_gdf = uprn_gdf.loc[lambda df: df["UPRN"].isin(uprns["UPRN"]), :].compute()

# %%
# Save filtered uprns if needed.
uprn_df.to_csv("./filtered_uprns.csv")

# %% [markdown]
# ## Get OA neighbours for sampling
#
# To calculate the address spacing from UPRNs we need to measure the distance from each UPRN to its nearest neighbour. Doing this in one shot probably isn't possible due to the data volume, so we'll do it on an OA by OA basis. For each OA we'll look at the UPRNs within that OA and the UPRNs in the 1st order Queen's case neighbours (contiguous OAs) to avoid introducing edge effects (e.g. where the nearest UPRN neighbour is actually in an adjoining OA). We'll then take the mean of UPRNs in the candidate OA. We anticipate that flats will produce 0m nearest neighbour distances as UPRNs tend to represent delivery points or buildign centroids which are common for flats in single blocks or subdivided houses. This means that the average address space is likely to be lower than the average building spacing which could alternatively be computed by deduplciating UPRNs based on their easting and northing.

# %%
oa_2011 = read_dataframe(
    "../../inputs/data/Output_Areas_Dec_2011_Boundaries_EW_BFC_2022_-4439794289771318577.gpkg",
    columns=["OA11CD"],
)

# %%
# Get each OAs first order neighbours - NB tried libpysal for this but experienced memory errors.
# Takes a while, understandably.
neighbours = {}

for idx, oa in oa_2011.iterrows():
    cand = oa["geometry"]
    name = oa["OA11CD"]
    # Nb we want to include the candidate oa in the neighbours for convenience.
    neighbours[name] = oa_2011.loc[~oa_2011.geometry.disjoint(cand), "OA11CD"].to_list()
    if idx % 10_000 == 0:
        print(idx)

# %%
# Store output for future use.
with open("./neighbours.pkl", "wb") as handle:
    pickle.dump(neighbours, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open("./neighbours.pkl", "rb") as handle:
    neighbours = pickle.load(handle)

# %% [markdown]
# Here I want to tag each UPRN with the 2011 OA it falls in (existing lookups only has the 2021 OA). This should make it easier to select the relevant rows when deriving the mean nearest neighbours later on.

# %%
# Again too memory intensive to do in one hit.
uprn_oa2011 = []

for skip in range(0, 200_000, 20_000):
    oa_2011 = read_dataframe(
        "../../inputs/data/Output_Areas_Dec_2011_Boundaries_EW_BFC_2022_-4439794289771318577.gpkg",
        skip_features=skip,
        max_features=20_000,
        columns=["OA11CD"],
    )

    oa_2011 = dask_geopandas.from_geopandas(oa_2011, npartitions=10)

    uprn_oa2011.append(
        dask_geopandas.sjoin(
            uprn_gdf, oa_2011, how="inner", predicate="within"
        ).compute()[["UPRN", "X_COORDINATE", "Y_COORDINATE", "OA11CD"]]
    )
    print(skip + 20_000)
    del oa_2011

# %%
uprn_oa2011 = pandas.concat(uprn_oa2011, ignore_index=True)

# %%
# Store for later if needed.
uprn_oa2011.to_csv("./uprn_oa2011.csv", index=False)

# %%
# Read with optimal datatypes.
# Treating the OA11CD as a categorical, as opposed to a string or setting it as the dataframe index
# appears to give the best filtering performance, which is important given the volume of filtering required.
uprn_oa2011 = pandas.read_csv(
    "./uprn_oa2011.csv",
    dtype={
        "UPRN": "int64",
        "X_COORDINATE": "float32",
        "Y_COORDINATE": "float32",
        "OA11CD": "category",
    },
)

# %%
# This is the effective algorithm.
# For each candidate OA, construct a cKDTree from the UPRNs of that OA and its neighbours.
# for each UPRN in the candidate OA, get the distance of its nearest neighbour
# In this case, this is the second nearest point as the first is itself
# Take the mean of the distances for all UPRNs in the OA and record.
# Move to next OA in list.

oa_nn = {}
for idx, (cand_oa, oas) in enumerate(neighbours.items()):
    # get participating uprns by oas
    uprns = uprn_oa2011.loc[lambda df: df["OA11CD"] == cand_oa]
    test_uprns = uprn_oa2011.loc[lambda df: df["OA11CD"].isin(oas)]

    tree = cKDTree(test_uprns[["X_COORDINATE", "Y_COORDINATE"]].values)
    # k=2 as otherwhile you're measuring the distance to yourself.
    mean_address_distance = tree.query(
        uprns[["X_COORDINATE", "Y_COORDINATE"]].values, k=2
    )[0][:, 1].mean()
    oa_nn[cand_oa] = mean_address_distance
    if idx % 1000 == 0:
        print(idx)


# %%
with open("./oa11cd_nn.pkl", "wb") as handle:
    pickle.dump(oa_nn, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open("./oa11cd_nn.pkl", "rb") as handle:
    oa_nn = pickle.load(handle)

# %%
nns = (
    pandas.Series(oa_nn, name="Mean Nearest Neighbour")
    .reset_index(drop=False)
    .rename(columns={"index": "OA11CD"})
)

# %% [markdown]
# There are 47 OAs that produce NA results for mean nearest neighbour. As this is 0.03% of OAs I'm not particularly concerned by their prescence.
#
# The NAs in question are all English. Looking at a few of them seems to indicate that they might be places that are undergoing regeneration activities and may not have active UPRNs. This could reflect a mismatch between contemporary UPRN and building footprint data and 2011 OAs.

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
(
    ruc11_oa.merge(nns, left_on="Output Area 2011 Code", right_on="OA11CD")
    .groupby("Rural Urban Classification 2011 (2 fold)", observed=False)[
        "Mean Nearest Neighbour"
    ]
    .median()
)

# %%
(
    ruc11_oa.merge(nns, left_on="Output Area 2011 Code", right_on="OA11CD")
    .groupby("Rural Urban Classification 2011 (10 fold)", observed=False)[
        "Mean Nearest Neighbour"
    ]
    .mean()
)

# %% [markdown]
# Having looked at the means, I felt that they were somewhat influenced by outlier (including particularly large means in some rural OAs), so I decided to report the modes derived from the kernel density estimates of the distributions below.

# %%
f, ax = pyplot.subplots(figsize=(8, 5))
modes = {}
for setting in ["Urban", "Rural"]:
    z = gaussian_kde(
        (
            ruc11_oa.merge(nns, left_on="Output Area 2011 Code", right_on="OA11CD")
            .loc[
                lambda df: df["Rural Urban Classification 2011 (2 fold)"] == setting,
                "Mean Nearest Neighbour",
            ]
            .fillna(0)
        )
    ).evaluate(numpy.linspace(0, 80, 801))
    modal_value = numpy.linspace(0, 80, 801)[numpy.argmax(z)]
    modes[setting] = modal_value
    ax.plot(numpy.linspace(0, 80, 801), z, label=setting)

ax.legend()
ax.set_xlabel("Mean Distance to Nearest Neighbour UPRN (m)")
ax.set_ylabel("Density")

# %%
modes

# %%
f, ax = pyplot.subplots(figsize=(9, 5))

colours = {
    "Urban major conurbation": "#08519c",
    "Urban minor conurbation": "#3182bd",
    "Urban city and town": "#6baed6",
    "Urban city and town in a sparse setting": "#bdd7e7",
    "Rural town and fringe": "#8c2d04",
    "Rural town and fringe in a sparse setting": "#d94801",
    "Rural village": "#f16913",
    "Rural village in a sparse setting": "#fd8d3c",
    "Rural hamlets and isolated dwellings": "#fdae6b",
    "Rural hamlets and isolated dwellings in a sparse setting": "#fdd0a2",
}

modes = {}
for setting in colours.keys():
    z = gaussian_kde(
        (
            ruc11_oa.merge(nns, left_on="Output Area 2011 Code", right_on="OA11CD")
            .loc[
                lambda df: df["Rural Urban Classification 2011 (10 fold)"] == setting,
                "Mean Nearest Neighbour",
            ]
            .fillna(0)
        )
    ).evaluate(numpy.linspace(0, 100, 1001))
    modal_value = numpy.linspace(0, 100, 1001)[numpy.argmax(z)]
    modes[setting] = modal_value
    ax.plot(numpy.linspace(0, 100, 1001), z, label=setting, color=colours[setting])

ax.legend(fontsize=9)
ax.set_xlabel("Mean Distance to Nearest Neighbour UPRN (m)")
ax.set_ylabel("Density")

# %%
modes

# %%
lookup = {
    "Urban major conurbation": "Urban",
    "Urban minor conurbation": "Urban",
    "Urban city and town": "Urban",
    "Urban city and town in a sparse setting": "Urban",
    "Rural town and fringe": "Rural town and fringe",
    "Rural town and fringe in a sparse setting": "Rural town and fringe in a sparse setting",
    "Rural village": "Rural village",
    "Rural village in a sparse setting": "Rural village in a sparse setting",
    "Rural hamlets and isolated dwellings": "Rural hamlets and isolated dwellings",
    "Rural hamlets and isolated dwellings in a sparse setting": "Rural hamlets and isolated dwellings in a sparse setting",
}

ruc11_oa["condensed_ru11"] = ruc11_oa["Rural Urban Classification 2011 (10 fold)"].map(
    lookup
)

# %%
## Collapse Urban classes for clarity

f, ax = pyplot.subplots(figsize=(9, 5))

colours = {
    "Urban": "#3182bd",
    "Rural town and fringe": "#8c2d04",
    "Rural town and fringe in a sparse setting": "#d94801",
    "Rural village": "#f16913",
    "Rural village in a sparse setting": "#fd8d3c",
    "Rural hamlets and isolated dwellings": "#fdae6b",
    "Rural hamlets and isolated dwellings in a sparse setting": "#fdd0a2",
}

modes = {}
for setting in colours.keys():
    z = gaussian_kde(
        (
            ruc11_oa.merge(nns, left_on="Output Area 2011 Code", right_on="OA11CD")
            .loc[
                lambda df: df["condensed_ru11"] == setting,
                "Mean Nearest Neighbour",
            ]
            .fillna(0)
        )
    ).evaluate(numpy.linspace(0, 100, 1001))
    modal_value = numpy.linspace(0, 100, 1001)[numpy.argmax(z)]
    modes[setting] = modal_value
    ax.plot(
        numpy.linspace(0, 100, 1001),
        z,
        label="All Urban" if setting == "Urban" else setting,
        color=colours[setting],
    )

ax.legend(fontsize=9)
ax.set_xlabel("Mean Distance to Nearest Neighbour UPRN (m)")
ax.set_ylabel("Density")

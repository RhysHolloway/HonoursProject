import models
from models import Column, Dataset
import os.path

import pandas as pd

# Get root project path
def get_project_path(path: str):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), path).replace("\\", "/")
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
            os.makedirs(dir)
    return path

def data_path(file: str) -> str:
    return get_project_path(f"data/{file}")

output = get_project_path("output/plots")

# Define datasets
Lisiecki = Dataset.load(
    name="Lisiecki (2005)",
    df=lambda: pd.read_csv(data_path("lisiecki2005-d18o-stack-noaa.csv"), comment='#', header=0, sep='\t'),
    age_col="age_calkaBP",
    feature_cols={
        "d18O_benthic" : "benthic d18O records",
        # "d18O_error" : "error"
    },
    age_scale=1000,
)

Scotese = Dataset.load(
    name="Scotese et al. (2021)",
    df=lambda: pd.read_excel(data_path("Part 4. Phanerozoic_Paleotemperature_Summaryv4.xlsx"), sheet_name="Master", header=[0,1]),
    age_col=("Age", 'Unnamed: 0_level_1'),
    feature_cols={
        ("Average", "Tropical"): "Tropical Temperature",
        ("Average", "Deep Ocean"): "Deep Ocean Temperature", 
        # ("Average", "∆T trop"),
        ("North", "Polar >67˚N"): "North Polar Temperature", 
        ("South", "Polar <67˚S"): "South Polar Temperature"
    },
    age_scale=1000000,
)

_GMST: dict[Column, str] = {
    "GMST_05": "5%",
    "GMST_16": "16%",
    "GMST_50": "50%",
    "GMST_84": "84%",
    "GMST_95": "95%",
}

_CO2: dict[Column, str] = {
    "CO2_05": "5%",
    "CO2_16": "16%",
    "CO2_50": "50%",
    "CO2_84": "84%",
    "CO2_95": "95%",
}

Judd = Dataset.load(
    name="Judd et. al. (2024)",
    df=lambda: pd.read_csv(data_path("PhanDA_GMSTandCO2_percentiles.csv")),
    age_col="AverageAge",
    feature_cols=_GMST | _CO2,
    age_scale=1000000,    
)

_JuddSplit = Judd.split({
    "GMST Confidence": _GMST,
    "CO2 Confidence": _CO2,
})

JuddGMST, JuddCO2 = _JuddSplit["GMST Confidence"], _JuddSplit["CO2 Confidence"]

BuryEndGreenhouseEarth = Dataset.load(
    name="Bury paleoclimate - End of greenhouse Earth",
    df=lambda: pd.read_csv(data_path("bury_paleoclimate/tripati2005/tripati2005_select.csv"), encoding="utf-8-sig"),
    age_col="Age",
    feature_cols={"CaCO3": "CaCO3 (%)"},
    age_scale=1_000_000
).age_range(32_000_000, 40_000_000)

BuryBollingAllerod = Dataset.load(
    name="Bury paleoclimate - Bolling-Allerod transition",
    df=lambda: pd.read_csv(
        data_path("bury_paleoclimate/gisp2/gisp2_temp_accum_alley2000.txt"),
        header=0,
        names=["Age", "Temperature", "NA"],
        sep=r"\s+",
        nrows=1632,
    ),
    age_col="Age",
    feature_cols={"Temperature": "Temperature (C)"},
    age_scale=1000,
).age_range(14_600, 21_000)

BuryEndYoungerDryas = Dataset.load(
    name="Bury paleoclimate - End of Younger Dryas",
    df=lambda: pd.read_csv(
        data_path("bury_paleoclimate/cariaco2000/cariaco2000_pc56_greyscale.txt"),
        header=1,
        names=["Age", "Grayscale"],
        sep=r"\s+",
    ),
    age_col="Age",
    feature_cols={"Grayscale": "Grayscale (0-255)"},
).age_range(11_200, 12_500)

BuryDesertificationNorthAfrica = Dataset.load(
    name="Bury paleoclimate - Desertification of N. Africa",
    df=lambda: pd.read_csv(data_path("bury_paleoclimate/demenocal2000/658C.terr.2.1.interp.csv"), encoding="utf-8-sig"),
    age_col="Age(cal. yr BP)",
    feature_cols={"terr% (interp)": "Terrigenous dust (%)"},
).age_range(4800, 8300)

_BuryDeutnat = Dataset.load(
    name="Petit, J.R., et al., 2001",
    df=lambda: pd.read_csv(
        data_path("bury_paleoclimate/deutnat/deutnat.txt"),
        sep=r"\s+",
        encoding="latin1",
        names=["i", "Age", "d2H", "deltaTS"],
        skiprows=range(0, 111),
    ),
    age_col="Age",
    feature_cols={"d2H": "d2H (%)"}
)

BuryEndGlaciationI = _BuryDeutnat.age_range(12_000, 58_000)
BuryEndGlaciationI.name = "Bury paleoclimate - End of glaciation I"

BuryEndGlaciationII = _BuryDeutnat.age_range(128_000, 151_000)
BuryEndGlaciationII.name = "Bury paleoclimate - End of glaciation II"

BuryEndGlaciationIII = _BuryDeutnat.age_range(238_000, 270_000)
BuryEndGlaciationIII.name = "Bury paleoclimate - End of glaciation III"

BuryEndGlaciationIV = _BuryDeutnat.age_range(324_600, 385_300)
BuryEndGlaciationIV.name = "Bury paleoclimate - End of glaciation IV"

# Dataset, Gaussian detrending bandwidth, and transition age from the
# paleoclimate empirical tests in Bury et al.'s PNAS repository.
BuryPaleoclimate = [
    (BuryEndGreenhouseEarth, 25, [34_000_000]),
    (BuryBollingAllerod, 25, [15_000]),
    (BuryEndYoungerDryas, 100, [11_600]),
    (BuryDesertificationNorthAfrica, 10, [5700]),
    (BuryEndGlaciationI, 25, [17_000]),
    (BuryEndGlaciationII, 25, [135_000]),
    (BuryEndGlaciationIII, 10, [242_000]),
    (BuryEndGlaciationIV, 50, [334_100]),
]

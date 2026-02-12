import models
from models import Dataset
from models.lstm import LSTM, LSTMLoader
from models.metrics import Metrics
import os.path

import pandas as pd

# Get root project path
def get_project_path(path: str):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), path).replace("\\", "/")
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
).transform(models.resample_df)

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
).transform(lambda df, features: models.resample_df(df, features, steps=0.5))

GMST = [
    "GMST_05", "GMST_95",
    "GMST_16", "GMST_84",
    "GMST_50"
]

CO2 = [
    "CO2_05", "CO2_95",
    "CO2_16", "CO2_84",
    "CO2_50"
]

JuddGMST, JuddCO2 = Dataset.load(
    name="Judd et. al. (2024)",
    df=lambda: pd.read_csv(data_path("PhanDA_GMSTandCO2_percentiles.csv")),
    age_col="AverageAge",
    feature_cols=GMST + CO2,
    age_scale=1000000,    
).transform(lambda df, features: models.resample_df(df, features, steps=2)).split({
    "GMST": GMST,
    "CO2": CO2,
})  
      
# Foster = Dataset(
#     name="Foster, G. L., et al. (2017)",
#     df=lambda: pd.read_excel(data_path("41467_2017_BFncomms14845_MOESM2874_ESM.xlsx"), sheet_name="proxies", header=[1,2,3]),
#     age_col=('Age', '(Ma)'),
#     feature_cols=[
#         'Î´13c_permille', 'select_d13c_permille', 
#         'sauerstof_isotop_permille', 
#         'select_d18o_permille (arag-0.6)', 
#         "T (-1.08‰) arag, bel corr"
#     ],
#     transform=lambda df, age_col, feature_cols: resample_df(df, age_col, feature_cols, steps=2),
#     age_scale=1000000,
# )

# Run models on datasets

# Metrics(window = 0.2).run_with_output([Lisiecki, Scotese, Judd], plot_path=output)
# DLM().run_with_output([Lisiecki, Scotese, Judd], plot_path=output)
# HMM().run_with_output([Lisiecki, Scotese, Judd], plot_path=output)

loader = LSTMLoader(get_project_path("models/lstm/"), extension="h5")
LSTM = loader.with_args(verbose=False)

LSTM.run_with_output([Scotese, JuddGMST, JuddCO2], print=False, plot_path=output)
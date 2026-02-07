from models import *
from processing import Dataset
from util import resample_df, get_project_path, pd

def data_path(file: str):
    return get_project_path(f"data/{file}")

# Define datasets
 
Lisiecki = Dataset.load(
    name="Lisiecki (2005)",
    df=lambda: pd.read_csv(data_path("lisiecki2005-d18o-stack-noaa.csv"), comment='#', header=0, sep='\t'),
    age_col="age_calkaBP",
    feature_cols={
        "d18O_benthic" : "benthic d18O records",
        # "d18O_error" : "error"
    },
    transform=lambda df, age_col, feature_cols: resample_df(df, age_col, feature_cols, steps=1),
    age_scale=1000,
)

Scotese = Dataset.load(
    name="Scotese et al. (2021)",
    df=lambda: pd.read_excel(data_path("Part 4. Phanerozoic_Paleotemperature_Summaryv4.xlsx"), sheet_name="Master", header=[0,1]),
    age_col=("Age", 'Unnamed: 0_level_1'),
    feature_cols=[
        ("Average", "Tropical"),
        ("Average", "Deep Ocean"), 
        # ("Average", "∆T trop"),
        ("North", "Polar >67˚N"), 
        ("South", "Polar <67˚S")
    ],
    transform=lambda df, age_col, feature_cols: resample_df(df, age_col, feature_cols, steps=0.5),
    age_scale=1000000,
)

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
    transform=lambda df, age_col, feature_cols: resample_df(df, age_col, feature_cols, steps=2),
    age_scale=1000000,    
).split({
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

# Metrics(window = 0.2).run([Lisiecki, Scotese, Judd])
# DLM().run([Lisiecki, Scotese, Judd])
# HMM().run([Lisiecki, Scotese, Judd])

loader = LSTMLoader(get_project_path("models/lstm/"), extension="h5")
LSTM = loader.with_args(verbose=False)

LSTM.run([Scotese, JuddGMST, JuddCO2])
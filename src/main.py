from util import resample_df, get_project_path, pd

from model import run_models_on_data
from lstm import lstm
from hmm import hmm

data_path = lambda file: get_project_path(f"../data/{file}")
 
# run_models_on_data(
#     name="Lisiecki (2005)",
#     df=lambda: pd.read_csv(data_path("lisiecki2005-d18o-stack-noaa.csv"), comment='#', header=0, sep='\t'),
#     age_col="age_calkaBP",
#     feature_cols=["d18O_benthic", "d18O_error"],
#     transform=lambda df, age_col, feature_cols: resample_df(df, age_col, feature_cols, steps=1),
#     age_format="kya",
#     models=[
#         hmm(
#             n_regimes=5, # Lowest BIC
#             # min_state_duration=50,
#             # min_duration_between_switches=100,
#             p_threshold=0.95
#         ),
#         lstm(
#             seq_len=16,
#             train=0.4,
#             threshold=[90],
#             epochs=300,
#             smoothing_window=10,
#             distance=100,
#         )
#     ],
# )

run_models_on_data(
    name="Scotese et al. (2021)",
    df=lambda: pd.read_excel(data_path("Part 4. Phanerozoic_Paleotemperature_Summaryv4.xlsx"), sheet_name="Master", header=[0,1]),
    age_col=("Age", 'Unnamed: 0_level_1'),
    feature_cols=[
        ("Average", "Tropical"),
        ("Average", "Deep Ocean"), 
        ("Average", "∆T trop"),
        ("North", "Polar >67˚N"), 
        ("South", "Polar <67˚S")
    ],
    models = [
        # hmm(
        #     min_covar=1e-4,
        #     p_threshold=0.7,
        #     min_state_duration=30,
        #     min_duration_between_switches=50,
        # ),
        lstm(
            seq_len=16,
            train=0.4,
            threshold=[.925, .95, .99],
            epochs=300
        ),
    ]
)
    
# run_models_on_data(
#     name="Judd et. al. (2024)",
#     df=lambda: pd.read_csv(data_path("PhanDA_GMSTandCO2_percentiles.csv")),
#     age_col="AverageAge",
#     feature_cols=[
#         "GMST_05", "GMST_95",
#         "GMST_16", "GMST_84",
#         "GMST_50",
#         "CO2_05", "CO2_95",
#         "CO2_16", "CO2_84",
#         "CO2_50",
#     ],
#     transform=lambda df, age_col, feature_cols: resample_df(df, age_col, feature_cols, steps=2),
#     models=[
#         hmm(
#             n_regimes=3
#         ),
#         lstm(
#             seq_len=16,
#             train=0.35,
#             threshold=95,
#             epochs=400,
#             patience=35,
#             distance=30,
#         ),
#     ]
# )    
      
# run_models_on_data(
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
#     models=[
#         hmm(),
#         lstm(
#             seq_len=32,
#             train=0.4,
#             threshold=99,
#             epochs=150,
#             patience=30,            
#         ),
#     ],
# )

# def PhanSST():

#     existing_cols = [
#         # "MgCa", "SrCa", "MnSr",
#         # "GDGT0", "GDGT1", "GDGT2", "GDGT3",
#         # "BIT", "MI"
#     ]
    
#     def transform(df, age_col, feature_cols): 
#             df = df.pivot_table(index=[age_col], columns='ProxyType', values='ProxyValue', aggfunc="mean").join(
#                 df.set_index([age_col])
#                 [existing_cols]
#                 .drop_duplicates()
#             ).reset_index()
#             df = prepare_df(df, age_col, feature_cols)
#             return resample_df(df, age_col, feature_cols, step_years=0.1)

#     detect_tipping_point(
#         name="PhanSST",
#         df=lambda: pd.read_csv(os.path.join(DATA_PATH, "PhanSST_v001.csv")),
#         model=hmm_tipping_points(),
#         # model=lstm_tipping_points(
#         #     seq_len=30,
#         #     train_fraction=0.4,
#         #     threshold=95,
#         #     latent_dim=32,
#         #     batch_size=16,
#         #     distance=10,
#         #     epochs=200,
#         #     patience=35,
#         # ),
#         transform=transform,
#         age_col='Age',
#         feature_cols=existing_cols + [
#             "d18c", "d18p", "mg", "tex", "uk",
#             # "ProxyValue"
#         ],
#         verbose=True
#     )
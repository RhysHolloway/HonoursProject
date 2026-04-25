if __name__ == "__main__":
    print("Importing environment...")
    import os
    from models.lstm import LSTMLoader
    from models.datasets import *

    import models.lstm.test as test
    
    # Run LSTM model on new datasets and generate figures
    
    lstm = LSTMLoader(get_project_path("bury_models/"))
    
    PLOT_OUTPUT = get_project_path("output/plots")
    
    # lstm.with_args(verbose=True, window=35).run_with_output([Lisiecki], PLOT_OUTPUT, title=False)
    lstm.with_args(window=120).run_with_output([BuryEndYoungerDryas2], PLOT_OUTPUT, title=False)
    
    print("Generating project figures from data...")
    test.load_and_save(
        metrics_path = get_project_path("output/metrics/"),
        output = os.path.join(PLOT_OUTPUT),
        lstm=lstm.with_args(name="LSTM"),
        models=\
            test.test_models(set([1500, 500])) +
            [
                DatasetModel(
                    Lisiecki,
                    Lisiecki.df.columns[0],
                    2_600_000,
                    35,
                )
            ] +
            BuryPaleoclimate,
    )

if __name__ == "__main__":
    print("Importing environment and loading datasets...")
    import os
    from .lstm import LSTMLoader, test
    from .metrics import Metrics
    from . import get_project_path
    from .datasets import *
    
    # Run LSTM model on new datasets and generate figures
    
    # Change to output/models/lstm/ for generated models
    lstm = LSTMLoader(get_project_path("data/bury_models/"))
    
    PLOT_OUTPUT = get_project_path("output/plots")
    
    BuryEndYoungerDryasTransition = BuryEndYoungerDryas.age_range(min=11_600)
    
    lstm.with_args(span=100, detrend="Gaussian").run_with_output(
        [BuryEndYoungerDryasTransition], 
        transitions=[11_600],
        path=PLOT_OUTPUT, title=False)
    
    YoungerDryasLowess = [
        BuryEndYoungerDryasTransition.rename(BuryEndYoungerDryasTransition.name + " LOWESS"),
        BuryEndYoungerDryasTransition.age_range(max=11_900, name = BuryEndYoungerDryas.name + " (11.9kya-11.6kya)"),
    ]
    
    LisieckiTest = [
        Lisiecki, 
        Lisiecki.age_range(1_600_000, 2_700_000, name="Lisiecki (2005) 2.7mya-1.6mya"), 
        Lisiecki.age_range(0, 2_700_000, name="Lisiecki (2005) 2.7mya-0ya"), 
    ]
    
    for datasets, span, transitions in [
        (YoungerDryasLowess, 20, [11_600]),
        (LisieckiTest, 35, [1_600_00])
    ]:
        for model in [Metrics(span=span), lstm.with_args(span=span)]:
            model.run_with_output(
                datasets, 
                transitions=transitions,
                path=PLOT_OUTPUT, title=False)
    
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

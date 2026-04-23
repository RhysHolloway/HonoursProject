if __name__ == "__main__":
    print("Importing environment...")
    import os
    from models.lstm import LSTMLoader
    from models.datasets import *

    import models.lstm.test as test
    from models.lstm.test import DatasetModel
    
    lstm = LSTMLoader(get_project_path("bury_models/")).with_args(verbose=True, spacing=10)
    
    plot_output = get_project_path("output/plots")
        
    # Run models on datasets
    
    lstm.run_with_output([Lisiecki], path=plot_output)
    
    print("Generating project figures from data...")
    test.load_and_save(
        path = get_project_path("output/metrics/"),
        output = os.path.join(plot_output),
        lstm=lstm,
        models=\
            test.test_models(set([1500, 500])) +
            [
                # Datasets and transition points
                # DatasetModel(Lisiecki,  20, []),
                # DatasetModel(Scotese,   0.2, [250.0]),
                # DatasetModel(JuddAvgs,  5,  [250.0]),
            ] +
            [
                DatasetModel(dataset, bandwidth, tcrit, spacing=10)
                for dataset, bandwidth, tcrit in BuryPaleoclimate
            ],
    )

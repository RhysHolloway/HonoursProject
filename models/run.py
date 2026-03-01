if __name__ == "__main__":
    print("Importing environment...")
    import os
    from models.lstm import LSTMLoader
    from models.datasets import *

    import models.lstm.test as test
    
    lstm = LSTMLoader(get_project_path("bury_models/")).with_args(verbose=True, spacing=5)
    
    plot_output = get_project_path("output/plots")
        
    # Run models on datasets
    
    print("Generating project figures from data...")
    test.load_and_save(
        metrics_path = get_project_path("env/testing/metrics/"),
        output = os.path.join(plot_output, "tests/"),
        lstm=lstm,
        models=test.get_or_generate_models(path=get_project_path("env/testing/models/")) +
            [(tup[0].name, [tup[0]], float(tup[1])) for tup in [
                # Datasets and transition points
                (Scotese, 250)
            ]],
    )
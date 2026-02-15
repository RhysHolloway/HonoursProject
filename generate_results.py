from typing import Any
from models.lstm import LSTMLoader, LSTM
from models.datasets import *

from models.lstm.test import test as test_lstm
      
# Run models on datasets

# Metrics(window = 0.2).run_with_output([Lisiecki, Scotese, Judd], plot_path=output)
# DLM().run_with_output([Lisiecki, Scotese, Judd], plot_path=output)
# HMM().run_with_output([Lisiecki, Scotese, Judd], plot_path=output)

loader = LSTMLoader(get_project_path("output/models/lstm/"))
LSTM = loader.with_args(verbose=False)

LSTM.run_with_output(map(Dataset.normalize, [Scotese, JuddGMST, JuddCO2]), print=False, plot_path=output)

def test(tup: tuple[Dataset, Any]):
    dataset, transition = tup
    return (dataset.name, [dataset], transition)

test_lstm(
    LSTM,
    output=get_project_path("output/plots/empirical/")
)

test_lstm(
    LSTM,
    map(test, [
        (Scotese.normalize(), 250)
    ]),
    get_project_path("output/plots/empirical/")
)
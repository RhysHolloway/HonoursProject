from functools import reduce
from typing import Final, Iterable, Literal, Tuple
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

type BifId = Literal["BP", "LP", "HB"]
type BifType = Tuple[BifId, bool]
BIFS: Final[list[BifId]] = ["BP", "LP", "HB"]

def bif_types():
    for null in [True, False]:
        for type in BIFS:
            yield (type, null)

def bif_maximum(type: BifType, bif_max: int) -> int:
    type, null = type
    if null:
        return ((bif_max - 2 * (bif_max // 3)) if (type == "BP") else (bif_max // 3))
    else:
        return bif_max

INDEX_COL: Final[str] = 'sequence_ID'
LABEL_COLS: Final[list[str]] = [INDEX_COL, 'class_label', 'bif', 'null']

type Sims = pd.Series
type Resids = pd.Series
type Labels = pd.DataFrame
type Groups = pd.DataFrame
type TrainData = tuple[dict[int, tuple[Sims, Resids]], Labels, Groups]

# Returns combined (indexed sims + resids, labels, groups)
def combine(batches: Iterable[TrainData]) -> TrainData:
    
    def _reduce_combine(
        a: TrainData, 
        b: TrainData
    ) -> TrainData:
        seq_max = max(a[0].keys())
        new_labels = b[1].copy()
        new_labels.index += seq_max # index starts at 1 so no conflict between last key in previous and first key in next
        new_groups = b[2].copy()
        new_groups.index += seq_max
        return (a[0] | {key + seq_max:val for key, val in b[0].items()}, pd.concat([a[1], new_labels]), pd.concat([a[2], new_groups]))
    
    return reduce(_reduce_combine, batches)


def compute_residuals(data: pd.Series) -> pd.Series:
        smooth_data = lowess(data.values, data.index.values, frac=0.2)[:, 1]
        return pd.Series(data.values - smooth_data, index=data.index, name="Residuals")
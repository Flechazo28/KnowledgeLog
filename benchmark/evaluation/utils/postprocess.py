from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def post_average(metric_file, tech, complex, frequent):
    df = pd.read_csv(metric_file, index_col=False)
    df = df.drop_duplicates(['Dataset'])
    mean_row = df.select_dtypes(include=[np.number]).mean().round(3)
    new_row = pd.DataFrame([['Average']], columns=['Dataset']).join(pd.DataFrame([mean_row.values], columns=mean_row.index))
    df = pd.concat([df, new_row], ignore_index=True)
    output_dir = PROJECT_ROOT / "result" / "summary_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{tech}.csv"
    df.to_csv(output_path, index=False)
    df = pd.read_csv(output_path)
    transposed_df = df.transpose()
    transposed_df.to_csv(output_path)

"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os
from datetime import datetime

sys.path.append('../')

from logparser.KnowledgeLog import LogParser
from evaluation.settings import benchmark_settings
from evaluation.utils.common import common_args
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average


datasets_2k = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]

datasets_full = [
    "Proxifier",
    "Apache",
    "OpenSSH",
    "HDFS",
    "OpenStack",
    "HPC",
    "Zookeeper",
    "HealthApp",
    "Hadoop",
    "Spark",
    "BGL",
    "Linux",
    "Mac",
    "Thunderbird",
]

datasets_lunar = [
    "CTS",
    "HiBench",
    "LoFI",
]

# datasets_full = ['Proxifier']


if __name__ == "__main__":
    args = common_args()
    data_type = "lunar" if args.lunar_data else ("full" if args.full_data else "2k")
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.lunar_data:
        input_dir = "../../lunar_dataset/"
    elif args.full_data:
        input_dir = "../../full_dataset/"
    else:
        input_dir = "../../2k_datasets/"
    if args.lunar_data:
        datasets = datasets_lunar
    elif args.full_data:
        datasets = datasets_full
    else:
        datasets = datasets_2k
    if args.datasets:
        requested_datasets = []
        for item in args.datasets:
            requested_datasets.extend([name.strip() for name in item.split(",") if name.strip()])
        unknown_datasets = sorted(set(requested_datasets) - set(datasets))
        if unknown_datasets:
            raise ValueError(f"Unknown datasets for {data_type} data: {unknown_datasets}")
        datasets = requested_datasets

    output_dir = args.output_dir or f"../../result/{run_time}_{len(datasets)}datasets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent,
        overwrite=args.output_dir is None
    )

    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'] if args.lunar_data else setting['log_file'].replace("_2k", f"_{data_type}")
        indir = os.path.join(input_dir, os.path.dirname(log_file))
        log_file_basename = os.path.basename(log_file)
        if os.path.exists(os.path.join(output_dir, f"{log_file_basename}_structured.csv")):
            parser = None
            print("parseing result exist.")
        else:
            parser = LogParser
        # run evaluator for a dataset
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=parser,
            param_dict={
                # 'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': setting['regex'],
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': [],
                'data_type': data_type,
                'model': args.model,
                'run_id': run_time,
            },
            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file,
        )  # it internally saves the results into a summary file
    metric_file = os.path.join(output_dir, result_file)
    summary_name = (
        f"{run_time}_{len(datasets)}datasets_"
        f"KnowledgeLog_{data_type}_complex={args.complex}_frequent={args.frequent}_"
        f"{args.model}"
    )
    post_average(metric_file, summary_name, args.complex, args.frequent)

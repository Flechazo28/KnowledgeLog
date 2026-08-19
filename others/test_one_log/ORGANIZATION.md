# test_one_log Organization

This folder contains small, focused LILAC prompt-stability tests. Scripts stay
at the folder root so existing commands such as `test_one_log/run_test.py` keep
working. Generated outputs are grouped under `results/`.

## Scripts

| File | Purpose |
|---|---|
| `run_test.py` | Generic end-to-end runner for arbitrary inline or file-based log/template cases, with independent repetitions. |
| `run_six_logs.py` | Tests the selected six high-frequency Mac and Thunderbird logs. |
| `run_java_io_exception_logs.py` | Extracts and tests representative `java.io.IOException` logs from HDFS, Spark, Hadoop, Zookeeper, and BGL. |

## Results

| Directory | Content |
|---|---|
| `results/single_log/` | Historical one-log Linux repeated-inference results and run logs. |
| `results/generic/` | New generic single-log and multi-case end-to-end test outputs. |
| `results/six_logs/` | Historical six-log Mac/Thunderbird test outputs and run logs. |
| `results/thunderbird_3logs/` | Thunderbird three-error focused test outputs. |
| `results/java_io_exception/` | `java.io.IOException` extraction/test outputs. |
| `_cache/` | Python bytecode cache moved out of the root listing. |

## Current java.io.IOException Representative Set

The latest representative extraction is:

`results/java_io_exception/java_io_exception_unique_templates_20260611_183753/`

It deduplicates by ground-truth `EventTemplate`, keeping one representative log
per dataset/template pair:

| Dataset | Unique templates |
|---|---:|
| HDFS | 11 |
| Spark | 5 |
| Hadoop | 8 |
| Zookeeper | 2 |
| BGL | 1 |

## Recommended Output Paths

New runs should write under the matching `results/` subdirectory. The scripts'
defaults already do this, but timestamped output directories are still clearer:

```bash
cd /mnt/Disk3/wjy23/wjy23/loghub-2.0/benchmark/logparser/KnowledgeLog-main

TS=$(date +%Y%m%d_%H%M%S)

/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python -u \
  test_one_log/run_java_io_exception_logs.py \
  --output-dir "test_one_log/results/java_io_exception/java_io_exception_result_${TS}"
```

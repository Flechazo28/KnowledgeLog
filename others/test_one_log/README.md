# Generic end-to-end single-log test

See `ORGANIZATION.md` for the current folder layout. Scripts stay at this
folder root; generated outputs are grouped under `results/`.

`run_test.py` tests arbitrary logs through the same single-log path used by the
project parser: preprocessing, domain-knowledge prompt, model call, response
extraction, common regex processing, postprocessing, and exact comparison with
the supplied ground-truth template. It deliberately does not create or use a
`ParsingCache`; every repetition directly calls the model for the supplied log.

## One log from the command line

```bash
/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python -u \
  test_one_log/run_test.py \
  --log 'session opened for user root by (uid=0)' \
  --expected-template 'session opened for user <*> by <*>(uid=<*>)' \
  --repetitions 10
```

`{variables}` is also accepted in the ground-truth template and is normalized
to the final `<*>` representation before comparison.

## Several inline logs

```bash
/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python \
  test_one_log/run_test.py \
  --case auth_case \
    'session opened for user root by (uid=0)' \
    'session opened for user <*> by <*>(uid=<*>)' \
  --case cmd_case \
    '(root) CMD (/projects/tbird/temps/get_temps a)' \
    '(<*>) CMD (<*>)' \
  --repetitions 5
```

## JSON or JSONL case files

JSON arrays and `{"cases": [...]}` objects are supported. Existing JSONL-style
sample files with a `.json` suffix are supported too. Each case may use these
field names:

- Log: `log`, `log_message`, `query`, or `Content`.
- Truth: `expected_template`, `ground_truth_template`, `true_template`,
  `answer`, or `EventTemplate`.

Example:

```json
[
  {
    "id": "auth_case",
    "log": "session opened for user root by (uid=0)",
    "expected_template": "session opened for user <*> by <*>(uid=<*>)"
  },
  {
    "id": "cmd_case",
    "log": "(root) CMD (/projects/tbird/temps/get_temps a)",
    "expected_template": "(<*>) CMD (<*>)"
  }
]
```

Run all cases, or select any subset by repeating `--case-id`:

```bash
/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python -u \
  test_one_log/run_test.py \
  --cases-file /path/to/cases.json \
  --case-id auth_case \
  --case-id cmd_case \
  --repetitions 5
```

Use `--list-cases` with the same input arguments to verify case IDs and values
without making model calls.

Each run creates a timestamped directory under `results/generic/` unless
`--output-dir` is provided. Outputs are:

- `configuration.json`: selected cases, enabled rule groups, generated domain
  knowledge, preprocessed logs, model, and common-regex path.
- `results.jsonl`: final template, validity, elapsed time, and exact-match result
  for every independent run.
- `summary.md`: per-case accuracy, template frequencies, and individual results.

## Six fixed Mac and Thunderbird logs

`run_six_logs.py` tests only the six high-frequency error examples selected
from the latest Mac and Thunderbird analysis. By default, it makes one
independent API call for each log:

```bash
TS=$(date +%Y%m%d_%H%M%S)
/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python \
  test_one_log/run_six_logs.py \
  --output-dir "test_one_log/results/six_logs/six_logs_result_${TS}"
```

Use `--repetitions 10` to call the model ten times for each of the six logs.
The output records the raw response, extracted `{variables}` template, final
`<*>` template, expected template, and exact-match result.

## BGL two regression logs

`run_bgl_two_errors.py` tests only the two BGL logs that were correct in the
2026-06-10 run but became `data <*> <*>` in the 2026-06-12 run:

```bash
TS=$(date +%Y%m%d_%H%M%S)
/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python -u \
  test_one_log/run_bgl_two_errors.py \
  --output-dir "test_one_log/results/bgl_two_errors/bgl_two_errors_result_${TS}"
```

Use `--repetitions 10` to call the model ten times for each BGL log.

## Spark three high-frequency error logs

`run_spark_three_errors.py` tests only the top three Spark PA-error clusters
from the 2026-06-12 all-dataset run:

```bash
TS=$(date +%Y%m%d_%H%M%S)
/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python -u \
  test_one_log/run_spark_three_errors.py \
  --output-dir "test_one_log/results/spark_three_errors/spark_three_errors_result_${TS}"
```

Use `--repetitions 10` to call the model ten times for each Spark log.

## java.io.IOException representative logs

`run_java_io_exception_logs.py` extracts logs whose evaluation `Content` field
contains `java.io.IOException` from HDFS, Spark, Hadoop, Zookeeper, and BGL. By
default it deduplicates by ground-truth `EventTemplate`, keeping one
representative log per dataset/template pair.

```bash
TS=$(date +%Y%m%d_%H%M%S)
/mnt/Disk3/wjy23/wjy23/anaconda3/envs/loghub/bin/python -u \
  test_one_log/run_java_io_exception_logs.py \
  --output-dir "test_one_log/results/java_io_exception/java_io_exception_result_${TS}"
```

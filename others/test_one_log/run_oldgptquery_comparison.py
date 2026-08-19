#!/usr/bin/env python3
"""Compare useful old domain rules with the current KnowledgeLog rules."""

import ast
import json
import re
from collections import Counter
from datetime import datetime

from run_webmaster_rule_ablation import ROOT, common_regexes, gq, parse


REPETITIONS = 2
OLD_FILE = (
    ROOT
    / "benchmark/logparser/KnowledgeLog/oldgptquery"
    / "gpt_query老版本-之前已经调优过的版本.md"
)

CASES = {
    "absolute_uri": (
        "Home page is http://smartmontools.sourceforge.net/",
        "Home page is <*>",
    ),
    "pam_fields": (
        "authentication failure; logname= uid=0 euid=0 tty=NODEVssh "
        "ruser= rhost=co-f-211-78-4-214.kgex.com.tw user=root",
        "authentication failure; logname=<*> uid=<*> euid=<*> tty=<*> "
        "ruser=<*> rhost=<*> user=<*>",
    ),
    "empty_role_uid": (
        "session opened for user root by (uid=0)",
        "session opened for user <*> by <*>(uid=<*>)",
    ),
    "dotted_counter": (
        "program interrupt: trap instruction.........0",
        "program interrupt: trap instruction.........<*>",
    ),
    "core_dump": (
        "generating core.6466",
        "generating core.<*>",
    ),
    "attempt_id": (
        "attempt_201706081738_1426_m_000013_57053: Committed",
        "<*>: Committed",
    ),
    "version_token": (
        "Real Time Clock Driver v1.12",
        "Real Time Clock Driver <*>",
    ),
    "compact_quantity": (
        "agpgart: detected 4MB dedicated video ram.",
        "agpgart: detected <*> dedicated video ram.",
    ),
    "spaced_unit": (
        "[instance: 3f555a09-a38e-443e-8223-0cbb912e72f9] "
        "memory limit: 96258.00 MB, free: 95746.00 MB",
        "[instance: <*>] memory limit: <*> MB, free: <*> MB",
    ),
}

GUARDS = {
    "attempt_start_constant": (
        "ATTEMPT_START task_1445182159119_0017_m_000006",
        "ATTEMPT_START <*>",
    ),
    "fixed_component": (
        "pam_unix(sshd:auth): check pass; user unknown",
        "pam_unix(sshd:auth): check pass; user unknown",
    ),
}


def old_prompt_rules():
    tree = ast.parse(OLD_FILE.read_text(encoding="utf-8"))
    wanted = {
        "CORE_PROMPT_RULES",
        "STRUCTURE_PROMPT_RULES",
        "SEMANTIC_ROLE_PROMPT_RULES",
        "LITERAL_PROMPT_RULES",
    }
    groups = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            groups[target.id] = ast.literal_eval(node.value)
    return [
        rule
        for name in (
            "CORE_PROMPT_RULES",
            "STRUCTURE_PROMPT_RULES",
            "SEMANTIC_ROLE_PROMPT_RULES",
            "LITERAL_PROMPT_RULES",
        )
        for rule in groups[name]
    ]


def candidate_rules():
    v = gq.VARIABLE_PLACEHOLDER
    return {
        "absolute_uri": gq.RegexRule(
            "old_absolute_uri",
            r"(?<![A-Za-z0-9+.-])(?:https?|hdfs|file)://"
            r"[^\s\]\)\}\>,\"']+(?<![.,:;])",
            v,
            re.IGNORECASE,
        ),
        "pam_fields": gq.RegexRule(
            "old_pam_fields",
            r"\blogname=\S*\s+uid=\S+\s+euid=\S+\s+tty=\S+\s+"
            r"ruser=\S*\s+rhost=\S+\s+user=\S+",
            f"logname={v} uid={v} euid={v} tty={v} "
            f"ruser={v} rhost={v} user={v}",
        ),
        "empty_role_uid": gq.RegexRule(
            "old_empty_role_uid",
            r"(\b(?:by|from|for|as)\s+)"
            r"\(\s*([A-Za-z][A-Za-z0-9_.-]*)\s*=\s*[^()]*\)",
            rf"\1{v}(\2={v})",
        ),
        "dotted_counter": gq.RegexRule(
            "old_dotted_counter",
            r"(\.{2,})[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?)\s*$",
            rf"\1{v}",
        ),
        "core_dump": gq.RegexRule(
            "old_core_dump",
            r"\bcore\.\d+\b",
            f"core.{v}",
            re.IGNORECASE,
        ),
        "attempt_id": gq.RegexRule(
            "old_attempt_id",
            r"\b(?:attempt|container)_[A-Za-z0-9_]+"
            r"(?:_[A-Za-z0-9_]+)*\b",
            v,
            re.IGNORECASE,
        ),
        "attempt_id_refined": gq.RegexRule(
            "refined_attempt_id",
            r"\b(?:attempt|container)_\d[A-Za-z0-9_]*\b",
            v,
            re.IGNORECASE,
        ),
        "version_token": gq.RegexRule(
            "old_version_token",
            r"\bv\d+(?:\.\d+)+\b",
            v,
            re.IGNORECASE,
        ),
        "compact_quantity": gq.RegexRule(
            "old_compact_quantity",
            r"(?<![A-Za-z0-9_.-])[-+]?\d+(?:\.\d+)?"
            r"(?:KiB|Kbytes|GHz|KB|kB|MB|GB|TB|ms|B|K|M|G|C)"
            r"(?![A-Za-z0-9_])",
            v,
        ),
        "spaced_unit": gq.RegexRule(
            "old_spaced_unit",
            r"(?<![A-Za-z0-9_.-])([-+]?\d+(?:\.\d+)?)(\s+)"
            r"(KB|MB|GB|TB|VCPUS|VCPU|CPU)\b",
            rf"{v}\2\3",
            re.IGNORECASE,
        ),
    }


def main():
    current_prompt = list(gq.CURATED_PROMPT_RULES)
    old_prompt = old_prompt_rules()
    current_pre = list(gq.PREPROCESSING_RULE_GROUPS["curated"])
    candidates = candidate_rules()
    regs = common_regexes()
    records = []

    for case, (log, expected) in CASES.items():
        variants = {
            "current": (current_prompt, current_pre),
            "old_prompt_current_pre": (old_prompt, current_pre),
            f"add_old_{case}": (
                current_prompt,
                [candidates[case]] + current_pre,
            ),
        }
        if case == "attempt_id":
            variants["add_refined_attempt_id"] = (
                current_prompt,
                [candidates["attempt_id_refined"]] + current_pre,
            )
        for variant, (prompt, preprocessing) in variants.items():
            for repetition in range(1, REPETITIONS + 1):
                result = parse(log, expected, prompt, preprocessing, regs)
                result.update(
                    case=case,
                    variant=variant,
                    repetition=repetition,
                )
                records.append(result)
                print(
                    f"{case:18s} {variant:28s} {repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if result['correct'] else 'wrong'} "
                    f"{result['final']!r}",
                    flush=True,
                )

    guard_variants = {
        "current": (current_prompt, current_pre),
        "add_old_attempt_id": (
            current_prompt,
            [candidates["attempt_id"]] + current_pre,
        ),
        "add_refined_attempt_id": (
            current_prompt,
            [candidates["attempt_id_refined"]] + current_pre,
        ),
    }
    for case, (log, expected) in GUARDS.items():
        for variant, (prompt, preprocessing) in guard_variants.items():
            for repetition in range(1, REPETITIONS + 1):
                result = parse(log, expected, prompt, preprocessing, regs)
                result.update(
                    case=case,
                    variant=variant,
                    repetition=repetition,
                )
                records.append(result)
                print(
                    f"{case:18s} {variant:28s} {repetition}/{REPETITIONS}: "
                    f"{'CORRECT' if result['correct'] else 'wrong'} "
                    f"{result['final']!r}",
                    flush=True,
                )

    output_dir = (
        ROOT
        / "others/test_one_log/results/domain_rule_ablation"
        / f"oldgptquery_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {}
    for record in records:
        key = f"{record['case']}::{record['variant']}"
        summary.setdefault(key, []).append(record)
    summary = {
        key: {
            "correct": sum(item["correct"] for item in items),
            "runs": len(items),
            "templates": dict(Counter(item["final"] for item in items)),
        }
        for key, items in summary.items()
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

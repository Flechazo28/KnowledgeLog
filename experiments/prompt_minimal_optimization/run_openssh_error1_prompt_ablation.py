#!/usr/bin/env python3
"""Test prompt-only fixes for the OpenSSH missing-account boundary."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import regex


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

from logparser.KnowledgeLog import gpt_query as gq  # noqa: E402


MODEL = "Llama-3.3-70B-Instruct"
OUTPUT = Path(__file__).with_name("openssh_error1_prompt_ablation.json")
ACCOUNT_RULE_INDEX = 17

INSTRUCTION = (
    "I want you to act like an expert of log parsing. I will give you rules "
    "that you must follow and a log message delimited by backticks. You must "
    "identify and abstract all dynamic variables in logs with {variables}, then "
    "output a static log template. Please note that there might be no variables "
    "in logs. You must print the input log's template following this exact "
    "format: 'Log template:`...`'"
)

CASES = [
    {
        "id": "missing_account",
        "repetitions": 5,
        "log": "Failed password for invalid user from 103.204.177.144 port 2223 ssh2",
        "expected": "Failed password for <*> from <*> port <*> ssh2",
    },
    {
        "id": "named_invalid_account",
        "repetitions": 2,
        "log": "Failed password for invalid user webmaster from 173.234.31.186 port 38926 ssh2",
        "expected": "Failed password for invalid user <*> from <*> port <*> ssh2",
    },
    {
        "id": "ordinary_account",
        "repetitions": 2,
        "log": "Failed password for root from 5.36.59.76 port 42393 ssh2",
        "expected": "Failed password for <*> from <*> port <*> ssh2",
    },
    {
        "id": "accepted_account",
        "repetitions": 2,
        "log": "Accepted password for fztu from 119.137.62.142 port 49116 ssh2",
        "expected": "Accepted password for <*> from <*> port <*> ssh2",
    },
    {
        "id": "fixed_auth_result",
        "repetitions": 2,
        "log": "pam_unix(sshd:auth): check pass; user unknown",
        "expected": "pam_unix(sshd:auth): check pass; user unknown",
    },
]

VARIANTS = {
    "current": gq.CURATED_PROMPT_RULES[ACCOUNT_RULE_INDEX],
    "generic_principal_role": (
        "Abstract a concrete account identity in a user, login, account, "
        "authentication, or principal field. In an authentication failure, a "
        "phrase occupying the principal role is variable when no concrete account "
        "identity is present; preserve a fixed qualifier when a concrete identity "
        "follows it. Preserve fixed result text such as user unknown."
    ),
    "password_principal_span": (
        "In a password-authentication message, the text between the account cue "
        "'for' and the source cue 'from' is the principal field. Abstract a concrete "
        "account identity; preserve the qualifier 'invalid user' when an account "
        "follows it, but abstract the complete principal field when no account "
        "follows. Preserve fixed result text such as user unknown."
    ),
    "concise_authentication_grammar": (
        "In password-authentication messages, abstract the principal span between "
        "'for' and 'from'. If that span is 'invalid user' plus an account identity, "
        "preserve 'invalid user' and abstract only the identity. Preserve fixed "
        "authentication result text such as user unknown."
    ),
    "strict_authentication_precedence": (
        "Highest-priority authentication boundary rule: in a 'Failed password for "
        "... from ...' event, inspect the principal field between 'for' and 'from'. "
        "If that field is exactly 'invalid user' with no following account identity, "
        "replace the complete field with {variables}. If an account identity follows "
        "'invalid user', preserve 'invalid user' and replace only the identity with "
        "{variables}. This rule overrides general fixed-phrase guidance."
    ),
    "unresolved_principal_value": (
        "Treat an unresolved principal description as a dynamic account-field value, "
        "not as fixed event wording. In a password failure, replace the complete "
        "principal description with {variables} when no concrete account identity "
        "follows it; when a qualifier precedes a concrete identity, preserve the "
        "qualifier and abstract only the identity."
    ),
    "explicit_authentication_boundary": (
        "For 'Failed password for PRINCIPAL from SOURCE', output PRINCIPAL as one "
        "{variables} when PRINCIPAL contains no concrete account identity. For "
        "'Failed password for invalid user ACCOUNT from SOURCE', preserve 'invalid "
        "user' and output ACCOUNT as {variables}. Abstract SOURCE separately and "
        "preserve the remaining event text."
    ),
    "no_phantom_account_placeholder": (
        "Never create {variables} for a token that is absent from the original log. "
        "In a password-authentication event, abstract the existing text that occupies "
        "the principal field between 'for' and 'from'. If the field contains only "
        "'invalid user' and no account identity, replace that complete existing phrase "
        "with {variables}; do not preserve it and append an empty placeholder. If an "
        "account follows 'invalid user', preserve the qualifier and abstract the account."
    ),
    "existing_principal_span": (
        "A placeholder must replace existing log text and must never represent a "
        "missing token. For password authentication, the existing span between 'for' "
        "and 'from' is the principal field. Abstract that complete span when it has no "
        "concrete account identity; otherwise abstract the identity and preserve any "
        "fixed qualifier that precedes it."
    ),
    "explicit_boundary_counterexample": (
        "Authentication boundary rule: when the original text is 'for invalid user "
        "from', output 'for {variables} from'; never output 'for invalid user "
        "{variables} from' because no account token exists there. When the original "
        "text is 'for invalid user ACCOUNT from', preserve 'invalid user' and replace "
        "ACCOUNT with {variables}."
    ),
    "strict_authentication_precedence_at_end": (
        "Highest-priority authentication boundary rule: in a 'Failed password for "
        "... from ...' event, inspect the principal field between 'for' and 'from'. "
        "If that field is exactly 'invalid user' with no following account identity, "
        "replace the complete field with {variables}. If an account identity follows "
        "'invalid user', preserve 'invalid user' and replace only the identity with "
        "{variables}. This final rule overrides all earlier rules."
    ),
    "explicit_boundary_counterexample_at_end": (
        "Final authentication boundary rule: when the original text is 'for invalid "
        "user from', output 'for {variables} from'; never preserve 'invalid user' and "
        "never insert a placeholder for a missing account. When the original text is "
        "'for invalid user ACCOUNT from', preserve 'invalid user' and replace ACCOUNT "
        "with {variables}. This final rule overrides all earlier rules."
    ),
    "minimal_boundary_at_end": (
        "Final rule: replace the existing phrase between 'for' and 'from' with "
        "{variables} when a failed-password log contains no concrete account identity. "
        "Do not invent a placeholder for an absent token. If a concrete identity follows "
        "'invalid user', preserve that qualifier and abstract only the identity."
    ),
    "general_principal_field_at_end": (
        "Abstract the complete principal field between 'for' and 'from' in a "
        "password-authentication failure when it contains no concrete account "
        "identity. When a qualifier precedes a concrete identity, preserve the "
        "qualifier and abstract only the identity."
    ),
    "account_qualifier_at_end": (
        "Abstract an account field as one {variables} when it contains only an "
        "account-status qualifier, such as 'invalid user', and no concrete account "
        "identity. When the qualifier is followed by a concrete identity, preserve "
        "the qualifier and abstract only the identity."
    ),
    "failed_password_principal_at_end": (
        "In a failed-password message, abstract the complete principal field between "
        "'for' and 'from' when no concrete account identity follows 'invalid user'. "
        "When an identity follows 'invalid user', preserve 'invalid user' and abstract "
        "only the identity."
    ),
    "always_abstract_principal_at_end": (
        "Always abstract the complete principal field between 'for' and 'from' in a "
        "failed-password message when no concrete account identity is present. When "
        "'invalid user' is followed by an account identity, preserve 'invalid user' "
        "and abstract only the identity."
    ),
    "always_abstract_unresolved_account_at_end": (
        "Always abstract an unresolved account field as one {variables}, including "
        "the existing account-status text when no concrete identity is present. When "
        "a concrete identity follows the account-status text, preserve the status "
        "text and abstract only the identity."
    ),
    "preserve_qualifier_conditionally_at_end": (
        "Preserve an account-status qualifier such as 'invalid user' only when it "
        "modifies a concrete account identity. If no identity follows the qualifier "
        "in the principal field of a failed-password message, abstract the complete "
        "principal field as one {variables}."
    ),
    "pruned_prompt_authentication_grammar_at_end": (
        "In a 'Failed password for ... from ...' event, inspect the principal field "
        "between 'for' and 'from'. If that field is exactly 'invalid user' with no "
        "following account identity, replace the complete field with {variables}. If "
        "an identity follows 'invalid user', preserve 'invalid user' and abstract only "
        "the identity."
    ),
    "compact_prompt_authentication_grammar_at_end": (
        "In a 'Failed password for ... from ...' event, inspect the principal field "
        "between 'for' and 'from'. If that field is exactly 'invalid user' with no "
        "following account identity, replace the complete field with {variables}. If "
        "an identity follows 'invalid user', preserve 'invalid user' and abstract only "
        "the identity."
    ),
    "focused_prompt_authentication_grammar_at_end": (
        "In a 'Failed password for ... from ...' event, inspect the principal field "
        "between 'for' and 'from'. If that field is exactly 'invalid user' with no "
        "following account identity, replace the complete field with {variables}. If "
        "an identity follows 'invalid user', preserve 'invalid user' and abstract only "
        "the identity."
    ),
}

COMPACT_BASE_RULES = [
    "Replace each complete dynamic field with {variables}; preserve static words, field order, and punctuation.",
    "Treat a complete hostname:port or IP:port as one variable field.",
    "Abstract a numeric port when its network role is explicit.",
    "For a key-value field, preserve its key and delimiter and abstract only its variable value.",
    "Abstract a concrete account identity when context identifies a user, login, account, authentication identity, or principal.",
    "Never create a placeholder for text that is absent from the original log.",
]

FOCUSED_BASE_RULES = [
    "Replace each complete dynamic field with {variables} and preserve all static event wording.",
    "Abstract source addresses and explicitly labeled port numbers.",
    "Abstract concrete account identities in authentication messages.",
    "Never create a placeholder for text that is absent from the original log.",
]


def build_knowledge(account_rule, variant, place_at_end=False):
    if variant.startswith("compact_prompt_"):
        rules = COMPACT_BASE_RULES + [account_rule]
    elif variant.startswith("focused_prompt_"):
        rules = FOCUSED_BASE_RULES + [account_rule]
    elif variant.startswith("pruned_prompt_"):
        rules = [
            rule
            for index, rule in enumerate(gq.CURATED_PROMPT_RULES)
            if index != ACCOUNT_RULE_INDEX and not 22 <= index <= 35
        ]
        rules.append(account_rule)
    else:
        rules = list(gq.CURATED_PROMPT_RULES)
        if place_at_end:
            del rules[ACCOUNT_RULE_INDEX]
            rules.append(account_rule)
        else:
            rules[ACCOUNT_RULE_INDEX] = account_rule
    return "\nYou must follow these final domain knowledge rules:\n" + "\n".join(
        f"{index}. {rule}" for index, rule in enumerate(rules, 1)
    )


def load_common_regexes():
    path = ROOT / "benchmark" / "logparser" / "KnowledgeLog" / "common.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [regex.compile(pattern) for pattern in payload["COMMON"]["regex"]]


def run_once(case, variant, account_rule, common_regexes):
    preprocessed = gq.apply_final_domain_preprocessing(case["log"])
    response = gq.infer_llm(
        INSTRUCTION,
        build_knowledge(
            account_rule,
            variant,
            place_at_end=variant.endswith("_at_end"),
        ),
        f"Log message: `{preprocessed}`",
        case["log"],
        model=MODEL,
        temperature=0.0,
    )
    extracted = gq._extract_template_from_response(response)
    final_template = None
    valid = False
    if extracted is not None:
        final_template, valid = gq.post_process_template(
            extracted,
            regs_common=common_regexes,
            log_message=case["log"],
        )
    return {
        "variant": variant,
        "case_id": case["id"],
        "log": case["log"],
        "preprocessed_log": preprocessed,
        "expected": case["expected"],
        "account_rule": account_rule,
        "raw_response": response,
        "final_template": final_template,
        "valid": bool(valid),
        "correct": bool(valid and final_template == case["expected"]),
    }


def summarize(records):
    summary = {}
    for variant in VARIANTS:
        selected = [record for record in records if record["variant"] == variant]
        per_case = {}
        for case in CASES:
            subset = [record for record in selected if record["case_id"] == case["id"]]
            per_case[case["id"]] = {
                "correct": sum(record["correct"] for record in subset),
                "total": len(subset),
                "templates": dict(
                    Counter(record["final_template"] for record in subset)
                ),
            }
        summary[variant] = {
            "correct": sum(record["correct"] for record in selected),
            "total": len(selected),
            "per_case": per_case,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    variants = VARIANTS
    if args.variant:
        requested = set(args.variant)
        variants = {
            name: rule for name, rule in VARIANTS.items() if name in requested
        }
        missing = requested - set(variants)
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")
    common_regexes = load_common_regexes()
    records = []
    total = len(variants) * sum(case["repetitions"] for case in CASES)
    index = 0
    for variant, account_rule in variants.items():
        for case in CASES:
            for repetition in range(1, case["repetitions"] + 1):
                index += 1
                print(
                    f"[{index}/{total}] {variant} / {case['id']} / {repetition}",
                    flush=True,
                )
                record = run_once(
                    case,
                    variant,
                    account_rule,
                    common_regexes,
                )
                record["repetition"] = repetition
                records.append(record)
                print(
                    f"  {'correct' if record['correct'] else 'wrong'}: "
                    f"{record['final_template']!r}",
                    flush=True,
                )
                args.output.write_text(
                    json.dumps(
                        {"summary": summarize(records), "records": records},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

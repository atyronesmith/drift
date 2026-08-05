"""
Generate a summary report of functional equivalence test results.

Run after: pytest tests/test_functional_equivalence.py

Shared harness: functional_equivalence_lib
"""

import json

from functional_equivalence_lib import (
    RESULTS_DIR,
    TESTS,
    collect_variants,
    extract_code,
    load_fn,
    run_test,
    TRIALS,
)


def main():
    print("# Functional Equivalence Report")
    print()

    all_results = {}

    for trial, timestamp in TRIALS.items():
        run_dir = RESULTS_DIR / trial / timestamp
        if not run_dir.exists():
            continue

        trial_results = {}
        for i in range(10):
            run_file = run_dir / f"run_{i:03d}.json"
            if not run_file.exists():
                continue
            data = json.loads(run_file.read_text())
            content = data["steps"][1]["response"]["content"]
            code = extract_code(content)
            fn, err = load_fn(code)

            run_id = f"run_{i:03d}"
            if fn is None:
                trial_results[run_id] = {t: ("fail", f"load error: {err}") for t in TESTS}
                continue

            trial_results[run_id] = {}
            for test in TESTS:
                trial_results[run_id][test] = run_test(fn, test)

        all_results[trial] = trial_results

    # Summary per trial
    for trial, runs in all_results.items():
        print(f"## {trial}")
        print()

        # Count pass/fail/skip per test
        test_summary = {t: {"pass": 0, "fail": 0, "skip": 0} for t in TESTS}
        for run_id, tests in runs.items():
            for test, (status, _) in tests.items():
                test_summary[test][status] += 1

        n = len(runs)
        print(f"| Test | Pass | Fail | Skip | Rate |")
        print(f"|------|------|------|------|------|")
        for test in TESTS:
            s = test_summary[test]
            rate = s["pass"] / (s["pass"] + s["fail"]) * 100 if (s["pass"] + s["fail"]) > 0 else 100
            print(f"| {test} | {s['pass']}/{n} | {s['fail']}/{n} | {s['skip']}/{n} | {rate:.0f}% |")
        print()

        # Show failures
        failures = []
        for run_id, tests in runs.items():
            for test, (status, reason) in tests.items():
                if status == "fail":
                    failures.append((run_id, test, reason))

        if failures:
            print("### Failures")
            print()
            for run_id, test, reason in failures:
                print(f"- **{run_id}** / `{test}`: {reason}")
            print()

    # Overall summary
    print("## Overall Summary")
    print()
    total_variants = sum(len(runs) for runs in all_results.values())
    total_tests = total_variants * len(TESTS)
    total_pass = sum(
        1
        for runs in all_results.values()
        for tests in runs.values()
        for status, _ in tests.values()
        if status == "pass"
    )
    total_fail = sum(
        1
        for runs in all_results.values()
        for tests in runs.values()
        for status, _ in tests.values()
        if status == "fail"
    )
    total_skip = sum(
        1
        for runs in all_results.values()
        for tests in runs.values()
        for status, _ in tests.values()
        if status == "skip"
    )

    print(f"- **{total_variants} code variants** tested across 3 pipeline configurations")
    print(f"- **{total_pass}/{total_tests}** test assertions passed ({total_pass/total_tests*100:.1f}%)")
    print(f"- **{total_fail}/{total_tests}** test assertions failed ({total_fail/total_tests*100:.1f}%)")
    print(f"- **{total_skip}/{total_tests}** tests skipped (row_factory dependency)")
    print()

    # Per-variant pass rate
    print("### Per-Variant Functional Pass Rate")
    print()
    core_tests = [t for t in TESTS if t != "works_without_row_factory"]
    print(f"| Trial | Variant | Core Tests ({len(core_tests)}) | Robustness | Bug Found |")
    print(f"|-------|---------|------------|------------|-----------|")
    for trial, runs in all_results.items():
        short = trial.replace("pipeline-", "")
        for run_id, tests in runs.items():
            core_pass = sum(1 for t in core_tests if tests[t][0] == "pass")
            robust = tests["works_without_row_factory"][0]
            has_bug = any(tests[t][0] == "fail" for t in core_tests)
            bug_str = ""
            if has_bug:
                bug_str = ", ".join(t for t in core_tests if tests[t][0] == "fail")
            print(f"| {short} | {run_id} | {core_pass}/{len(core_tests)} | {robust} | {bug_str} |")
    print()

    print("## Expected Failures")
    print()
    print(
        "See ``tests/functional_equivalence_lib.py`` and ``docs/RESULTS.md`` "
        "(Experiment 3) for why remaining failures are research findings, not harness bugs."
    )
    print()
    print("- **pipeline-mixed/run_005**: duplicate dict key in generated code (3 core tests)")
    print("- **pipeline-mixed/run_005**: robustness may fail for the same bug")
    print("- **pipeline-structured/run_001, run_005**: robustness skips (row_factory dependency)")


if __name__ == "__main__":
    main()

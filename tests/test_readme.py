import glob
import json
import os
import re

import pytest

README = "README.md"
REQUIRED_SECTIONS = [
    "State and action representation",
    "Reward design",
    "RL algorithm",
    "Training setup",
    "Observed agent behaviour",
    "Failure cases",
    "What could be improved",
    "Results",
    "Baselines",
    "Reproducing",
]


@pytest.fixture(scope="module")
def readme():
    assert os.path.exists(README), "the README is the primary deliverable"
    return open(README, encoding="utf-8").read()


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_required_section_present(readme, section):
    assert re.search(re.escape(section), readme, re.IGNORECASE), \
        f"the write-up must cover: {section}"


def test_every_referenced_image_exists(readme):
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        if path.startswith("http"):
            continue
        assert os.path.exists(path), f"README references a missing file: {path}"


def test_readme_links_the_metrics_file(readme):
    assert "results/metrics.json" in readme


def test_readme_cites_the_key_references(readme):
    for citation in ("Ng", "Anderson", "Schulman", "van Hasselt"):
        assert citation in readme, f"missing citation: {citation}"


def test_readme_documents_the_reproduction_commands(readme):
    for cmd in ("experiments.validate_algos", "experiments.train",
                "experiments.ablations", "experiments.plots"):
        assert cmd in readme, f"missing reproduction command: {cmd}"


def test_readme_states_the_actual_seed_count(readme):
    """Derived from SEEDS, so the claim cannot drift from what was run."""
    from experiments.ablations import SEEDS

    n = len(SEEDS)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    pattern = rf"(?<![0-9])({n}|{words.get(n, n)}) seeds"
    assert re.search(pattern, readme, re.IGNORECASE), \
        f"README must state that {n} seeds were run"


def test_headline_numbers_match_metrics_json(readme):
    """Any 'SPL 0.xx' claim must match results/metrics.json to two decimals."""
    with open("results/metrics.json") as f:
        metrics = json.load(f)
    claimed = re.findall(r"SPL[^\d\n]{0,12}(\d\.\d\d)", readme)
    if not claimed:
        pytest.skip("no inline SPL claims to verify")
    actual = {
        f"{v[s]['spl']:.2f}"
        for v in metrics.values() for s in v if "spl" in v[s]
    }
    for c in claimed:
        assert c in actual, f"README claims SPL {c}, which is not in metrics.json"


def test_every_generated_plot_is_referenced(readme):
    """The reverse of the check above, and the one that was missing.

    `test_every_referenced_image_exists` catches a link to a file that is not
    there. It cannot catch a figure that was generated, sits in results/plots,
    and is never linked from the report -- which is invisible to a reader and
    was true of 7 of 12 plots at one point. The generator and the report have
    to agree in both directions.
    """
    generated = {os.path.basename(p) for p in glob.glob("results/plots/*.png")}
    if not generated:
        pytest.skip("no plots generated yet")
    referenced = set(re.findall(r"results/plots/([A-Za-z0-9_]+\.png)", readme))
    orphans = sorted(generated - referenced)
    assert not orphans, (
        f"{len(orphans)} figure(s) generated but never referenced in the README: "
        f"{orphans}. Either embed them or stop generating them."
    )

"""Print a concise paper-safe interpretation of Phase 7 metrics."""

from __future__ import annotations

import argparse
import json

from src.config import PROJECT_ROOT


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "eval"), default="eval")
    args = parser.parse_args()
    path = PROJECT_ROOT / "outputs" / "phase7" / args.split / "metrics.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    general = result["models"]["general"]
    personalized = result["models"]["personalized"]
    deployment = result["models"]["deployment_gated"]
    interval = result[
        "paired_bootstrap_personalized_vs_general"
    ]["macro_f1_change"]
    supported = interval["ci95_low"] > 0

    print("ASVspoof2019 LA — final held-out results")
    print(f"Samples: {result['samples']}")
    print(
        f"General: accuracy={percent(general['accuracy'])}, "
        f"macro F1={general['macro_f1']:.4f}, "
        f"AUC={general['roc_auc']:.4f}, EER={percent(general['eer'])}"
    )
    print(
        f"Personalized: accuracy={percent(personalized['accuracy'])}, "
        f"macro F1={personalized['macro_f1']:.4f}, "
        f"AUC={personalized['roc_auc']:.4f}, "
        f"EER={percent(personalized['eer'])}"
    )
    print(
        f"Deployment gated: accuracy={percent(deployment['accuracy'])}, "
        f"macro F1={deployment['macro_f1']:.4f} "
        "(AUC/EER not defined for the multi-gate rule)"
    )
    print(
        "Paired macro-F1 change 95% CI: "
        f"[{interval['ci95_low']:.6f}, {interval['ci95_high']:.6f}]"
    )
    print(
        "Statistically supported personalized improvement:",
        "YES" if supported else "NO",
    )


if __name__ == "__main__":
    main()

"""
MODULE 18 — Publication Report Generator (v1.0 — Research Grade)

Generates comprehensive accuracy reports suitable for peer-reviewed publication.

Outputs:
  - Full 5x5 confusion matrix with per-stage P/R/F1
  - Overall accuracy with 95% Wilson score confidence interval
  - Cohen's Kappa with standard error
  - Area statistics table (hectares per stage per epoch)
  - All metrics formatted as JSON + markdown tables

References:
  - Congalton & Green 2019 (Assessing the Accuracy of RS Data)
  - Foody 2002 (Status of land cover classification accuracy assessment)
  - Wilson 1927 (Probable inference, the law of succession)
"""

import json
import os
import math
import logging
import config

pub_log = logging.getLogger("PUBLICATION")

STAGE_NAMES = {
    1: "S1: Dense Mangrove",
    2: "S2: Degradation",
    3: "S3: Clearing",
    4: "S4: Water Filling",
    5: "S5: Operational Pond",
}


def _wilson_ci(p, n, z=1.96):
    """
    Wilson score confidence interval for proportion p with n observations.
    More robust than normal approximation for small samples or extreme p.
    Wilson 1927.
    """
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def _kappa_standard_error(kappa, n, k=5):
    """
    Approximate standard error for Cohen's Kappa.
    Fleiss, Cohen & Everitt 1969.
    """
    if n <= 0:
        return 0.0
    return math.sqrt((1.0 - kappa * kappa) / max(1, n - 1)) * math.sqrt(2.0 / max(1, k))


def _standard_normal_cdf(z):
    """Approximate standard normal CDF using Abramowitz & Stegun."""
    if z < -8:
        return 0.0
    if z > 8:
        return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p_const = 0.3275911
    sign = 1
    if z < 0:
        sign = -1
        z = -z
    t = 1.0 / (1.0 + p_const * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z / 2)
    return 0.5 * (1.0 + sign * y)


def compute_area_statistics(epoch_results):
    """
    Compute area statistics from epoch results (hectares per stage).

    Args:
        epoch_results: list of dicts with 'year', 'stage_distribution', 'total_pixels'

    Returns:
        dict with area tables and change statistics
    """
    pixel_area_ha = (config.TARGET_SCALE ** 2) / 10000.0

    area_table = []
    for epoch in epoch_results:
        year = epoch.get("year", "unknown")
        distribution = epoch.get("stage_distribution", {})
        total_pixels = epoch.get("total_pixels", 0)

        row = {"year": year, "total_pixels": total_pixels}
        for stage_id in range(1, 6):
            pct = float(distribution.get(str(stage_id), distribution.get(stage_id, 0)) or 0)
            px_count = int(total_pixels * pct / 100.0) if total_pixels > 0 else 0
            area_ha = px_count * pixel_area_ha
            row[f"S{stage_id}_pct"] = round(pct, 1)
            row[f"S{stage_id}_ha"] = round(area_ha, 1)
            row[f"S{stage_id}_px"] = px_count

        total_area_ha = total_pixels * pixel_area_ha
        row["total_area_ha"] = round(total_area_ha, 1)
        area_table.append(row)

    # Compute change between first and last epochs
    change_stats = {}
    if len(area_table) >= 2:
        first = area_table[0]
        last = area_table[-1]
        for stage_id in range(1, 6):
            key = f"S{stage_id}_ha"
            first_ha = first.get(key, 0)
            last_ha = last.get(key, 0)
            change_ha = last_ha - first_ha
            change_pct = (change_ha / max(first_ha, 0.1)) * 100
            change_stats[f"S{stage_id}"] = {
                "first_ha": round(first_ha, 1),
                "last_ha": round(last_ha, 1),
                "change_ha": round(change_ha, 1),
                "change_pct": round(change_pct, 1),
            }

    return {
        "area_table": area_table,
        "change_stats": change_stats,
        "pixel_area_ha": pixel_area_ha,
        "scale_m": config.TARGET_SCALE,
    }


def generate_confusion_matrix_report(kappa_results):
    """
    Generate a publication-ready confusion matrix report from Kappa validation.

    Args:
        kappa_results: dict from m17_stratified_gt.compute_kappa_from_samples()

    Returns:
        dict with formatted confusion matrix, CI, and per-class metrics
    """
    if not kappa_results or not kappa_results.get("available"):
        return {"available": False, "reason": kappa_results.get("reason", "no data") if kappa_results else "no data"}

    cm = kappa_results.get("confusion_matrix", [])
    labels = kappa_results.get("labels", [])
    n_samples = kappa_results.get("n_samples", 0)
    kappa = kappa_results.get("kappa", 0.0)
    oa = kappa_results.get("overall_accuracy", 0.0)
    per_class = kappa_results.get("per_class_f1", {})

    # Overall accuracy with 95% CI (Wilson score)
    oa_ci_low, oa_ci_high = _wilson_ci(oa, n_samples)

    # Kappa with standard error
    kappa_se = _kappa_standard_error(kappa, n_samples, k=len(labels))
    kappa_ci_low = max(-1.0, kappa - 1.96 * kappa_se)
    kappa_ci_high = min(1.0, kappa + 1.96 * kappa_se)

    # Kappa significance test (z-test against H0: kappa=0)
    kappa_z = kappa / max(kappa_se, 1e-9)
    kappa_p = 2 * (1 - _standard_normal_cdf(abs(kappa_z)))

    report = {
        "available": True,
        "confusion_matrix": cm,
        "labels": labels,
        "n_samples": n_samples,
        "overall_accuracy": round(oa, 4),
        "overall_accuracy_95ci": [round(oa_ci_low, 4), round(oa_ci_high, 4)],
        "kappa": round(kappa, 4),
        "kappa_se": round(kappa_se, 4),
        "kappa_95ci": [round(kappa_ci_low, 4), round(kappa_ci_high, 4)],
        "kappa_z_statistic": round(kappa_z, 4),
        "kappa_p_value": round(kappa_p, 6),
        "kappa_significant": kappa_p < 0.05,
        "per_class_metrics": per_class,
        "method": kappa_results.get("method", "unknown"),
    }

    return report


def format_report_as_markdown(report, area_stats=None, epoch_year=None):
    """
    Format the full accuracy report as publication-ready markdown.
    """
    lines = []
    lines.append("# Accuracy Assessment Report")
    lines.append(f"**Epoch**: {epoch_year or 'N/A'}")
    lines.append(f"**Scale**: {config.TARGET_SCALE}m")
    lines.append(f"**AOI**: {config.AOI.get('name', 'Godavari Delta')}")
    lines.append("")

    # Overall metrics
    if report.get("available"):
        lines.append("## Overall Classification Accuracy")
        lines.append("")
        lines.append("| Metric | Value | 95% CI |")
        lines.append("|--------|-------|--------|")

        oa = report.get("overall_accuracy", 0)
        oa_ci = report.get("overall_accuracy_95ci", [0, 0])
        lines.append(f"| Overall Accuracy | {oa:.2%} | [{oa_ci[0]:.2%}, {oa_ci[1]:.2%}] |")

        kappa = report.get("kappa", 0)
        kappa_ci = report.get("kappa_95ci", [0, 0])
        lines.append(f"| Cohen's Kappa | {kappa:.4f} | [{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}] |")

        n = report.get("n_samples", 0)
        lines.append(f"| N (samples) | {n} | - |")

        p_val = report.get("kappa_p_value", 1.0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        lines.append(f"| Kappa p-value | {p_val:.6f} | {sig} |")
        lines.append("")

        # Per-class metrics table
        per_class = report.get("per_class_metrics", {})
        if per_class:
            lines.append("## Per-Stage Classification Metrics")
            lines.append("")
            lines.append("| Stage | Precision | Recall | F1-Score |")
            lines.append("|-------|-----------|--------|----------|")
            for stage_key in sorted(per_class.keys()):
                m = per_class[stage_key]
                stage_num = int(stage_key.replace("S", "")) if stage_key.startswith("S") else 0
                name = STAGE_NAMES.get(stage_num, stage_key)
                lines.append(
                    f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
                )
            lines.append("")

        # Confusion matrix
        cm = report.get("confusion_matrix", [])
        labels = report.get("labels", [])
        if cm and labels:
            lines.append("## Confusion Matrix")
            lines.append("")
            header = "| Reference \\ Predicted |"
            for lbl in labels:
                header += f" S{lbl} |"
            header += " Total |"
            lines.append(header)

            sep = "|" + "---|" * (len(labels) + 2)
            lines.append(sep)

            for i, row in enumerate(cm):
                line = f"| **S{labels[i]}** |"
                for val in row:
                    line += f" {val} |"
                line += f" {sum(row)} |"
                lines.append(line)

            col_totals = [sum(cm[i][j] for i in range(len(cm))) for j in range(len(cm[0]))] if cm else []
            total_line = "| **Total** |"
            for ct in col_totals:
                total_line += f" {ct} |"
            total_line += f" {sum(col_totals)} |"
            lines.append(total_line)
            lines.append("")

    # Area statistics
    if area_stats and area_stats.get("area_table"):
        lines.append("## Stage Area Statistics")
        lines.append("")
        lines.append("| Year | S1 (ha) | S2 (ha) | S3 (ha) | S4 (ha) | S5 (ha) | Total (ha) |")
        lines.append("|------|---------|---------|---------|---------|---------|------------|")
        for row in area_stats["area_table"]:
            lines.append(
                f"| {row['year']} | {row.get('S1_ha', 0):.1f} | {row.get('S2_ha', 0):.1f} | "
                f"{row.get('S3_ha', 0):.1f} | {row.get('S4_ha', 0):.1f} | "
                f"{row.get('S5_ha', 0):.1f} | {row.get('total_area_ha', 0):.1f} |"
            )
        lines.append("")

        change = area_stats.get("change_stats", {})
        if change:
            lines.append("### Net Area Change (First to Last Epoch)")
            lines.append("")
            lines.append("| Stage | Delta Area (ha) | Delta (%) |")
            lines.append("|-------|-----------------|-----------|")
            for stage_key in sorted(change.keys()):
                c = change[stage_key]
                stage_num = int(stage_key.replace("S", "")) if stage_key.startswith("S") else 0
                sign = "+" if c["change_ha"] >= 0 else ""
                lines.append(
                    f"| {STAGE_NAMES.get(stage_num, stage_key)} | "
                    f"{sign}{c['change_ha']:.1f} | {sign}{c['change_pct']:.1f}% |"
                )
            lines.append("")

    lines.append("---")
    lines.append("*Generated by Aquaculture Detection Pipeline v15.0*")
    lines.append(f"*Method: {report.get('method', 'gmw_independent_stratified')}*")

    return "\n".join(lines)


def save_publication_report(
    kappa_results,
    gmw_results=None,
    jrc_results=None,
    epoch_results=None,
    epoch_year=None,
):
    """
    Generate and save the full publication report.

    Args:
        kappa_results: from m17_stratified_gt
        gmw_results: from m11 compare_with_gmw
        jrc_results: from m11 compare_with_jrc
        epoch_results: list of epoch dicts for area statistics
        epoch_year: current epoch year
    """
    cm_report = generate_confusion_matrix_report(kappa_results)

    area_stats = None
    if epoch_results:
        area_stats = compute_area_statistics(epoch_results)

    full_report = {
        "classification_accuracy": cm_report,
        "gmw_comparison": gmw_results or {},
        "jrc_comparison": jrc_results or {},
        "area_statistics": area_stats or {},
        "epoch_year": epoch_year,
        "pipeline_version": "v15.0",
    }

    # Save as JSON
    json_path = os.path.join(config.STATS_DIR, "publication_report.json")
    os.makedirs(config.STATS_DIR, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    # Save as markdown
    md_content = format_report_as_markdown(cm_report, area_stats, epoch_year)
    md_path = os.path.join(config.STATS_DIR, "publication_report.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    # Also save to web data
    web_json_path = os.path.join(config.WEB_DATA_DIR, "publication_report.json")
    os.makedirs(config.WEB_DATA_DIR, exist_ok=True)
    with open(web_json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    pub_log.info(f"Publication report saved: {json_path}")
    print(f"[M18] Publication report saved: {os.path.basename(json_path)}")
    print(f"[M18] Markdown report: {os.path.basename(md_path)}")

    if cm_report.get("available"):
        oa = cm_report.get("overall_accuracy", 0)
        kappa = cm_report.get("kappa", 0)
        n = cm_report.get("n_samples", 0)
        p_val = cm_report.get("kappa_p_value", 1.0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"[M18] OA={oa:.2%}, Kappa={kappa:.4f} ({sig}), N={n}")

    return full_report

"""
Integrated M1 Data Quality Pipeline.

Orchestrates all M1 detectors per consumer, produces cleaned data and quality reports.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .ami import compute_interval_quality_score
from .anomaly import (
    detect_frozen_readings,
    detect_outliers_contextual,
    detect_outliers_iqr,
    detect_outliers_rolling,
    detect_outliers_zscore,
)
from .apfc import APFCSwitchingDetector
from .dg import DGTransitionDetector
from .imputation import impute_missing_and_anomalous
from .noise import DemandNoiseFilter


@dataclass
class QualityReport:
    """Summary of data quality for a single consumer."""

    consumer_id: str
    total_intervals: int
    missing_pct: float
    frozen_pct: float
    zscore_outlier_pct: float
    iqr_outlier_pct: float
    contextual_outlier_pct: float
    rolling_outlier_pct: float
    ct_artefact_pct: float
    pf_artefact_pct: float
    dg_period_pct: float
    apfc_event_count: int
    total_anomaly_pct: float
    imputed_pct: float
    mean_quality_score: float
    channels_complete_pct: float


def run_quality_pipeline(
    df: pd.DataFrame,
    demand_col: str = "demand_kwh",
    timestamp_col: str = "timestamp",
    consumer_col: str = "consumer_id",
    kva_col: Optional[str] = None,
    kw_col: Optional[str] = None,
    pf_col: Optional[str] = None,
    voltage_col: Optional[str] = None,
    frequency_col: Optional[str] = None,
    grid_import_col: Optional[str] = None,
    site_load_col: Optional[str] = None,
    freq: str = "15min",
) -> Tuple[pd.DataFrame, List[QualityReport]]:
    """
    Full M1 quality pipeline: detect issues, clean, impute, score.

    Orchestrates all M1 detectors:
    1. Basic anomaly detection (frozen, z-score, IQR, contextual, rolling)
    2. Demand noise filter (CT artefacts, PF artefacts) — if kVA/kW available
    3. DG transition detection — if grid_import available
    4. APFC event detection — if kVA/kW/PF available
    5. Imputation of flagged intervals
    6. Per-interval quality scoring

    Returns:
        Tuple of (cleaned DataFrame, list of QualityReport per consumer)
    """
    logger.info("═══ M1 Data Quality Pipeline ═══")

    # Instantiate detectors ONCE outside the loop (stateless, reusable)
    noise_filter = DemandNoiseFilter()
    dg_detector = DGTransitionDetector()
    apfc_detector = APFCSwitchingDetector()

    cleaned_dfs = []
    quality_reports = []

    for cid, group in df.groupby(consumer_col):
        logger.info(f"Processing consumer: {cid}")
        group = group.sort_values(timestamp_col).copy()

        # Ensure datetime index
        if timestamp_col in group.columns:
            group = group.set_index(timestamp_col)
        if not isinstance(group.index, pd.DatetimeIndex):
            group.index = pd.to_datetime(group.index)

        series = group[demand_col]

        # ── 1. Basic anomaly detection ──────────────────────────────────
        frozen = detect_frozen_readings(series)
        zscore_outliers = detect_outliers_zscore(series)
        iqr_outliers = detect_outliers_iqr(series)
        contextual_outliers = detect_outliers_contextual(series)
        rolling_outliers = detect_outliers_rolling(series)

        basic_anomaly = frozen | zscore_outliers | iqr_outliers | contextual_outliers | rolling_outliers

        # ── 2. Demand noise filter ──────────────────────────────────────
        ct_artefacts = pd.Series(False, index=group.index)
        pf_artefacts = pd.Series(False, index=group.index)

        has_kva_kw = kva_col and kw_col and kva_col in group.columns and kw_col in group.columns
        if has_kva_kw:
            # Clear cache for new consumer
            noise_filter._cache_key = None

            freq_series = group[frequency_col] if (frequency_col and frequency_col in group.columns) else None
            pf_series = group[pf_col] if (pf_col and pf_col in group.columns) else None

            ct_artefacts = noise_filter.detect_ct_artefacts(group[kva_col], group[kw_col], freq_series)
            pf_artefacts = noise_filter.detect_pf_artefacts(group[kva_col], group[kw_col], pf_series)

        # ── 3. DG transition detection ──────────────────────────────────
        dg_mask = pd.Series(False, index=group.index)

        has_grid_import = grid_import_col and grid_import_col in group.columns
        if has_grid_import:
            site_load = group[site_load_col] if (site_load_col and site_load_col in group.columns) else None
            voltage = group[voltage_col] if (voltage_col and voltage_col in group.columns) else None

            dg_result = dg_detector.mark_dg_periods(group[grid_import_col], site_load, voltage)
            dg_mask = dg_result["is_dg"]
            group["is_dg"] = dg_mask
            group["dg_event_type"] = dg_result["dg_event_type"]
            group["dg_period_id"] = dg_result["dg_period_id"]
            group["dg_confidence"] = dg_result["dg_confidence"]

        # ── 4. APFC detection ───────────────────────────────────────────
        apfc_count = 0

        has_pf = pf_col and pf_col in group.columns
        if has_kva_kw and has_pf:
            apfc_result = apfc_detector.classify_apfc_events(
                group[kva_col], group[kw_col], group[pf_col]
            )
            group["is_apfc_event"] = apfc_result["is_apfc_event"]
            group["apfc_event_type"] = apfc_result["apfc_event_type"]
            apfc_count = apfc_result["is_apfc_event"].sum()

        # ── 5. Combined anomaly mask & imputation ───────────────────────
        combined_anomaly = basic_anomaly | ct_artefacts | pf_artefacts

        # DG periods are NOT imputed — they're excluded from training entirely
        clean_series = impute_missing_and_anomalous(
            series, anomaly_mask=combined_anomaly, method="hybrid"
        )

        # ── 6. Build output ─────────────────────────────────────────────
        clean_group = group.copy()
        clean_group[demand_col] = clean_series
        clean_group[f"{demand_col}_original"] = series
        clean_group["is_anomaly"] = combined_anomaly
        clean_group["is_imputed"] = combined_anomaly | series.isna()
        clean_group["is_ct_artefact"] = ct_artefacts
        clean_group["is_pf_artefact"] = pf_artefacts

        # Use the proper quality scoring function
        quality_score = compute_interval_quality_score(clean_group)
        clean_group["quality_score"] = quality_score

        cleaned_dfs.append(clean_group.reset_index())

        # ── 7. Quality report ───────────────────────────────────────────
        report = QualityReport(
            consumer_id=str(cid),
            total_intervals=len(series),
            missing_pct=float(series.isna().mean() * 100),
            frozen_pct=float(frozen.mean() * 100),
            zscore_outlier_pct=float(zscore_outliers.mean() * 100),
            iqr_outlier_pct=float(iqr_outliers.mean() * 100),
            contextual_outlier_pct=float(contextual_outliers.mean() * 100),
            rolling_outlier_pct=float(rolling_outliers.mean() * 100),
            ct_artefact_pct=float(ct_artefacts.mean() * 100),
            pf_artefact_pct=float(pf_artefacts.mean() * 100),
            dg_period_pct=float(dg_mask.mean() * 100),
            apfc_event_count=int(apfc_count),
            total_anomaly_pct=float(combined_anomaly.mean() * 100),
            imputed_pct=float((combined_anomaly | series.isna()).mean() * 100),
            mean_quality_score=float(quality_score.mean()),
            channels_complete_pct=100.0,  # Updated when full AMI channels are available
        )
        quality_reports.append(report)

        logger.info(
            f"  {cid}: {report.total_anomaly_pct:.1f}% anomalous, "
            f"{report.dg_period_pct:.1f}% DG, "
            f"{report.apfc_event_count} APFC events, "
            f"quality={report.mean_quality_score:.2f}"
        )

    cleaned = pd.concat(cleaned_dfs, ignore_index=True)

    logger.info(f"═══ Pipeline complete: {len(quality_reports)} consumers processed ═══")

    return cleaned, quality_reports

"""
Commercial quantification layer (spec §C5).

Translates dispatch + forecast → ₹/day, IRR, FLS quote, audit-ready brief.
"""

from .irr import IRRResult, compute_project_irr, irr_heatmap, sensitivity_grid
from .quote import FLSQuote, FLSQuoteGenerator
from .brief import build_substation_brief, BriefContent

__all__ = [
    "IRRResult",
    "compute_project_irr",
    "irr_heatmap",
    "sensitivity_grid",
    "FLSQuote",
    "FLSQuoteGenerator",
    "build_substation_brief",
    "BriefContent",
]

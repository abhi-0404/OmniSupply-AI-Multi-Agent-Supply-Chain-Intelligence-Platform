"""
Data quality validators for OmniSupply platform.
"""

import logging
from typing import Dict, List, Any

from ..models import DataQualityResult

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Runs basic data quality checks on loaded datasets."""

    def check_all(self, data: Dict[str, List[Any]]) -> Dict[str, DataQualityResult]:
        results = {}
        for name, records in data.items():
            results[name] = self._check_dataset(name, records)
        return results

    def _check_dataset(self, name: str, records: List[Any]) -> DataQualityResult:
        issues = []
        total = len(records)

        if total == 0:
            return DataQualityResult(
                dataset_name=name,
                status="FAILED",
                issues_found=1,
                issues=["Dataset is empty"],
                records_checked=0,
                records_valid=0,
            )

        null_count = 0
        for rec in records:
            d = rec.model_dump() if hasattr(rec, "model_dump") else vars(rec)
            if all(v is None for v in d.values()):
                null_count += 1

        if null_count > 0:
            issues.append(f"{null_count} records with all-null fields")

        status = "PASSED" if not issues else "WARNING"
        return DataQualityResult(
            dataset_name=name,
            status=status,
            issues_found=len(issues),
            issues=issues,
            records_checked=total,
            records_valid=total - null_count,
        )

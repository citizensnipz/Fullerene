"""Typed models for deterministic Fullerene verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


class VerificationStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class VerificationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class VerificationResult:
    check_name: str
    status: VerificationStatus
    severity: VerificationSeverity
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationResult":
        return cls(
            check_name=data["check_name"],
            status=VerificationStatus(data["status"]),
            severity=VerificationSeverity(data["severity"]),
            message=data["message"],
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class VerificationSummary:
    overall_status: VerificationStatus
    results: list[VerificationResult] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "results": [result.to_dict() for result in self.results],
            "failed_checks": list(self.failed_checks),
            "warnings": list(self.warnings),
            "metadata": _serialize_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationSummary":
        return cls(
            overall_status=VerificationStatus(data["overall_status"]),
            results=[
                VerificationResult.from_dict(result)
                for result in data.get("results", [])
            ],
            failed_checks=list(data.get("failed_checks", [])),
            warnings=list(data.get("warnings", [])),
            metadata=data.get("metadata", {}),
        )

"""Public Fullerene runtime API."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AdjustmentRecord",
    "AdjustmentStatus",
    "AdjustmentTarget",
    "AttentionFacet",
    "AttentionItem",
    "AttentionResult",
    "AttentionSource",
    "BehaviorFacet",
    "Belief",
    "BeliefSource",
    "BeliefStatus",
    "ContextFacet",
    "ContextItem",
    "ContextItemType",
    "ContextWindow",
    "DecisionAction",
    "DeterministicPlanBuilder",
    "EchoFacet",
    "ExecutorFacet",
    "Event",
    "EventType",
    "ExecutionMode",
    "ExecutionRecord",
    "ExecutionResult",
    "ExecutionStatus",
    "Facet",
    "FacetResult",
    "FileStateStore",
    "FixedWeightAttentionScorer",
    "Goal",
    "GoalSource",
    "GoalStatus",
    "GoalsFacet",
    "GoalStore",
    "InMemoryStateStore",
    "InternalActionExecutor",
    "LearningFacet",
    "LearningResult",
    "LearningSignal",
    "MemoryFacet",
    "MemoryRecord",
    "MemoryStore",
    "MemoryType",
    "Nexus",
    "NexusDecision",
    "NexusRecord",
    "NexusRuntime",
    "NexusState",
    "Plan",
    "PlannerFacet",
    "PlanStatus",
    "PlanStep",
    "PlanStepStatus",
    "PolicyFacet",
    "PolicyRule",
    "PolicyRuleType",
    "PolicySource",
    "PolicyStatus",
    "PolicyStore",
    "PolicyTargetType",
    "RiskLevel",
    "ActionType",
    "SignalSource",
    "SignalType",
    "VerificationResult",
    "VerificationSeverity",
    "VerificationStatus",
    "VerificationSummary",
    "VerifierFacet",
    "SQLiteGoalStore",
    "SQLiteMemoryStore",
    "SQLitePolicyStore",
    "SQLiteWorldModelStore",
    "StaticContextAssembler",
    "StateStore",
    "WorldModelFacet",
    "WorldModelStore",
]


def __getattr__(name: str):
    if name in {"Facet"}:
        return getattr(import_module("fullerene.facets.base"), name)
    if name in {
        "AttentionFacet",
        "BehaviorFacet",
        "ContextFacet",
        "EchoFacet",
        "ExecutorFacet",
        "GoalsFacet",
        "LearningFacet",
        "MemoryFacet",
        "PlannerFacet",
        "PolicyFacet",
        "VerifierFacet",
        "WorldModelFacet",
    }:
        return getattr(import_module("fullerene.facets"), name)
    if name in {
        "AttentionItem",
        "AttentionResult",
        "AttentionSource",
        "FixedWeightAttentionScorer",
    }:
        return getattr(import_module("fullerene.attention"), name)
    if name in {
        "AdjustmentRecord",
        "AdjustmentStatus",
        "AdjustmentTarget",
        "LearningResult",
        "LearningSignal",
        "SignalSource",
        "SignalType",
    }:
        return getattr(import_module("fullerene.learning"), name)
    if name in {
        "ActionType",
        "ExecutionMode",
        "ExecutionRecord",
        "ExecutionResult",
        "ExecutionStatus",
        "InternalActionExecutor",
    }:
        return getattr(import_module("fullerene.executor"), name)
    if name in {
        "DeterministicPlanBuilder",
        "Plan",
        "PlanStatus",
        "PlanStep",
        "PlanStepStatus",
        "RiskLevel",
    }:
        return getattr(import_module("fullerene.planner"), name)
    if name in {
        "ContextItem",
        "ContextItemType",
        "ContextWindow",
        "StaticContextAssembler",
    }:
        return getattr(import_module("fullerene.context"), name)
    if name in {"Goal", "GoalSource", "GoalStatus", "GoalStore", "SQLiteGoalStore"}:
        return getattr(import_module("fullerene.goals"), name)
    if name in {"MemoryRecord", "MemoryStore", "MemoryType", "SQLiteMemoryStore"}:
        return getattr(import_module("fullerene.memory"), name)
    if name in {
        "PolicyRule",
        "PolicyRuleType",
        "PolicySource",
        "PolicyStatus",
        "PolicyStore",
        "PolicyTargetType",
        "SQLitePolicyStore",
    }:
        return getattr(import_module("fullerene.policy"), name)
    if name in {
        "VerificationResult",
        "VerificationSeverity",
        "VerificationStatus",
        "VerificationSummary",
    }:
        return getattr(import_module("fullerene.verifier"), name)
    if name in {
        "Belief",
        "BeliefSource",
        "BeliefStatus",
        "SQLiteWorldModelStore",
        "WorldModelStore",
    }:
        return getattr(import_module("fullerene.world_model"), name)
    if name in {
        "DecisionAction",
        "Event",
        "EventType",
        "FacetResult",
        "NexusDecision",
        "NexusRecord",
        "NexusState",
    }:
        return getattr(import_module("fullerene.nexus.models"), name)
    if name in {"Nexus", "NexusRuntime"}:
        return getattr(import_module("fullerene.nexus.runtime"), name)
    if name in {"FileStateStore", "InMemoryStateStore", "StateStore"}:
        return getattr(import_module("fullerene.state.store"), name)
    raise AttributeError(f"module 'fullerene' has no attribute {name!r}")

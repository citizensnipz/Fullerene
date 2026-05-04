"""Context assembly for Fullerene Context v0 and v1."""

from __future__ import annotations

from typing import Any, Sequence

from fullerene.context.models import (
    STATIC_RECENT_EPISODIC_V0,
    ContextAssemblyConfig,
    ContextItem,
    ContextItemType,
    ContextWindow,
)
from fullerene.goals import GoalStore
from fullerene.goals.normalization import GoalDeduplicationResult, dedupe_active_goals
from fullerene.memory import (
    MemoryRecord,
    MemoryStore,
    MemoryType,
    extract_event_tags,
    tokenize,
)
from fullerene.nexus.models import Event, FacetResult, NexusState
from fullerene.policy import (
    PolicyRule,
    PolicyRuleType,
    PolicyStore,
    PolicyTargetType,
)
from fullerene.world_model import Belief, WorldModelStore

EXTERNAL_POLICY_TARGETS = (
    PolicyTargetType.FILE_WRITE,
    PolicyTargetType.FILE_DELETE,
    PolicyTargetType.SHELL,
    PolicyTargetType.NETWORK,
    PolicyTargetType.MESSAGE,
    PolicyTargetType.GIT,
    PolicyTargetType.TOOL,
)


class StaticContextAssembler:
    """Build a small, deterministic context window from recent episodic memory."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        max_items: int = 5,
    ) -> None:
        self.store = store
        self.max_items = max(int(max_items), 1)

    def assemble(
        self,
        recent_records: Sequence[MemoryRecord] | None = None,
    ) -> ContextWindow:
        records = self._load_recent_records(recent_records)
        items = [self._memory_to_context_item(record) for record in records]
        included_memory_ids = [item.id for item in items]
        source_types = [ContextItemType.MEMORY.value] if items else []

        return ContextWindow(
            items=items,
            max_items=self.max_items,
            strategy=STATIC_RECENT_EPISODIC_V0,
            metadata={
                "source_types": source_types,
                "source_memory_type": MemoryType.EPISODIC.value,
                "item_count": len(items),
                "included_goal_ids": [],
                "included_memory_ids": included_memory_ids,
                "included_belief_ids": [],
                "salience_threshold": 0.0,
                "limits": {
                    "max_goals": 0,
                    "max_memories": self.max_items,
                    "max_beliefs": 0,
                },
                "reasons": (
                    ["static_recent_episodic_v0"]
                    if items
                    else ["no_recent_episodic_memories"]
                ),
            },
        )

    def _load_recent_records(
        self,
        recent_records: Sequence[MemoryRecord] | None,
    ) -> list[MemoryRecord]:
        if recent_records is not None:
            episodic_records = [
                record
                for record in recent_records
                if record.memory_type == MemoryType.EPISODIC
            ]
            return list(episodic_records[: self.max_items])
        if self.store is None:
            return []
        return self.store.list_recent(
            limit=self.max_items,
            memory_type=MemoryType.EPISODIC,
        )

    @staticmethod
    def _memory_to_context_item(record: MemoryRecord) -> ContextItem:
        return ContextItem(
            id=record.id,
            item_type=ContextItemType.MEMORY,
            content=record.content,
            source_id=record.source_event_id,
            created_at=record.created_at,
            metadata={
                "memory_type": record.memory_type.value,
                "salience": record.salience,
                "confidence": record.confidence,
                "tags": list(record.tags),
                "memory_metadata": dict(record.metadata),
            },
        )


class DynamicContextAssembler:
    """Build a bounded working packet from the current event and active state."""

    def __init__(
        self,
        *,
        memory_store: MemoryStore | None = None,
        goal_store: GoalStore | None = None,
        world_model_store: WorldModelStore | None = None,
        policy_store: PolicyStore | None = None,
        config: ContextAssemblyConfig | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.goal_store = goal_store
        self.world_model_store = world_model_store
        self.policy_store = policy_store
        self.config = config or ContextAssemblyConfig()

    def assemble(
        self,
        *,
        event: Event,
        state: NexusState | None = None,
        facet_results: Sequence[FacetResult] | None = None,
    ) -> ContextWindow:
        working_state = state or NexusState()
        items: list[ContextItem] = []
        reasons: list[str] = ["included_current_event"]

        event_item = self._event_item(event)
        items.append(event_item)

        goal_items, goal_deduplication = self._goal_items()
        items.extend(goal_items)
        reasons.append(f"included_goals={len(goal_items)}")
        if goal_deduplication.deduped_goal_count > 0:
            reasons.append(f"deduped_goals={goal_deduplication.deduped_goal_count}")

        relevant_memory_items, recent_memory_items = self._memory_items(event)
        items.extend(relevant_memory_items)
        items.extend(recent_memory_items)
        reasons.append(f"included_relevant_memories={len(relevant_memory_items)}")
        reasons.append(f"included_recent_memories={len(recent_memory_items)}")

        belief_items = self._belief_items(event)
        items.extend(belief_items)
        reasons.append(f"included_beliefs={len(belief_items)}")

        policy_item = self._policy_item()
        if policy_item is not None:
            items.append(policy_item)
            reasons.append("included_policy_summary")
        elif self.config.include_policy_summary:
            reasons.append("policy_summary_unavailable")

        signal_items = self._signal_items(
            state=working_state,
            facet_results=facet_results,
        )
        items.extend(signal_items)
        reasons.append(f"included_signal_summaries={len(signal_items)}")

        metadata = {
            "source_types": self._source_types(items),
            "item_count": len(items),
            "included_goal_ids": [item.id for item in goal_items],
            "deduped_goal_count": goal_deduplication.deduped_goal_count,
            "deduped_goal_ids": list(goal_deduplication.deduped_goal_ids),
            "normalized_goal_keys": list(goal_deduplication.normalized_goal_keys),
            "included_memory_ids": [
                item.id for item in [*relevant_memory_items, *recent_memory_items]
            ],
            "included_belief_ids": [item.id for item in belief_items],
            "salience_threshold": self.config.salience_threshold,
            "limits": {
                "max_goals": self.config.max_goals,
                "max_memories": self.config.max_memories,
                "max_beliefs": self.config.max_beliefs,
            },
            "config": self.config.to_dict(),
            "reasons": reasons,
        }
        return ContextWindow(
            items=items,
            max_items=self.config.max_items,
            strategy=self.config.strategy,
            metadata=metadata,
        )

    def _event_item(self, event: Event) -> ContextItem:
        return ContextItem(
            id=event.event_id,
            item_type=ContextItemType.EVENT,
            content=event.content,
            source_id=event.event_id,
            created_at=event.timestamp,
            metadata={
                "context_source": "current_event",
                "event_type": event.event_type.value,
                "event_metadata": dict(event.metadata),
            },
        )

    def _goal_items(self) -> tuple[list[ContextItem], GoalDeduplicationResult]:
        if self.goal_store is None or self.config.max_goals == 0:
            return [], dedupe_active_goals([], limit=0)
        goal_fetch_limit = max(self.config.max_goals * 5, self.config.max_goals, 10)
        goals = self.goal_store.list_active_goals(limit=goal_fetch_limit)
        deduped_goals = dedupe_active_goals(goals, limit=self.config.max_goals)
        return (
            [
                ContextItem(
                    id=goal.id,
                    item_type=ContextItemType.GOAL,
                    content=goal.description,
                    source_id=goal.id,
                    created_at=goal.updated_at,
                    metadata={
                        "context_source": "active_goal",
                        "priority": goal.priority,
                        "status": goal.status.value,
                        "tags": list(goal.tags),
                        "source": goal.source.value,
                        "goal_metadata": dict(goal.metadata),
                    },
                )
                for goal in deduped_goals.goals
            ],
            deduped_goals,
        )

    def _memory_items(self, event: Event) -> tuple[list[ContextItem], list[ContextItem]]:
        if self.memory_store is None or self.config.max_memories == 0:
            return [], []

        remaining = self.config.max_memories
        relevant_records = self._filter_memory_records(
            self.memory_store.retrieve_relevant(event, limit=self.config.max_memories)
        )
        recent_records = self._filter_memory_records(
            self.memory_store.list_recent(
                limit=self.config.max_memories,
                memory_type=MemoryType.EPISODIC,
            )
        )

        deduped_relevant: list[MemoryRecord] = []
        seen_memory_ids: set[str] = set()
        for record in relevant_records:
            if record.id in seen_memory_ids:
                continue
            deduped_relevant.append(record)
            seen_memory_ids.add(record.id)
            remaining -= 1
            if remaining <= 0:
                break

        deduped_recent: list[MemoryRecord] = []
        if remaining > 0:
            for record in recent_records:
                if record.id in seen_memory_ids:
                    continue
                deduped_recent.append(record)
                seen_memory_ids.add(record.id)
                remaining -= 1
                if remaining <= 0:
                    break

        return (
            [
                self._memory_to_context_item(record, context_source="relevant")
                for record in deduped_relevant
            ],
            [
                self._memory_to_context_item(record, context_source="recent")
                for record in deduped_recent
            ],
        )

    def _filter_memory_records(
        self,
        records: Sequence[MemoryRecord],
    ) -> list[MemoryRecord]:
        filtered: list[MemoryRecord] = []
        for record in records:
            if record.memory_type != MemoryType.EPISODIC:
                continue
            if record.salience < self.config.salience_threshold:
                continue
            filtered.append(record)
        return filtered

    def _belief_items(self, event: Event) -> list[ContextItem]:
        if self.world_model_store is None or self.config.max_beliefs == 0:
            return []
        beliefs = self.world_model_store.list_active_beliefs(
            limit=max(self.config.max_beliefs * 2, self.config.max_beliefs)
        )
        ranked_beliefs = sorted(
            beliefs,
            key=lambda belief: self._belief_sort_key(event, belief),
            reverse=True,
        )[: self.config.max_beliefs]
        return [
            ContextItem(
                id=belief.id,
                item_type=ContextItemType.BELIEF,
                content=belief.claim,
                source_id=belief.id,
                created_at=belief.updated_at,
                metadata={
                    "context_source": "active_belief",
                    "confidence": belief.confidence,
                    "status": belief.status.value,
                    "tags": list(belief.tags),
                    "source": belief.source.value,
                    "belief_metadata": dict(belief.metadata),
                },
            )
            for belief in ranked_beliefs
        ]

    def _belief_sort_key(
        self,
        event: Event,
        belief: Belief,
    ) -> tuple[float, float, float, str]:
        event_tags = extract_event_tags(event)
        event_keywords = tokenize(event.content)
        belief_tags = set(belief.tags)
        belief_keywords = tokenize(belief.claim)
        shared_tags = event_tags & belief_tags
        shared_keywords = event_keywords & belief_keywords
        tag_overlap = len(shared_tags) / len(belief_tags) if belief_tags else 0.0
        keyword_overlap = (
            len(shared_keywords) / len(belief_keywords) if belief_keywords else 0.0
        )
        relevance_score = round(tag_overlap + keyword_overlap + belief.confidence, 3)
        return (
            1.0 if shared_tags or shared_keywords else 0.0,
            relevance_score,
            belief.confidence,
            belief.id,
        )

    def _policy_item(self) -> ContextItem | None:
        if not self.config.include_policy_summary or self.policy_store is None:
            return None
        enabled_policies = self._list_enabled_policies(limit=20)
        enabled_policy_count = self._count_enabled_policies(enabled_policies)
        major_constraints = self._major_policy_constraints(enabled_policies)
        default_constraint = "External side effects require approval by default."
        if default_constraint not in major_constraints:
            major_constraints.insert(0, default_constraint)
        content = (
            f"Enabled policies: {enabled_policy_count}. "
            f"Constraints: {'; '.join(major_constraints[:4])}"
        )
        return ContextItem(
            id="policy-summary",
            item_type=ContextItemType.POLICY,
            content=content,
            source_id="policy-store",
            metadata={
                "context_source": "policy_summary",
                "enabled_policy_count": enabled_policy_count,
                "major_constraints": major_constraints[:4],
            },
        )

    def _signal_items(
        self,
        *,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> list[ContextItem]:
        if not self.config.include_signal_summaries:
            return []
        items: list[ContextItem] = []
        planner_item = self._planner_signal_item(state, facet_results)
        if planner_item is not None:
            items.append(planner_item)
        executor_item = self._executor_signal_item(state, facet_results)
        if executor_item is not None:
            items.append(executor_item)
        attention_item = self._attention_signal_item(state, facet_results)
        if attention_item is not None:
            items.append(attention_item)
        affect_item = self._affect_signal_item(state, facet_results)
        if affect_item is not None:
            items.append(affect_item)
        learning_item = self._learning_signal_item(state, facet_results)
        if learning_item is not None:
            items.append(learning_item)
        return items

    def _planner_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        planner_state = self._facet_bucket("planner", state, facet_results)
        if not planner_state:
            return None
        last_plan = planner_state.get("last_plan")
        if not isinstance(last_plan, dict):
            return None
        raw_steps = last_plan.get("steps")
        step_count = len(raw_steps) if isinstance(raw_steps, list) else 0
        confidence = self._numeric_value(planner_state.get("last_plan_confidence"))
        trigger_reason = (
            self._coerce_string(planner_state.get("last_trigger_reason")) or "none"
        )
        content = (
            f"Planner: {step_count} step(s), confidence {confidence:.2f}, "
            f"trigger {trigger_reason}."
        )
        return ContextItem(
            id="signal-planner",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="planner",
            metadata={"signal_type": "planner"},
        )

    def _executor_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        executor_state = self._facet_bucket("executor", state, facet_results)
        if not executor_state:
            return None
        execution_result = executor_state.get("last_execution_result")
        if not isinstance(execution_result, dict):
            return None
        overall_status = (
            self._coerce_string(execution_result.get("overall_status")) or "unknown"
        )
        dry_run = bool(execution_result.get("dry_run", True))
        content = f"Executor: {overall_status}, {'dry-run' if dry_run else 'live'}."
        return ContextItem(
            id="signal-executor",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="executor",
            metadata={"signal_type": "executor"},
        )

    def _attention_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        attention_state = self._facet_bucket("attention", state, facet_results)
        if not attention_state:
            return None
        dominant_source = self._coerce_string(
            attention_state.get("last_dominant_source")
        )
        raw_focus_ids = attention_state.get("last_focus_item_ids")
        focus_count = len(raw_focus_ids) if isinstance(raw_focus_ids, list) else 0
        if dominant_source is None and focus_count == 0:
            return None
        content = (
            f"Attention: dominant source {dominant_source or 'none'}, "
            f"{focus_count} focus item(s)."
        )
        return ContextItem(
            id="signal-attention",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="attention",
            metadata={"signal_type": "attention"},
        )

    def _affect_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        affect_state = self._facet_bucket("affect", state, facet_results)
        if not affect_state:
            return None
        snapshot = affect_state.get("last_affect_state")
        if not isinstance(snapshot, dict):
            return None
        valence = self._numeric_value(snapshot.get("valence"), allow_negative=True)
        arousal = self._numeric_value(snapshot.get("arousal"))
        dominance = self._numeric_value(snapshot.get("dominance"))
        novelty = self._numeric_value(snapshot.get("novelty"))
        content = (
            f"Affect: V={valence:.2f}, A={arousal:.2f}, "
            f"D={dominance:.2f}, N={novelty:.2f}."
        )
        return ContextItem(
            id="signal-affect",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="affect",
            metadata={"signal_type": "affect"},
        )

    def _learning_signal_item(
        self,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> ContextItem | None:
        learning_state = self._facet_bucket("learning", state, facet_results)
        if not learning_state:
            return None
        signal_count = int(learning_state.get("last_signal_count", 0) or 0)
        proposal_count = int(learning_state.get("last_proposal_count", 0) or 0)
        adjustment_count = int(learning_state.get("last_adjustment_count", 0) or 0)
        if signal_count == 0 and proposal_count == 0 and adjustment_count == 0:
            return None
        content = (
            f"Learning: {signal_count} signal(s), {adjustment_count} adjustment(s), "
            f"{proposal_count} proposal(s)."
        )
        return ContextItem(
            id="signal-learning",
            item_type=ContextItemType.SIGNAL,
            content=content,
            source_id="learning",
            metadata={"signal_type": "learning"},
        )

    def _facet_bucket(
        self,
        facet_name: str,
        state: NexusState,
        facet_results: Sequence[FacetResult] | None,
    ) -> dict[str, Any] | None:
        if facet_results is not None:
            for result in reversed(facet_results):
                if result.facet_name != facet_name:
                    continue
                if result.state_updates:
                    return dict(result.state_updates)
                if result.metadata:
                    return dict(result.metadata)
        facet_state = state.facet_state.get(facet_name)
        if isinstance(facet_state, dict):
            return facet_state
        return None

    def _list_enabled_policies(self, limit: int) -> list[PolicyRule]:
        if hasattr(self.policy_store, "list_policies"):
            raw_policies = getattr(self.policy_store, "list_policies")(
                limit=limit,
                enabled_only=True,
            )
            if isinstance(raw_policies, list):
                return [
                    rule for rule in raw_policies if isinstance(rule, PolicyRule)
                ]
        if hasattr(self.policy_store, "list_enabled_policies"):
            raw_policies = getattr(self.policy_store, "list_enabled_policies")()
            if isinstance(raw_policies, list):
                return [
                    rule
                    for rule in raw_policies[:limit]
                    if isinstance(rule, PolicyRule)
                ]
        return []

    def _count_enabled_policies(self, fallback_policies: list[PolicyRule]) -> int:
        if hasattr(self.policy_store, "count_enabled_policies"):
            raw_count = getattr(self.policy_store, "count_enabled_policies")()
            if isinstance(raw_count, int) and raw_count >= 0:
                return raw_count
        if hasattr(self.policy_store, "list_enabled_policies"):
            raw_policies = getattr(self.policy_store, "list_enabled_policies")()
            if isinstance(raw_policies, list):
                return len(raw_policies)
        return len(fallback_policies)

    @staticmethod
    def _major_policy_constraints(policies: Sequence[PolicyRule]) -> list[str]:
        constraints: list[str] = []
        for target_type in EXTERNAL_POLICY_TARGETS:
            matching_rules = [
                rule
                for rule in policies
                if rule.target_type == target_type
                and rule.rule_type
                in {
                    PolicyRuleType.DENY,
                    PolicyRuleType.REQUIRE_APPROVAL,
                    PolicyRuleType.ALLOW,
                }
            ]
            if not matching_rules:
                continue
            top_rule = matching_rules[0]
            if top_rule.rule_type == PolicyRuleType.DENY:
                constraints.append(f"{target_type.value} denied by {top_rule.name}")
            elif top_rule.rule_type == PolicyRuleType.REQUIRE_APPROVAL:
                constraints.append(
                    f"{target_type.value} requires approval via {top_rule.name}"
                )
            else:
                constraints.append(f"{target_type.value} allowed by {top_rule.name}")
        return constraints

    @staticmethod
    def _source_types(items: Sequence[ContextItem]) -> list[str]:
        source_types: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = item.item_type.value
            if value in seen:
                continue
            seen.add(value)
            source_types.append(value)
        return source_types

    @staticmethod
    def _memory_to_context_item(
        record: MemoryRecord,
        *,
        context_source: str,
    ) -> ContextItem:
        item = StaticContextAssembler._memory_to_context_item(record)
        item.metadata["context_source"] = context_source
        return item

    @staticmethod
    def _coerce_string(raw_value: Any) -> str | None:
        if not isinstance(raw_value, str):
            return None
        cleaned = " ".join(raw_value.split())
        return cleaned or None

    @staticmethod
    def _numeric_value(
        raw_value: Any,
        *,
        allow_negative: bool = False,
    ) -> float:
        if not isinstance(raw_value, (int, float)):
            return 0.0
        value = float(raw_value)
        if allow_negative:
            return max(-1.0, min(value, 1.0))
        return max(0.0, min(value, 1.0))

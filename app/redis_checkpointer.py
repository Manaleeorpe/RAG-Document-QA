"""
Custom LangGraph checkpointer backed by basic Redis commands only.

Compatible with Upstash Redis — no Redis Search (FT.*) commands are used.

What RedisSaver (langgraph-checkpoint-redis) was doing internally:
  - Used redisvl to create a Redis Search index (FT.CREATE / FT._LIST)
  - Stored each checkpoint as a Redis Hash, indexed by that search index
  - Queried checkpoints via FT.SEARCH (vector + full-text capable)
  → This requires the Redis Stack / RediSearch module, which Upstash does NOT provide.

What this class does instead:
  - Uses only SET, GET, RPUSH, LRANGE, EXPIRE, PIPELINE — all available on Upstash
  - Serializes checkpoints to JSON (binary fields base64-encoded)
  - Stores them under predictable key names (see layout below)

Key layout (all keys share the same TTL, refreshed on each read/write):
  ckpt:{thread_id}:{ns}:{checkpoint_id}   → full checkpoint + metadata (SET)
  writes:{thread_id}:{ns}:{checkpoint_id} → pending writes for that checkpoint (SET)
  latest:{thread_id}:{ns}                 → most recent checkpoint_id for the thread (SET)
  idx:{thread_id}:{ns}                    → ordered list of checkpoint IDs oldest→newest (RPUSH)
"""

import base64
import json
from typing import Any, Iterator, Optional, Sequence, Tuple

import redis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)

CHECKPOINT_TTL = 86400  # 24 hours in seconds


class UpstashRedisSaver(BaseCheckpointSaver):
    def __init__(self, redis_client: redis.Redis, ttl: int = CHECKPOINT_TTL):
        super().__init__()
        self._r = redis_client
        self._ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int = CHECKPOINT_TTL) -> "UpstashRedisSaver":
        client = redis.Redis.from_url(url, decode_responses=True)
        return cls(client, ttl)

    # ── Key builders ─────────────────────────────────────────────────────────

    def _ckpt_key(self, tid: str, ns: str, cid: str) -> str:
        return f"ckpt:{tid}:{ns}:{cid}"

    def _writes_key(self, tid: str, ns: str, cid: str) -> str:
        return f"writes:{tid}:{ns}:{cid}"

    def _latest_key(self, tid: str, ns: str) -> str:
        return f"latest:{tid}:{ns}"

    def _idx_key(self, tid: str, ns: str) -> str:
        return f"idx:{tid}:{ns}"

    # ── Serialization ────────────────────────────────────────────────────────

    def _encode(self, value: Any) -> dict:
        """Serialize any LangGraph value to a JSON-safe dict via the built-in serde."""
        type_, data_bytes = self.serde.dumps_typed(value)
        return {"type": type_, "data": base64.b64encode(data_bytes).decode()}

    def _decode(self, stored: dict) -> Any:
        """Deserialize a value from a JSON-safe dict."""
        return self.serde.loads_typed((stored["type"], base64.b64decode(stored["data"])))

    # ── BaseCheckpointSaver interface ────────────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")

        cid = get_checkpoint_id(config)
        if not cid:
            cid = self._r.get(self._latest_key(tid, ns))
            if not cid:
                return None

        raw = self._r.get(self._ckpt_key(tid, ns, cid))
        if not raw:
            return None

        stored = json.loads(raw)
        checkpoint = self._decode(stored["checkpoint"])
        metadata = stored["metadata"]
        parent_cid = stored.get("parent_checkpoint_id")

        # Load pending writes if present
        pending_writes: list[tuple[str, str, Any]] = []
        raw_writes = self._r.get(self._writes_key(tid, ns, cid))
        if raw_writes:
            for w in json.loads(raw_writes):
                pending_writes.append((w["task_id"], w["channel"], self._decode(w["value"])))

        # Refresh TTL on read so active sessions don't expire mid-conversation
        pipe = self._r.pipeline()
        pipe.expire(self._ckpt_key(tid, ns, cid), self._ttl)
        pipe.expire(self._latest_key(tid, ns), self._ttl)
        pipe.expire(self._idx_key(tid, ns), self._ttl)
        pipe.execute()

        parent_config = (
            {"configurable": {"thread_id": tid, "checkpoint_ns": ns, "checkpoint_id": parent_cid}}
            if parent_cid else None
        )
        return CheckpointTuple(
            config={"configurable": {"thread_id": tid, "checkpoint_ns": ns, "checkpoint_id": cid}},
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if not config:
            return

        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        before_id = get_checkpoint_id(before) if before else None

        # Stored oldest→newest; reverse so we yield newest first
        all_ids = list(reversed(self._r.lrange(self._idx_key(tid, ns), 0, -1)))

        count = 0
        for cid in all_ids:
            if limit is not None and count >= limit:
                break
            if before_id and cid >= before_id:
                continue

            raw = self._r.get(self._ckpt_key(tid, ns, cid))
            if not raw:
                continue

            stored = json.loads(raw)
            metadata = stored["metadata"]
            if filter and not all(metadata.get(k) == v for k, v in filter.items()):
                continue

            checkpoint = self._decode(stored["checkpoint"])
            parent_cid = stored.get("parent_checkpoint_id")
            parent_config = (
                {"configurable": {"thread_id": tid, "checkpoint_ns": ns, "checkpoint_id": parent_cid}}
                if parent_cid else None
            )
            yield CheckpointTuple(
                config={"configurable": {"thread_id": tid, "checkpoint_ns": ns, "checkpoint_id": cid}},
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=[],
            )
            count += 1

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = checkpoint["id"]
        parent_cid = config["configurable"].get("checkpoint_id")

        payload = json.dumps({
            "checkpoint": self._encode(checkpoint),
            "metadata": metadata,
            "parent_checkpoint_id": parent_cid,
        })

        pipe = self._r.pipeline()
        pipe.set(self._ckpt_key(tid, ns, cid), payload, ex=self._ttl)
        pipe.set(self._latest_key(tid, ns), cid, ex=self._ttl)
        pipe.rpush(self._idx_key(tid, ns), cid)
        pipe.expire(self._idx_key(tid, ns), self._ttl)
        pipe.execute()

        return {"configurable": {"thread_id": tid, "checkpoint_ns": ns, "checkpoint_id": cid}}

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = config["configurable"]["checkpoint_id"]

        payload = json.dumps([
            {"task_id": task_id, "channel": channel, "value": self._encode(value)}
            for channel, value in writes
        ])
        self._r.set(self._writes_key(tid, ns, cid), payload, ex=self._ttl)

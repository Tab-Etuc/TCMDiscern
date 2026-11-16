"""Leakage-conscious Train-only partitioning for CDET experiments.

Connect patients and identical clinical inputs transitively, so neither a
repeated user_id nor duplicate case text can bridge Fit, Cal and Audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .datasets import normalized_input


@dataclass(frozen=True)
class TrainPartition:
    fit: tuple[int, ...]
    cal: tuple[int, ...]
    audit: tuple[int, ...]
    groups: tuple[int, ...]

    def verify(self, total: int) -> None:
        parts = (self.fit, self.cal, self.audit)
        assert sum(map(len, parts)) == total
        assert set(self.fit).isdisjoint(self.cal)
        assert set(self.fit).isdisjoint(self.audit)
        assert set(self.cal).isdisjoint(self.audit)
        assert set(self.fit) | set(self.cal) | set(self.audit) == set(range(total))
        group_parts = [{self.groups[i] for i in x} for x in parts]
        assert not (group_parts[0] & group_parts[1])
        assert not (group_parts[0] & group_parts[2])
        assert not (group_parts[1] & group_parts[2])


def grouped_partition(
    rows: list[dict],
    *,
    seed: int = 20261115,
    portions=(0.70, 0.15, 0.15),
) -> TrainPartition:
    if len(rows) < 20 or len(portions) != 3 or abs(sum(portions) - 1) > 1e-8:
        raise ValueError("Bad partition proportions or insufficient rows")
    n = len(rows)
    parent = list(range(n))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def merge(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    patient = {}
    observed_text = {}
    for idx, row in enumerate(rows):
        uid = str(row.get("user_id") or "").strip()
        if uid:
            if uid in patient:
                merge(idx, patient[uid])
            patient[uid] = idx
        clinical = normalized_input(row, full=True)
        if clinical:
            if clinical in observed_text:
                merge(idx, observed_text[clinical])
            observed_text[clinical] = idx

    groups = tuple(root(idx) for idx in range(n))
    ids = defaultdict(list)
    for idx, group in enumerate(groups):
        ids[group].append(idx)
    group_ids = list(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)
    assigned = {}
    sizes = np.zeros(3, dtype=int)
    targets = np.asarray(portions, dtype=float) * n

    # Keep at least one Train group of every observed class in Fit. These gold
    # labels belong to official Train, before Cal/Audit is treated as held out.
    support = Counter(str(r["norm_syndrome"]) for r in rows)
    representative = defaultdict(list)
    for gid, indices in ids.items():
        for label in {str(rows[i]["norm_syndrome"]) for i in indices}:
            representative[label].append(gid)
    for label, _ in sorted(support.items(), key=lambda item: item[1]):
        choices = representative[label]
        if not any(assigned.get(g) == 0 for g in choices):
            gid = min(choices, key=lambda group: len(ids[group]))
            if gid not in assigned:
                assigned[gid] = 0
                sizes[0] += len(ids[gid])
    for gid in group_ids:
        if gid in assigned:
            continue
        # Balance relative fill of all three partitions, preserving groups.
        partition = int(np.argmin(sizes / targets))
        assigned[gid] = partition
        sizes[partition] += len(ids[gid])
    partitions = [tuple(i for i in range(n) if assigned[groups[i]] == split) for split in range(3)]
    result = TrainPartition(*partitions, groups)
    result.verify(n)
    return result

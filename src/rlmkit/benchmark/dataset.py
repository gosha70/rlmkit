# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Benchmark dataset format and loader for strategy evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml


@dataclass
class BenchmarkCase:
    """A single benchmark case: content + query + optional expected answer."""

    id: str
    content: str
    query: str
    expected_answer: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_length(self) -> int:
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "content_length": self.content_length,
            "expected_answer": self.expected_answer,
            "category": self.category,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkDataset:
    """A collection of benchmark cases with metadata."""

    name: str
    description: str = ""
    cases: List[BenchmarkCase] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, index: int) -> BenchmarkCase:
        return self.cases[index]

    def filter_by_category(self, category: str) -> "BenchmarkDataset":
        """Return a new dataset with only cases from the given category."""
        filtered = [c for c in self.cases if c.category == category]
        return BenchmarkDataset(
            name=f"{self.name} [{category}]",
            description=self.description,
            cases=filtered,
            metadata=self.metadata,
        )

    def filter_by_difficulty(self, difficulty: str) -> "BenchmarkDataset":
        """Return a new dataset with only cases of the given difficulty."""
        filtered = [c for c in self.cases if c.difficulty == difficulty]
        return BenchmarkDataset(
            name=f"{self.name} [{difficulty}]",
            description=self.description,
            cases=filtered,
            metadata=self.metadata,
        )

    def filter_by_tag(self, tag: str) -> "BenchmarkDataset":
        """Return a new dataset with only cases containing the given tag."""
        filtered = [c for c in self.cases if tag in c.tags]
        return BenchmarkDataset(
            name=f"{self.name} [#{tag}]",
            description=self.description,
            cases=filtered,
            metadata=self.metadata,
        )

    @property
    def categories(self) -> List[str]:
        return sorted(set(c.category for c in self.cases))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "case_count": len(self.cases),
            "categories": self.categories,
            "cases": [c.to_dict() for c in self.cases],
            "metadata": self.metadata,
        }


def load_dataset(path: str) -> BenchmarkDataset:
    """Load a benchmark dataset from a YAML file.

    Expected YAML format::

        name: "My Benchmark"
        description: "Testing strategy performance"
        cases:
          - id: "case_1"
            content: "..."
            query: "..."
            expected_answer: "..."  # optional
            category: "factual"     # optional
            difficulty: "easy"      # optional
            tags: ["short"]         # optional
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(filepath) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Dataset file must contain a YAML mapping, got {type(data).__name__}")

    cases = []
    for i, case_data in enumerate(data.get("cases", [])):
        if not isinstance(case_data, dict):
            raise ValueError(f"Case {i} must be a mapping")
        if "content" not in case_data or "query" not in case_data:
            raise ValueError(f"Case {i} must have 'content' and 'query' fields")

        cases.append(
            BenchmarkCase(
                id=case_data.get("id", f"case_{i}"),
                content=case_data["content"],
                query=case_data["query"],
                expected_answer=case_data.get("expected_answer"),
                category=case_data.get("category", "general"),
                difficulty=case_data.get("difficulty", "medium"),
                tags=case_data.get("tags", []),
                metadata=case_data.get("metadata", {}),
            )
        )

    return BenchmarkDataset(
        name=data.get("name", filepath.stem),
        description=data.get("description", ""),
        cases=cases,
        metadata=data.get("metadata", {}),
    )


def load_dataset_from_dict(data: Dict[str, Any]) -> BenchmarkDataset:
    """Load a benchmark dataset from an in-memory dictionary (same schema as YAML)."""
    cases = []
    for i, case_data in enumerate(data.get("cases", [])):
        cases.append(
            BenchmarkCase(
                id=case_data.get("id", f"case_{i}"),
                content=case_data["content"],
                query=case_data["query"],
                expected_answer=case_data.get("expected_answer"),
                category=case_data.get("category", "general"),
                difficulty=case_data.get("difficulty", "medium"),
                tags=case_data.get("tags", []),
                metadata=case_data.get("metadata", {}),
            )
        )

    return BenchmarkDataset(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        cases=cases,
        metadata=data.get("metadata", {}),
    )

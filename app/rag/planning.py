"""Build one deterministic question plan for retrieval and answer generation."""

import re
from dataclasses import dataclass

MAX_QUESTION_PARTS = 8
_NUMBERED_ITEM = re.compile(r"(?<!\w)(\d{1,2})[.)]\s+")
_QUESTION_CLAUSE = re.compile(r"[^?]+\?")
_DEPENDENT_DIRECTIVE = re.compile(r"^(?:if|when)\s+(?:yes|no|applicable)\b", re.IGNORECASE)


def _clean_question(value: str) -> str:
    return " ".join(value.split()).strip()


@dataclass(frozen=True)
class QuestionPart:
    part_id: str
    question: str
    label: str | None = None  # Display only; retrieval and generation use question unchanged.


@dataclass(frozen=True)
class QuestionPlan:
    """The shared boundary between retrieval and structured generation."""

    original_question: str
    parts: tuple[QuestionPart, ...]

    @property
    def retrieval_queries(self) -> tuple[str, ...]:
        """Search exactly the question strings supplied to the answer model."""
        queries = [self.original_question, *(part.question for part in self.parts)]
        return tuple(dict.fromkeys(queries))


def _numbered_part_questions(question: str) -> list[tuple[str, str | None]]:
    matches = list(_NUMBERED_ITEM.finditer(question))
    if not matches:
        return []
    stem = question[: matches[0].start()].rstrip(" ,:;")
    if not stem:
        return []

    parts = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        item = question[match.end() : end].strip(" \t\r\n,;:.?")
        if item:
            parts.append((f"{stem} {item}?", item))
    return parts


def build_question_plan(question: str) -> QuestionPlan:
    """Create the original-question fallback and conservative answerable parts.

    Only explicit question clauses and numbered choices are split. Dependent directives such as
    ``If yes, describe`` remain folded into the original question, and arbitrary uses of ``and`` or
    ``or`` are never treated as decomposition boundaries.
    """
    original = question
    structural_question = _clean_question(original)
    candidates = _numbered_part_questions(structural_question)
    preserve_original_part = False
    if not candidates:
        clauses = [
            clause
            for match in _QUESTION_CLAUSE.finditer(structural_question)
            if (clause := _clean_question(match.group())) and not _DEPENDENT_DIRECTIVE.match(clause)
        ]
        candidates = (
            [(clause, None) for clause in clauses] if len(clauses) > 1 else [(original, None)]
        )
        preserve_original_part = len(clauses) <= 1

    parts = []
    seen = set()
    for candidate, label in candidates:
        normalized = original if preserve_original_part else _clean_question(candidate)
        key = normalized.casefold()
        if normalized and key not in seen:
            parts.append(
                QuestionPart(part_id=f"part_{len(parts) + 1}", question=normalized, label=label)
            )
            seen.add(key)
        if len(parts) == MAX_QUESTION_PARTS:
            break

    return QuestionPlan(original_question=original, parts=tuple(parts))

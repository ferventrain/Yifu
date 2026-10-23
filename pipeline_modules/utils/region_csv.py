"""Helpers for Allen region CSV columns (structure paths, names, acronyms)."""

from __future__ import annotations

import ast


def parse_structure_id_path(path_text) -> list[int]:
    if isinstance(path_text, (list, tuple)):
        return [int(value) for value in path_text]
    text = str(path_text).strip()
    if text.startswith("/"):
        return [int(part) for part in text.strip("/").split("/") if part]
    path_values = ast.literal_eval(text)
    return [int(value) for value in path_values]


def parse_acronym_text(acronym_text) -> str:
    """Use the last entry when the CSV stores an acronym list."""
    try:
        acronym_values = ast.literal_eval(str(acronym_text))
        if isinstance(acronym_values, list) and acronym_values:
            return str(acronym_values[-1]).strip()
    except Exception:
        pass
    return str(acronym_text).strip()


def split_name_and_acronym(name_text) -> tuple[str, str]:
    text = str(name_text).strip()
    if "," not in text:
        return text, ""
    base_name, acronym = text.rsplit(",", 1)
    return base_name.strip(), acronym.strip()

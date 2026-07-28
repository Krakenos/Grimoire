"""DB-backed overrides layered on top of the file/env settings.

grimoire/core/settings.py loads a `Settings` singleton once at import time, and nothing in the
codebase ever mutates it. That's fine for structural config (DB engine, encryption key, embedding
model, ...), but it means the management panel can't let operators tune the settings they change
often — summarization API, prompts, sampler params, tokenization — without restarting every process.

This module lets the panel persist per-section overrides in the `setting_override` table and
reconstructs an effective `Settings` object from file/env defaults + those overrides on demand.
Celery tasks call get_effective_settings() at the start of each run, so an edit saved in the panel
takes effect on the very next summarization with no restart required.
"""

import copy
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from grimoire.core.settings import Settings, loaded_settings

# Sections that may be overridden from the panel. Each corresponds to a top-level key in the
# settings YAML / Settings model that holds a nested dict.
EDITABLE_SECTIONS = ("summarization_api", "summarization", "tokenization")


def _override_rows(session: Session):
    from grimoire.db.models import SettingOverride

    query = select(SettingOverride).where(SettingOverride.section.in_(EDITABLE_SECTIONS))
    return session.scalars(query).all()


def _load_overrides(session: Session) -> dict[str, dict]:
    overrides = {}
    for row in _override_rows(session):
        try:
            value = json.loads(row.value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            overrides[row.section] = value
    return overrides


def get_effective_settings(session: Session | None = None) -> Settings:
    """Build a fresh Settings object = file/env defaults merged with any DB overrides.

    Re-runs the existing Pydantic validators (newline normalization, params stop/stop_sequence
    defaults, etc.) so DB-sourced values are treated identically to YAML-sourced ones.
    """
    if session is not None:
        overrides = _load_overrides(session)
    else:
        from grimoire.db.connection import SessionLocal

        with SessionLocal() as local_session:
            overrides = _load_overrides(local_session)

    base = copy.deepcopy(loaded_settings)
    for section, override in overrides.items():
        if isinstance(base.get(section), dict):
            base[section] = {**base[section], **override}
        else:
            base[section] = override

    return Settings(**base)


def get_overridden_sections(session: Session) -> set[str]:
    """Sections that currently have a DB override (drives the panel's "reset to default" UI)."""
    return {row.section for row in _override_rows(session)}


def upsert_override(session: Session, section: str, value: dict[str, Any]) -> None:
    from grimoire.db.models import SettingOverride

    if section not in EDITABLE_SECTIONS:
        raise ValueError(f"Unknown or non-editable settings section: {section}")

    row = session.scalar(select(SettingOverride).where(SettingOverride.section == section))
    serialized = json.dumps(value)
    if row is None:
        session.add(SettingOverride(section=section, value=serialized))
    else:
        row.value = serialized
    session.commit()


def delete_override(session: Session, section: str) -> None:
    from grimoire.db.models import SettingOverride

    row = session.scalar(select(SettingOverride).where(SettingOverride.section == section))
    if row is not None:
        session.delete(row)
        session.commit()

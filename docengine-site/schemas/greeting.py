"""Schema for the Hello World greeting table.

A schema is a Pydantic model that defines one DocEngine "table"; each YAML file
under ``data/greeting/`` is one row. Every schema must declare ``id`` and
``status`` (enforced by the engine's schema loader).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Greeting(BaseModel):
    """One greeting row rendered into the Hello World snippet."""

    name: str = Field(title="Name", description="Who/what is greeted.")
    message: str = Field(
        title="Message",
        description="The greeting text shown in the snippet.",
    )
    status: Literal["draft", "published"] = Field(
        default="draft",
        title="Status",
        description="Only 'published' rows are included when building from main.",
    )
    id: str = Field(
        title="ID",
        description="Unique identifier; used as the YAML filename.",
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )

    model_config = ConfigDict(from_attributes=True)

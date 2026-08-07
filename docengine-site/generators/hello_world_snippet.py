"""Generator for the Hello World snippet.

Reads published ``greeting`` rows and renders a single Mintlify snippet at
``snippets/docengine/hello-world.mdx`` (relative to the site's ``output_subpath``).
"""

from collections.abc import Iterator
from typing import Any

from app.core.generator_runner import PageGenerator
from app.core.template_engine import TemplateEngine


class HelloWorldSnippetGenerator(PageGenerator):
    """Render the Hello World greeting table as one Mintlify snippet."""

    name = "hello_world_snippet"
    description = "Generate the Hello World snippet"
    tables = ["greeting"]
    template = "hello_world_snippet.mdx.j2"

    generator_type = "single"
    priority = 10

    def generate(
        self, data: dict[str, Any], template_engine: TemplateEngine
    ) -> Iterator[tuple[str, str]]:
        yield self.render_output(
            template_engine,
            self.template,
            {"greetings": data.get("greeting") or []},
            "snippets/docengine/hello-world.mdx",
        )

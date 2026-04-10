"""
Unit tests for the Knowledgebase Nav generator.

These tests verify every public function in generate_tags.py using mocked
file systems and data.  No external services or real repository files are
needed. All inputs are created in temporary directories via pytest's
``tmp_path`` fixture.

Run with:
    pytest scripts/knowledgebase-nav/tests/test_generate_tags.py -v
"""

import json
import textwrap
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
# The script lives outside a normal Python package, so we add its parent
# directory to sys.path to make it importable.
# ---------------------------------------------------------------------------
import sys

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import generate_tags  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_config(tmp_path):
    """
    Create a minimal config.yaml in a temporary directory and return its path.

    The config defines one product ("widgets") with two allowed keywords
    so tests can verify config loading, validation, and keyword checking.
    """
    config_content = textwrap.dedent("""\
        products:
          - slug: widgets
            display_name: "W&B Widgets"
            allowed_keywords:
              - Alpha
              - Beta
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


@pytest.fixture
def sample_repo(tmp_path):
    """
    Create a minimal repo structure with articles, a docs.json, and return
    the repo root path.

    Structure:
        <tmp_path>/
            support/widgets/articles/
                article-one.mdx   (keywords: Alpha, Beta; featured: true)
                article-two.mdx   (keywords: Alpha)
                article-three.mdx (keywords: Beta)
            docs.json  (minimal with en language nav)
    """
    articles_dir = tmp_path / "support" / "widgets" / "articles"
    articles_dir.mkdir(parents=True)

    (articles_dir / "article-one.mdx").write_text(textwrap.dedent("""\
        ---
        title: "Article One"
        keywords: ["Alpha", "Beta"]
        featured: true
        ---

        This is the body of article one with enough text to test truncation of the preview.

        ---

        <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/alpha)</Badge>
    """), encoding="utf-8")

    (articles_dir / "article-two.mdx").write_text(textwrap.dedent("""\
        ---
        title: "Article Two"
        keywords: ["Alpha"]
        ---

        Short body.

        ---

        <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/alpha)</Badge>
    """), encoding="utf-8")

    (articles_dir / "article-three.mdx").write_text(textwrap.dedent("""\
        ---
        title: "Article Three"
        keywords: ["Beta"]
        ---

        Another article body for testing purposes.

        ---

        <Badge stroke shape="pill" color="orange" size="md">[Beta](/support/widgets/tags/beta)</Badge>
    """), encoding="utf-8")

    docs_json = {
        "navigation": {
            "languages": [
                {
                    "language": "en",
                    "tabs": [
                        {"tab": "Platform", "pages": ["index"]},
                    ],
                }
            ]
        }
    }
    (tmp_path / "docs.json").write_text(
        json.dumps(docs_json, indent=2) + "\n", encoding="utf-8"
    )

    # Create a support.mdx with product cards containing counts
    (tmp_path / "support.mdx").write_text(textwrap.dedent("""\
        ---
        title: Support
        ---

        ## Browse support articles by product

        <CardGroup cols={3}>
        <Card title="W&B Widgets" href="/support/widgets" arrow="true" icon="/icons/cropped-widgets.svg">
          {/* auto-generated counts */}
          0 articles &middot; 0 tags
          {/* end auto-generated counts */}
        </Card>
        </CardGroup>

        {/* ---- AUTO-GENERATED: featured articles ----
          This section is managed by scripts/knowledgebase-nav/generate_tags.py.
          To feature an article, add "featured: true" to its front matter.
          To remove it, set "featured: false" or remove the field.
          Do not edit the content between these markers by hand.
        ---- */}
        {/* ---- END AUTO-GENERATED: featured articles ---- */}
    """), encoding="utf-8")

    return tmp_path


@pytest.fixture
def template_env():
    """
    Create a Jinja2 template environment pointing to the real templates
    directory (relative to this test file).
    """
    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    return generate_tags.create_template_env(templates_dir)


# ===========================================================================
# Tests: load_config
# ===========================================================================

class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, sample_config):
        """
        A well-formed config.yaml should parse successfully and return
        a dict with a 'products' key containing the expected entries.
        """
        config = generate_tags.load_config(sample_config)
        assert "products" in config
        assert len(config["products"]) == 1
        assert config["products"][0]["slug"] == "widgets"
        assert config["products"][0]["display_name"] == "W&B Widgets"
        assert "Alpha" in config["products"][0]["allowed_keywords"]

    def test_load_config_file_not_found(self, tmp_path):
        """
        Attempting to load a non-existent config file should raise
        FileNotFoundError with a helpful message.
        """
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            generate_tags.load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_missing_products_key(self, tmp_path):
        """
        A config file without the 'products' key should raise ValueError
        because the generator cannot proceed without product definitions.
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text("other_key: value\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a 'products' key"):
            generate_tags.load_config(config_path)

    def test_load_config_missing_slug(self, tmp_path):
        """
        A product entry without 'slug' should raise ValueError so the
        user knows exactly which product definition is incomplete.
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text(textwrap.dedent("""\
            products:
              - display_name: "Test"
                allowed_keywords: []
        """), encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'slug'"):
            generate_tags.load_config(config_path)

    def test_load_config_missing_display_name(self, tmp_path):
        """
        A product entry without 'display_name' should raise ValueError.
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text(textwrap.dedent("""\
            products:
              - slug: test
                allowed_keywords: []
        """), encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'display_name'"):
            generate_tags.load_config(config_path)

    def test_load_config_missing_allowed_keywords(self, tmp_path):
        """
        A product without 'allowed_keywords' should raise ValueError.
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text(textwrap.dedent("""\
            products:
              - slug: test
                display_name: "Test"
        """), encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'allowed_keywords'"):
            generate_tags.load_config(config_path)


# ===========================================================================
# Tests: parse_frontmatter
# ===========================================================================

class TestParseFrontmatter:
    """Tests for the parse_frontmatter function."""

    def test_parse_valid_frontmatter(self, tmp_path):
        """
        A well-formed MDX file should return its front matter as a dict
        and the body text (without the Badge footer).
        """
        mdx = tmp_path / "test.mdx"
        mdx.write_text(textwrap.dedent("""\
            ---
            title: "Test Article"
            keywords: ["Alpha", "Beta"]
            ---

            Body content here.

            ---

            {/* ---- AUTO-GENERATED: tab badges ----
              Managed by scripts/knowledgebase-nav/generate_tags.py from keywords in front matter.
              Do not edit between these markers by hand.
            ---- */}
            <Badge>Tag</Badge>
            {/* ---- END AUTO-GENERATED: tab badges ---- */}
        """), encoding="utf-8")

        fm, body = generate_tags.parse_frontmatter(mdx)
        assert fm["title"] == "Test Article"
        assert fm["keywords"] == ["Alpha", "Beta"]
        assert "Body content here." in body
        assert "<Badge>" not in body

    def test_parse_frontmatter_missing_keywords(self, tmp_path):
        """
        An article without the 'keywords' field should still parse
        successfully.  The keywords will simply be missing from the
        dict (the caller defaults to an empty list).
        """
        mdx = tmp_path / "test.mdx"
        mdx.write_text(textwrap.dedent("""\
            ---
            title: "No Keywords"
            ---

            Body text.

            ---

            <Badge>Tag</Badge>
        """), encoding="utf-8")

        fm, body = generate_tags.parse_frontmatter(mdx)
        assert fm["title"] == "No Keywords"
        assert "keywords" not in fm
        assert "Body text." in body

    def test_parse_frontmatter_with_featured(self, tmp_path):
        """
        The 'featured' field should be parsed as a boolean from the
        YAML front matter.
        """
        mdx = tmp_path / "test.mdx"
        mdx.write_text(textwrap.dedent("""\
            ---
            title: "Featured Article"
            keywords: ["Alpha"]
            featured: true
            ---

            Body text.
        """), encoding="utf-8")

        fm, body = generate_tags.parse_frontmatter(mdx)
        assert fm["featured"] is True

    def test_parse_frontmatter_no_opening_delimiter(self, tmp_path):
        """
        A file that doesn't start with '---' should raise ValueError
        because it has no valid YAML front matter.
        """
        mdx = tmp_path / "test.mdx"
        mdx.write_text("No front matter here.\n", encoding="utf-8")

        with pytest.raises(ValueError, match="does not start with '---'"):
            generate_tags.parse_frontmatter(mdx)

    def test_parse_frontmatter_body_without_badge_footer(self, tmp_path):
        """
        An article without the trailing --- and Badge section should
        return the entire body text after front matter.
        """
        mdx = tmp_path / "test.mdx"
        mdx.write_text(textwrap.dedent("""\
            ---
            title: "Simple"
            ---

            Just a body, no badges.
        """), encoding="utf-8")

        fm, body = generate_tags.parse_frontmatter(mdx)
        assert body == "Just a body, no badges."

    def test_parse_frontmatter_hr_in_body_preserved(self, tmp_path):
        """A --- horizontal rule inside the body is not treated as a footer boundary."""
        mdx = tmp_path / "test.mdx"
        mdx.write_text(textwrap.dedent("""\
            ---
            title: "HR"
            keywords: ["Alpha"]
            ---

            Part one.

            ---

            Part two.

            {/* ---- AUTO-GENERATED: tab badges ----
              Managed by scripts/knowledgebase-nav/generate_tags.py from keywords in front matter.
              Do not edit between these markers by hand.
            ---- */}
            <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/alpha)</Badge>
            {/* ---- END AUTO-GENERATED: tab badges ---- */}
        """), encoding="utf-8")

        _, body = generate_tags.parse_frontmatter(mdx)
        assert "Part one." in body
        assert "Part two." in body
        assert "---" in body
        assert "AUTO-GENERATED" not in body

    def test_parse_frontmatter_marker_text_excluded_from_body(self, tmp_path):
        """Marker comments and Badge elements are excluded from the body."""
        mdx = tmp_path / "test.mdx"
        mdx.write_text(textwrap.dedent("""\
            ---
            title: "Short"
            keywords: ["Alpha"]
            ---

            One sentence.

            ---

            {/* ---- AUTO-GENERATED: tab badges ----
              Managed by scripts/knowledgebase-nav/generate_tags.py from keywords in front matter.
              Do not edit between these markers by hand.
            ---- */}
            <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/alpha)</Badge>
            {/* ---- END AUTO-GENERATED: tab badges ---- */}
        """), encoding="utf-8")

        _, body = generate_tags.parse_frontmatter(mdx)
        assert body == "One sentence."
        assert "Badge" not in body


# ===========================================================================
# Tests: plain_text and extract_body_preview
# ===========================================================================

class TestBodyPreview:
    """Tests for the plain_text and extract_body_preview functions."""

    def test_plain_text_strips_backticks(self):
        """
        Backtick-wrapped code should have backticks removed.  The code
        content itself is preserved.
        """
        result = generate_tags.plain_text("Use `wandb.init()` to start.")
        assert result == "Use wandb.init() to start."

    def test_plain_text_strips_mdx_components(self):
        """
        MDX or JSX tags should be removed; inner text between tags is kept.
        """
        result = generate_tags.plain_text("Text <Component>inside</Component> more.")
        assert result == "Text inside more."
        assert "<" not in result
        assert ">" not in result

    def test_plain_text_strips_markdown_syntax(self):
        """
        Markdown bold/italic markers (**text** or *text*) should be
        stripped, leaving the text content.
        """
        result = generate_tags.plain_text("This is **bold** and *italic*.")
        assert result == "This is bold and italic."

    def test_plain_text_collapses_whitespace(self):
        """
        Multiple spaces, tabs, and newlines should be collapsed to
        single spaces.
        """
        result = generate_tags.plain_text("word1   word2\n\nword3")
        assert result == "word1 word2 word3"

    def test_plain_text_strips_markdown_links_keeps_label(self):
        """Inline link URL and brackets must not appear in the preview."""
        result = generate_tags.plain_text(
            "Read [the guide](https://docs.wandb.ai/foo) for details."
        )
        assert result == "Read the guide for details."
        assert "http" not in result
        assert "[" not in result

    def test_plain_text_strips_bare_urls(self):
        """Raw https URLs should not pass through to the card."""
        result = generate_tags.plain_text("Visit https://wandb.ai today.")
        assert result == "Visit today."
        assert "wandb" not in result.lower()

    def test_plain_text_strips_mdx_brace_expressions(self):
        """Simple MDX {expression} segments should be removed."""
        result = generate_tags.plain_text("Value is {props.count} items.")
        assert result == "Value is items."
        assert "{" not in result

    def test_plain_text_decodes_html_entities(self):
        """Named entities become characters, then pass allowlist."""
        result = generate_tags.plain_text("Tom &amp; Jerry")
        assert "Tom" in result and "Jerry" in result
        assert "&amp;" not in result

    def test_plain_text_typographic_apostrophe_to_ascii(self):
        """Unicode apostrophe must not become a word break in the preview."""
        result = generate_tags.plain_text("Python\u2019s functions are great.")
        assert result == "Python's functions are great."

    def test_plain_text_typographic_double_quotes_to_ascii(self):
        result = generate_tags.plain_text("\u201cHello\u201d there.")
        assert result == '"Hello" there.'

    def test_plain_text_preserves_underscore_and_equals(self):
        """Identifiers and simple assignments stay readable in previews."""
        result = generate_tags.plain_text("Call my_awesome_fn when x=1.")
        assert "my_awesome_fn" in result
        assert "x=1" in result

    def test_plain_text_underscore_emphasis_when_delimited(self):
        """_italic_ is still stripped when underscores are not inside a word."""
        result = generate_tags.plain_text("Use _emphasis_ here.")
        assert result == "Use emphasis here."

    def test_extract_body_preview_short_text(self):
        """
        Text shorter than the max length should be returned as-is,
        without truncation or suffix.
        """
        result = generate_tags.extract_body_preview("Short text.")
        assert result == "Short text."

    def test_extract_body_preview_truncation(self):
        """
        Text longer than 120 characters should be truncated to exactly
        120 characters followed by ' ...' (space + three dots).
        """
        long_text = "A" * 200
        result = generate_tags.extract_body_preview(long_text)
        assert len(result) == 120 + 4  # 120 chars + " ..."
        assert result.endswith(" ...")
        assert result[:120] == "A" * 120

    def test_extract_body_preview_exactly_120_chars(self):
        """
        Text of exactly 120 characters should not be truncated.
        """
        text = "B" * 120
        result = generate_tags.extract_body_preview(text)
        assert result == text

    def test_extract_body_preview_custom_max_len(self):
        """
        The max_len parameter should control the truncation point.
        """
        result = generate_tags.extract_body_preview("Hello World!", max_len=5)
        assert result == "Hello ..."


# ===========================================================================
# Tests: _card_text_from_frontmatter_field
# ===========================================================================

class TestCardTextFromFrontmatterField:
    """Tests for the _card_text_from_frontmatter_field helper."""

    def test_returns_none_when_key_missing(self):
        """A missing key should return None so the resolver falls through."""
        assert generate_tags._card_text_from_frontmatter_field({}, "description") is None

    def test_returns_none_when_value_is_none(self):
        """An explicit None value should return None."""
        assert generate_tags._card_text_from_frontmatter_field(
            {"description": None}, "description"
        ) is None

    def test_returns_none_for_non_string_int(self):
        """An integer value should return None (no coercion)."""
        assert generate_tags._card_text_from_frontmatter_field(
            {"description": 42}, "description"
        ) is None

    def test_returns_none_for_non_string_list(self):
        """A list value should return None (no coercion)."""
        assert generate_tags._card_text_from_frontmatter_field(
            {"description": ["a", "b"]}, "description"
        ) is None

    def test_returns_none_for_non_string_bool(self):
        """A boolean value should return None (no coercion)."""
        assert generate_tags._card_text_from_frontmatter_field(
            {"description": True}, "description"
        ) is None

    def test_strips_outer_double_quotes(self):
        """Wrapping double quotes should be removed."""
        result = generate_tags._card_text_from_frontmatter_field(
            {"description": '"Hello world."'}, "description"
        )
        assert result == "Hello world."

    def test_strips_outer_single_quotes(self):
        """Wrapping single quotes should be removed."""
        result = generate_tags._card_text_from_frontmatter_field(
            {"description": "'Hello world.'"}, "description"
        )
        assert result == "Hello world."

    def test_does_not_strip_mismatched_quotes(self):
        """Mismatched outer quotes should not be stripped."""
        result = generate_tags._card_text_from_frontmatter_field(
            {"description": "\"Hello world.'"}, "description"
        )
        assert result == "\"Hello world.'"

    def test_does_not_strip_inner_quotes(self):
        """Quotes inside the string should be preserved."""
        result = generate_tags._card_text_from_frontmatter_field(
            {"description": 'She said "hello" today.'}, "description"
        )
        assert result == 'She said "hello" today.'

    def test_returns_none_when_empty_after_quote_strip(self):
        """A value that is just a pair of quotes should return None."""
        assert generate_tags._card_text_from_frontmatter_field(
            {"description": '""'}, "description"
        ) is None
        assert generate_tags._card_text_from_frontmatter_field(
            {"description": "''"}, "description"
        ) is None

    def test_returns_none_for_empty_string(self):
        """An empty string should return None."""
        assert generate_tags._card_text_from_frontmatter_field(
            {"description": ""}, "description"
        ) is None

    def test_collapses_newlines_to_single_space(self):
        """Internal newlines should be collapsed to a single space."""
        result = generate_tags._card_text_from_frontmatter_field(
            {"description": "Line one.\nLine two.\nLine three."}, "description"
        )
        assert result == "Line one. Line two. Line three."

    def test_collapses_newlines_with_surrounding_whitespace(self):
        """Whitespace around newlines should collapse too."""
        result = generate_tags._card_text_from_frontmatter_field(
            {"description": "Hello  \n  world."}, "description"
        )
        assert result == "Hello world."

    def test_preserves_single_line_string(self):
        """A normal single-line string passes through unchanged."""
        result = generate_tags._card_text_from_frontmatter_field(
            {"description": "Simple description."}, "description"
        )
        assert result == "Simple description."

    def test_short_string_no_strip(self):
        """A single character should not be processed as quotes."""
        result = generate_tags._card_text_from_frontmatter_field(
            {"description": "X"}, "description"
        )
        assert result == "X"


# ===========================================================================
# Tests: resolve_body_preview
# ===========================================================================

class TestResolveBodyPreview:
    """Tests for the resolve_body_preview function."""

    def test_docengine_description_takes_priority(self):
        """docengineDescription should win over description and body."""
        fm = {
            "docengineDescription": "From docengine.",
            "description": "From description.",
        }
        result = generate_tags.resolve_body_preview(fm, "Body text here.")
        assert result == "From docengine."

    def test_description_used_when_no_docengine(self):
        """description should be used when docengineDescription is absent."""
        fm = {"description": "From description."}
        result = generate_tags.resolve_body_preview(fm, "Body text here.")
        assert result == "From description."

    def test_body_fallback_when_neither_field(self):
        """Falls back to extract_body_preview when no override fields exist."""
        result = generate_tags.resolve_body_preview({}, "Short body.")
        assert result == "Short body."

    def test_body_fallback_when_both_fields_empty(self):
        """Empty strings in both fields should fall through to body."""
        fm = {"docengineDescription": "", "description": ""}
        result = generate_tags.resolve_body_preview(fm, "Fallback body.")
        assert result == "Fallback body."

    def test_description_used_when_docengine_is_none(self):
        """Explicit None for docengineDescription falls through to description."""
        fm = {"docengineDescription": None, "description": "SEO text."}
        result = generate_tags.resolve_body_preview(fm, "Body.")
        assert result == "SEO text."

    def test_description_used_when_docengine_is_non_string(self):
        """Non-string docengineDescription falls through to description."""
        fm = {"docengineDescription": 123, "description": "Fallback desc."}
        result = generate_tags.resolve_body_preview(fm, "Body.")
        assert result == "Fallback desc."

    def test_body_used_when_description_is_non_string(self):
        """Non-string description falls through to body."""
        fm = {"description": ["not", "a", "string"]}
        result = generate_tags.resolve_body_preview(fm, "Body text.")
        assert result == "Body text."

    def test_body_preview_truncation_preserved(self):
        """Body fallback still truncates at 120 characters."""
        long_body = "A" * 200
        result = generate_tags.resolve_body_preview({}, long_body)
        assert result == "A" * 120 + " ..."

    def test_docengine_not_truncated(self):
        """Frontmatter overrides should not be truncated."""
        long_text = "B" * 200
        fm = {"docengineDescription": long_text}
        result = generate_tags.resolve_body_preview(fm, "Body.")
        assert result == long_text

    def test_description_not_truncated(self):
        """description override should not be truncated."""
        long_text = "C" * 200
        fm = {"description": long_text}
        result = generate_tags.resolve_body_preview(fm, "Body.")
        assert result == long_text

    def test_docengine_outer_quotes_stripped(self):
        """Outer quotes on docengineDescription are stripped."""
        fm = {"docengineDescription": '"Quoted text."'}
        result = generate_tags.resolve_body_preview(fm, "Body.")
        assert result == "Quoted text."

    def test_description_outer_quotes_stripped(self):
        """Outer quotes on description are stripped."""
        fm = {"description": "'Quoted text.'"}
        result = generate_tags.resolve_body_preview(fm, "Body.")
        assert result == "Quoted text."

    def test_docengine_newlines_collapsed(self):
        """Newlines in docengineDescription are collapsed."""
        fm = {"docengineDescription": "Line one.\nLine two."}
        result = generate_tags.resolve_body_preview(fm, "Body.")
        assert result == "Line one. Line two."

    def test_docengine_only_quotes_falls_through(self):
        """docengineDescription that is just quotes falls through to description."""
        fm = {"docengineDescription": '""', "description": "Actual text."}
        result = generate_tags.resolve_body_preview(fm, "Body.")
        assert result == "Actual text."


# ===========================================================================
# Tests: tag_slug
# ===========================================================================

class TestTagSlug:
    """Tests for the tag_slug function."""

    def test_simple_slug(self):
        """Single lowercase word stays as-is."""
        assert generate_tags.tag_slug("security") == "security"

    def test_multi_word_slug(self):
        """Spaces become hyphens."""
        assert generate_tags.tag_slug("Environment Variables") == "environment-variables"

    def test_ampersand_slug(self):
        """Ampersands and surrounding spaces become a single hyphen."""
        assert generate_tags.tag_slug("Authentication & Access") == "authentication-access"

    def test_uppercase_slug(self):
        """All uppercase input becomes lowercase."""
        assert generate_tags.tag_slug("AWS") == "aws"

    def test_mixed_special_chars(self):
        """Multiple special characters collapse to one hyphen."""
        assert generate_tags.tag_slug("Quotas & Rate Limits") == "quotas-rate-limits"

    def test_no_leading_trailing_hyphens(self):
        """Leading/trailing special characters don't produce edge hyphens."""
        assert generate_tags.tag_slug("--test--") == "test"

    def test_run_crashes(self):
        """Multi-word with space produces expected slug."""
        assert generate_tags.tag_slug("Run Crashes") == "run-crashes"


# ===========================================================================
# Tests: crawl_articles
# ===========================================================================

class TestCrawlArticles:
    """Tests for the crawl_articles function."""

    def test_crawl_finds_articles(self, sample_repo):
        """
        crawl_articles should find all MDX files in the articles directory
        and return structured dicts with the expected keys.
        """
        articles = generate_tags.crawl_articles(sample_repo, "widgets")
        assert len(articles) == 3

        titles = {a["title"] for a in articles}
        assert titles == {"Article One", "Article Two", "Article Three"}

    def test_crawl_article_structure(self, sample_repo):
        """
        Each article dict should contain all the expected keys with
        correct types.
        """
        articles = generate_tags.crawl_articles(sample_repo, "widgets")
        article = next(a for a in articles if a["title"] == "Article One")

        assert article["keywords"] == ["Alpha", "Beta"]
        assert article["featured"] is True
        assert article["page_path"] == "support/widgets/articles/article-one"
        assert article["mdx_path"] == "support/widgets/articles/article-one.mdx"
        assert article["file_stem"] == "article-one"
        assert isinstance(article["body_preview"], str)
        assert isinstance(article["tag_links"], list)
        assert article["title_attr"] == "Article One"

    def test_crawl_featured_field_defaults_to_false(self, sample_repo):
        """
        Articles without a 'featured' field in their front matter should
        default to featured=False.
        """
        articles = generate_tags.crawl_articles(sample_repo, "widgets")
        article_two = next(a for a in articles if a["title"] == "Article Two")
        assert article_two["featured"] is False

    def test_crawl_missing_directory(self, tmp_path):
        """
        If the articles directory doesn't exist, crawl_articles should
        return an empty list and emit a warning.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            articles = generate_tags.crawl_articles(tmp_path, "nonexistent")
            assert articles == []
            assert len(w) == 1
            assert "not found" in str(w[0].message)

    def test_crawl_articles_sorted_by_filename(self, sample_repo):
        """
        Articles should be returned in alphabetical order by filename
        (since glob results are sorted).
        """
        articles = generate_tags.crawl_articles(sample_repo, "widgets")
        file_stems = [a["file_stem"] for a in articles]
        assert file_stems == sorted(file_stems)

    def test_crawl_title_with_quotes(self, tmp_path):
        """
        Titles containing double quotes should have them escaped as
        &quot; in the title_attr field for safe embedding in HTML.
        """
        articles_dir = tmp_path / "support" / "test" / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "test.mdx").write_text(textwrap.dedent("""\
            ---
            title: 'She said "hello"'
            keywords: ["Alpha"]
            ---

            Body text.
        """), encoding="utf-8")

        articles = generate_tags.crawl_articles(tmp_path, "test")
        assert articles[0]["title_attr"] == 'She said &quot;hello&quot;'

    def test_crawl_string_keywords_coerced_to_single_tag(self, tmp_path):
        """
        keywords: \"TagName\" (YAML string) must not iterate per character;
        it should become one tag after a warning.
        """
        articles_dir = tmp_path / "support" / "widgets" / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "one.mdx").write_text(
            textwrap.dedent("""\
                ---
                title: "S"
                keywords: "Alpha"
                ---

                Hi.
            """),
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            articles = generate_tags.crawl_articles(tmp_path, "widgets")
        assert articles[0]["keywords"] == ["Alpha"]
        assert [x["name"] for x in articles[0]["tag_links"]] == ["Alpha"]
        assert any("single tag" in str(x.message).lower() for x in w)

    def test_crawl_non_list_non_string_keywords_empty(self, tmp_path):
        """keywords: 42 (or other scalar) should warn and yield no tags."""
        articles_dir = tmp_path / "support" / "widgets" / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "bad.mdx").write_text(
            textwrap.dedent("""\
                ---
                title: "N"
                keywords: 42
                ---

                Hi.
            """),
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            articles = generate_tags.crawl_articles(tmp_path, "widgets")
        assert articles[0]["keywords"] == []
        assert articles[0]["tag_links"] == []
        assert any("int" in str(x.message).lower() for x in w)

    def test_crawl_body_preview_from_docengine_description(self, tmp_path):
        """docengineDescription in front matter overrides the body preview."""
        articles_dir = tmp_path / "support" / "widgets" / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "test.mdx").write_text(textwrap.dedent("""\
            ---
            title: "Test"
            keywords: ["Alpha"]
            docengineDescription: "Custom card text from docengine."
            ---

            This body text should not appear in the preview.
        """), encoding="utf-8")

        articles = generate_tags.crawl_articles(tmp_path, "widgets")
        assert articles[0]["body_preview"] == "Custom card text from docengine."

    def test_crawl_body_preview_from_description_fallback(self, tmp_path):
        """description is used for body_preview when docengineDescription is absent."""
        articles_dir = tmp_path / "support" / "widgets" / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "test.mdx").write_text(textwrap.dedent("""\
            ---
            title: "Test"
            keywords: ["Alpha"]
            description: "SEO and card text."
            ---

            This body text should not appear in the preview.
        """), encoding="utf-8")

        articles = generate_tags.crawl_articles(tmp_path, "widgets")
        assert articles[0]["body_preview"] == "SEO and card text."

    def test_crawl_body_preview_body_fallback(self, tmp_path):
        """Without description fields, body_preview uses the body snippet."""
        articles_dir = tmp_path / "support" / "widgets" / "articles"
        articles_dir.mkdir(parents=True)
        (articles_dir / "test.mdx").write_text(textwrap.dedent("""\
            ---
            title: "Test"
            keywords: ["Alpha"]
            ---

            Short body text.
        """), encoding="utf-8")

        articles = generate_tags.crawl_articles(tmp_path, "widgets")
        assert articles[0]["body_preview"] == "Short body text."


# ===========================================================================
# Tests: get_featured_articles
# ===========================================================================

class TestGetFeaturedArticles:
    """Tests for the get_featured_articles function."""

    def test_filters_featured(self):
        """Only articles with featured=True should be returned."""
        articles = [
            {"title": "B", "featured": True},
            {"title": "A", "featured": True},
            {"title": "C", "featured": False},
        ]
        result = generate_tags.get_featured_articles(articles)
        assert len(result) == 2
        assert all(a["featured"] for a in result)

    def test_sorted_by_title(self):
        """Returned featured articles should be sorted alphabetically."""
        articles = [
            {"title": "Zebra", "featured": True},
            {"title": "Apple", "featured": True},
        ]
        result = generate_tags.get_featured_articles(articles)
        assert [a["title"] for a in result] == ["Apple", "Zebra"]

    def test_no_featured_articles(self):
        """If no articles are featured, an empty list is returned."""
        articles = [{"title": "A", "featured": False}]
        assert generate_tags.get_featured_articles(articles) == []

    def test_empty_input(self):
        """An empty article list returns an empty result."""
        assert generate_tags.get_featured_articles([]) == []


# ===========================================================================
# Tests: build_tag_index
# ===========================================================================

class TestBuildTagIndex:
    """Tests for the build_tag_index function."""

    def test_builds_correct_mapping(self):
        """
        Articles should be grouped by their keywords.  An article with
        two keywords appears in both tag lists.
        """
        articles = [
            {"title": "A", "keywords": ["Alpha", "Beta"]},
            {"title": "B", "keywords": ["Alpha"]},
        ]
        index = generate_tags.build_tag_index(articles, ["Alpha", "Beta"])

        assert len(index["Alpha"]) == 2
        assert len(index["Beta"]) == 1

    def test_articles_sorted_by_title_within_tag(self):
        """Articles within each tag should be sorted alphabetically."""
        articles = [
            {"title": "Zebra", "keywords": ["Tag"]},
            {"title": "Apple", "keywords": ["Tag"]},
        ]
        index = generate_tags.build_tag_index(articles, ["Tag"])
        titles = [a["title"] for a in index["Tag"]]
        assert titles == ["Apple", "Zebra"]

    def test_unknown_keyword_warning(self):
        """
        A keyword not in the allowed list should emit a warning but
        still appear in the index.  This prevents silent data loss.
        """
        articles = [
            {
                "title": "A",
                "keywords": ["Unknown"],
                "mdx_path": "support/widgets/articles/example.mdx",
            },
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            index = generate_tags.build_tag_index(articles, ["Alpha"])

            assert "Unknown" in index
            assert len(w) == 1
            assert "Unknown keyword `Unknown`" in str(w[0].message)
            assert "support/widgets/articles/example.mdx" in str(w[0].message)
            assert "scripts/knowledgebase-nav/config.yaml" in str(w[0].message)

    def test_unknown_keyword_warned_only_once(self):
        """
        If multiple articles use the same unknown keyword, the warning
        should only be emitted once (not once per article).
        """
        articles = [
            {"title": "A", "keywords": ["Mystery"]},
            {"title": "B", "keywords": ["Mystery"]},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_tags.build_tag_index(articles, [])
            mystery_warnings = [x for x in w if "Mystery" in str(x.message)]
            assert len(mystery_warnings) == 1

    def test_empty_keywords(self):
        """An article with no keywords should not appear in any tag."""
        articles = [{"title": "A", "keywords": []}]
        index = generate_tags.build_tag_index(articles, [])
        assert len(index) == 0


# ===========================================================================
# Tests: render_tag_pages
# ===========================================================================

class TestRenderTagPages:
    """Tests for the render_tag_pages function."""

    def test_creates_tag_files(self, tmp_path, template_env):
        """
        render_tag_pages should create an MDX file for each tag in the
        tag index, in the correct directory.
        """
        tags_dir = tmp_path / "support" / "widgets" / "tags"

        tag_index = {
            "Alpha": [
                {"title_attr": "Article One", "body_preview": "Preview one.", "page_path": "support/widgets/articles/article-one"},
            ],
            "Beta": [
                {"title_attr": "Article Two", "body_preview": "Preview two.", "page_path": "support/widgets/articles/article-two"},
            ],
        }

        paths = generate_tags.render_tag_pages(tmp_path, "widgets", tag_index, template_env)

        assert (tags_dir / "alpha.mdx").exists()
        assert (tags_dir / "beta.mdx").exists()
        assert sorted(paths) == ["support/widgets/tags/alpha", "support/widgets/tags/beta"]

    def test_tag_page_content(self, tmp_path, template_env):
        """
        A generated tag page should contain the correct front matter
        (title and article count) and Card components for each article.
        """
        tag_index = {
            "Alpha": [
                {"title_attr": "Article One", "body_preview": "Preview.", "page_path": "support/widgets/articles/article-one"},
                {"title_attr": "Article Two", "body_preview": "Another.", "page_path": "support/widgets/articles/article-two"},
            ],
        }

        generate_tags.render_tag_pages(tmp_path, "widgets", tag_index, template_env)

        content = (tmp_path / "support" / "widgets" / "tags" / "alpha.mdx").read_text()
        assert 'title: "Alpha"' in content
        assert 'tag: "2"' in content
        assert '<Card title="Article One"' in content
        assert '<Card title="Article Two"' in content
        assert 'href="/support/widgets/articles/article-one"' in content

    def test_tag_page_article_count_in_frontmatter(self, tmp_path, template_env):
        """
        The ``tag`` field in front matter should be a JSON string of the
        article count (for example ``tag: "3"`` when three articles).
        """
        tag_index = {
            "Test": [
                {"title_attr": f"Art {i}", "body_preview": "P.", "page_path": f"p/{i}"}
                for i in range(5)
            ],
        }

        generate_tags.render_tag_pages(tmp_path, "widgets", tag_index, template_env)
        content = (tmp_path / "support" / "widgets" / "tags" / "test.mdx").read_text()
        assert 'tag: "5"' in content


# ===========================================================================
# Tests: render_product_index
# ===========================================================================

class TestRenderProductIndex:
    """Tests for the render_product_index function."""

    def test_browse_by_category(self, tmp_path, template_env):
        """
        The product index should contain a 'Browse by category' section
        with Cards for each tag showing article counts.
        """
        (tmp_path / "support").mkdir(parents=True, exist_ok=True)
        tag_index = {
            "Alpha": [{"title": "A"}],
            "Beta": [{"title": "B"}, {"title": "C"}],
        }

        generate_tags.render_product_index(
            tmp_path, "widgets", "W&B Widgets", tag_index, [], template_env
        )

        content = (tmp_path / "support" / "widgets.mdx").read_text()
        assert "## Browse by category" in content
        assert '<Card title="Alpha"' in content
        assert "1 article\n" in content
        assert '<Card title="Beta"' in content
        assert "2 articles\n" in content

    def test_product_index_with_featured(self, tmp_path, template_env):
        """
        When featured articles are provided, the product index should
        include a 'Featured articles' section with horizontal Cards,
        body previews, and Badge tag links.
        """
        (tmp_path / "support").mkdir(parents=True, exist_ok=True)
        tag_index = {"Alpha": [{"title": "A"}]}
        featured = [
            {
                "title": "Featured Art",
                "title_attr": "Featured Art",
                "body_preview": "A preview.",
                "page_path": "support/widgets/articles/featured-art",
                "tag_links": [{"name": "Alpha", "href": "/support/widgets/tags/alpha"}],
            }
        ]

        generate_tags.render_product_index(
            tmp_path, "widgets", "W&B Widgets", tag_index, featured, template_env
        )

        content = (tmp_path / "support" / "widgets.mdx").read_text()
        assert "## Featured articles" in content
        assert '<Card title="Featured Art"' in content
        assert "horizontal" in content
        assert "A preview." in content
        assert "<Badge" in content
        assert "[Alpha]" in content

    def test_product_index_no_featured(self, tmp_path, template_env):
        """
        When no articles are featured, the 'Featured articles' section
        should be omitted entirely.
        """
        (tmp_path / "support").mkdir(parents=True, exist_ok=True)
        tag_index = {"Alpha": [{"title": "A"}]}

        generate_tags.render_product_index(
            tmp_path, "widgets", "W&B Widgets", tag_index, [], template_env
        )

        content = (tmp_path / "support" / "widgets.mdx").read_text()
        assert "Featured articles" not in content
        assert "## Browse by category" in content

    def test_product_index_title(self, tmp_path, template_env):
        """
        The product index front matter title should be
        'Support: <display_name>'.
        """
        (tmp_path / "support").mkdir(parents=True, exist_ok=True)

        generate_tags.render_product_index(
            tmp_path, "widgets", "W&B Widgets", {}, [], template_env
        )

        content = (tmp_path / "support" / "widgets.mdx").read_text()
        assert 'title: "Support: W&B Widgets"' in content

    def test_product_index_tags_sorted_alphabetically(self, tmp_path, template_env):
        """
        Tags in the Browse by category section should be sorted
        alphabetically by name.
        """
        (tmp_path / "support").mkdir(parents=True, exist_ok=True)
        tag_index = {
            "Zebra": [{"title": "Z"}],
            "Apple": [{"title": "A"}],
            "Mango": [{"title": "M"}],
        }

        generate_tags.render_product_index(
            tmp_path, "widgets", "W&B Widgets", tag_index, [], template_env
        )

        content = (tmp_path / "support" / "widgets.mdx").read_text()
        apple_pos = content.index("Apple")
        mango_pos = content.index("Mango")
        zebra_pos = content.index("Zebra")
        assert apple_pos < mango_pos < zebra_pos


# ===========================================================================
# Tests: update_docs_json
# ===========================================================================

class TestUpdateDocsJson:
    """Tests for the update_docs_json function."""

    def test_creates_hidden_tabs(self, sample_repo):
        """
        update_docs_json should create hidden support tabs for each
        product with the correct page listings.
        """
        products = [
            {"slug": "widgets", "display_name": "W&B Widgets"},
        ]
        tag_paths = {
            "widgets": ["support/widgets/tags/alpha", "support/widgets/tags/beta"],
        }

        generate_tags.update_docs_json(sample_repo, products, tag_paths)

        docs = json.loads((sample_repo / "docs.json").read_text())
        en_tabs = docs["navigation"]["languages"][0]["tabs"]
        support_tab = next(t for t in en_tabs if t["tab"] == "Support: W&B Widgets")

        assert support_tab["hidden"] is True
        assert support_tab["pages"] == [
            "support/widgets",
            "support/widgets/tags/alpha",
            "support/widgets/tags/beta",
        ]

    def test_preserves_existing_tabs(self, sample_repo):
        """
        Non-support tabs (like 'Platform') should be preserved
        untouched after the update.
        """
        products = [{"slug": "widgets", "display_name": "W&B Widgets"}]
        generate_tags.update_docs_json(sample_repo, products, {"widgets": []})

        docs = json.loads((sample_repo / "docs.json").read_text())
        en_tabs = docs["navigation"]["languages"][0]["tabs"]
        platform_tab = next(t for t in en_tabs if t["tab"] == "Platform")
        assert platform_tab["pages"] == ["index"]

    def test_updates_existing_support_tab(self, sample_repo):
        """
        If a support tab already exists, its pages should be replaced
        (not duplicated) with the new listing.
        """
        # First, create the tab
        products = [{"slug": "widgets", "display_name": "W&B Widgets"}]
        generate_tags.update_docs_json(
            sample_repo, products,
            {"widgets": ["support/widgets/tags/old-tag"]},
        )

        # Then update it with different tags
        generate_tags.update_docs_json(
            sample_repo, products,
            {"widgets": ["support/widgets/tags/new-tag"]},
        )

        docs = json.loads((sample_repo / "docs.json").read_text())
        en_tabs = docs["navigation"]["languages"][0]["tabs"]
        support_tab = next(t for t in en_tabs if t["tab"] == "Support: W&B Widgets")

        assert support_tab["pages"] == [
            "support/widgets",
            "support/widgets/tags/new-tag",
        ]
        # Verify no duplicate tabs were created
        support_tabs = [t for t in en_tabs if "Support:" in t.get("tab", "")]
        assert len(support_tabs) == 1

    def test_docs_json_not_found(self, tmp_path):
        """
        If docs.json doesn't exist, a FileNotFoundError should be raised.
        """
        with pytest.raises(FileNotFoundError, match="docs.json not found"):
            generate_tags.update_docs_json(tmp_path, [], {})

    def test_missing_en_language(self, tmp_path):
        """
        If the English language entry is missing from navigation.languages,
        a ValueError should be raised with a helpful message.
        """
        docs = {"navigation": {"languages": [{"language": "ja", "tabs": []}]}}
        (tmp_path / "docs.json").write_text(json.dumps(docs), encoding="utf-8")

        with pytest.raises(ValueError, match="no entry with language='en'"):
            generate_tags.update_docs_json(tmp_path, [], {})

    def test_docs_json_trailing_newline(self, sample_repo):
        """
        The written docs.json should end with a trailing newline to
        match standard formatting.
        """
        products = [{"slug": "widgets", "display_name": "W&B Widgets"}]
        generate_tags.update_docs_json(sample_repo, products, {"widgets": []})

        raw = (sample_repo / "docs.json").read_text()
        assert raw.endswith("\n")


# ===========================================================================
# Tests: tojson_unicode filter
# ===========================================================================

class TestTojsonUnicode:
    """Tests for ``tojson_unicode`` (Jinja filter and helper)."""

    def test_create_template_env_registers_tojson_unicode(self, tmp_path):
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "empty.j2").write_text("", encoding="utf-8")
        env = generate_tags.create_template_env(templates_dir)
        assert env.filters["tojson_unicode"] is generate_tags.tojson_unicode

    def test_preserves_ampersand(self):
        """The & character should not be escaped to \\u0026."""
        result = generate_tags.tojson_unicode("Authentication & Access")
        assert result == '"Authentication & Access"'
        assert "\\u0026" not in result

    def test_preserves_angle_brackets(self):
        """< and > should not be escaped to unicode sequences."""
        result = generate_tags.tojson_unicode("A < B > C")
        assert "\\u003c" not in result.lower()
        assert "\\u003e" not in result.lower()

    def test_wraps_in_quotes(self):
        """String values should be wrapped in double quotes."""
        assert generate_tags.tojson_unicode("hello") == '"hello"'

    def test_integer_value(self):
        """Integer values should be serialized as plain numbers."""
        assert generate_tags.tojson_unicode(42) == "42"


# ===========================================================================
# Tests: update_support_index
# ===========================================================================

class TestUpdateSupportIndex:
    """Tests for the update_support_index function."""

    def test_updates_counts(self, tmp_path):
        """
        Product cards in support.mdx should have their article and tag
        counts replaced with the provided values.
        """
        (tmp_path / "support.mdx").write_text(textwrap.dedent("""\
            ---
            title: Support
            ---

            <Card title="W&B Widgets" href="/support/widgets" arrow="true">
              0 articles &middot; 0 tags
            </Card>
        """), encoding="utf-8")

        generate_tags.update_support_index(
            tmp_path,
            {"widgets": {"article_count": 42, "tag_count": 7}},
        )

        content = (tmp_path / "support.mdx").read_text()
        assert "42 articles &middot; 7 tags" in content
        assert "0 articles" not in content

    def test_singular_counts(self, tmp_path):
        """
        When a product has exactly 1 article or 1 tag, the count line
        should use singular forms: '1 article' and '1 tag'.
        """
        (tmp_path / "support.mdx").write_text(textwrap.dedent("""\
            ---
            title: Support
            ---

            <Card title="W&B Solo" href="/support/solo" arrow="true">
              0 articles &middot; 0 tags
            </Card>
        """), encoding="utf-8")

        generate_tags.update_support_index(
            tmp_path,
            {"solo": {"article_count": 1, "tag_count": 1}},
        )

        content = (tmp_path / "support.mdx").read_text()
        assert "1 article &middot; 1 tag" in content
        assert "articles" not in content
        assert "tags" not in content

    def test_preserves_surrounding_content(self, tmp_path):
        """
        All content outside the count line should remain untouched.
        """
        original = textwrap.dedent("""\
            ---
            title: Support
            mode: "center"
            ---

            ## Some heading

            <Card title="Product" href="/support/prod" arrow="true" icon="/icons/prod.svg">
              5 articles &middot; 3 tags
            </Card>

            ## Another section

            More content here.
        """)
        (tmp_path / "support.mdx").write_text(original, encoding="utf-8")

        generate_tags.update_support_index(
            tmp_path,
            {"prod": {"article_count": 10, "tag_count": 4}},
        )

        content = (tmp_path / "support.mdx").read_text()
        assert "## Some heading" in content
        assert "## Another section" in content
        assert "More content here." in content
        assert 'mode: "center"' in content
        assert "10 articles &middot; 4 tags" in content

    def test_multiple_products(self, tmp_path):
        """
        Multiple product cards should each be updated independently.
        """
        (tmp_path / "support.mdx").write_text(textwrap.dedent("""\
            ---
            title: Support
            ---

            <Card title="Models" href="/support/models" arrow="true">
              0 articles &middot; 0 tags
            </Card>
            <Card title="Weave" href="/support/weave" arrow="true">
              0 articles &middot; 0 tags
            </Card>
        """), encoding="utf-8")

        generate_tags.update_support_index(
            tmp_path,
            {
                "models": {"article_count": 100, "tag_count": 20},
                "weave": {"article_count": 15, "tag_count": 8},
            },
        )

        content = (tmp_path / "support.mdx").read_text()
        assert "100 articles &middot; 20 tags" in content
        assert "15 articles &middot; 8 tags" in content

    def test_missing_product_card_warns(self, tmp_path):
        """
        If a product slug has no matching Card in support.mdx, a warning
        should be emitted but the file should still be written.
        """
        (tmp_path / "support.mdx").write_text(textwrap.dedent("""\
            ---
            title: Support
            ---

            <Card title="Models" href="/support/models" arrow="true">
              5 articles &middot; 3 tags
            </Card>
        """), encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_tags.update_support_index(
                tmp_path,
                {"nonexistent": {"article_count": 1, "tag_count": 1}},
            )

            assert len(w) == 1
            assert "nonexistent" in str(w[0].message)

    def test_file_not_found(self, tmp_path):
        """
        If support.mdx does not exist, a FileNotFoundError should be raised.
        """
        with pytest.raises(FileNotFoundError, match="support.mdx not found"):
            generate_tags.update_support_index(tmp_path, {})


# ===========================================================================
# Tests: keyword footer sync
# ===========================================================================


class TestKeywordFooter:
    """Tests for build_keyword_footer_mdx and sync_support_article_footer."""

    def test_build_keyword_footer_empty(self):
        """No keywords means no footer block (empty string)."""
        assert generate_tags.build_keyword_footer_mdx("models", []) == ""
        assert generate_tags.build_tab_badges_mdx("models", []) == ""

    def test_build_keyword_footer_single(self):
        """One keyword produces a blank line, markers, and one Badge."""
        s = generate_tags.build_keyword_footer_mdx("weave", ["Code Capture"])
        assert s.startswith("\n\n")
        assert "[Code Capture](/support/weave/tags/code-capture)</Badge>" in s
        assert '<Badge stroke shape="pill" color="orange" size="md">' in s
        assert "AUTO-GENERATED: tab badges" in s
        assert s.endswith("{/* ---- END AUTO-GENERATED: tab badges ---- */}")

    def test_build_keyword_footer_preserves_keyword_order(self):
        """Badge order follows the keywords list order, not alphabetical."""
        s = generate_tags.build_keyword_footer_mdx("widgets", ["Beta", "Alpha"])
        assert s.index("Beta") < s.index("Alpha")

    def test_sync_adds_footer_when_missing(self, tmp_path):
        """An article with no tab Badges yet gets markers and Badges appended."""
        p = tmp_path / "a.mdx"
        p.write_text(
            textwrap.dedent("""\
                ---
                title: "T"
                keywords: ["Alpha"]
                ---

                Body only.
                """),
            encoding="utf-8",
        )
        assert generate_tags.sync_support_article_footer(p, "widgets") is True
        out = p.read_text()
        assert "Body only." in out
        assert "/support/widgets/tags/alpha" in out
        assert "AUTO-GENERATED: tab badges" in out
        assert out.endswith("{/* ---- END AUTO-GENERATED: tab badges ---- */}")

    def test_sync_idempotent_when_footer_matches(self, tmp_path):
        """If the marked footer already matches front matter, the file is not rewritten."""
        content = textwrap.dedent("""\
            ---
            title: "T"
            keywords: ["Alpha"]
            ---

            Body.

            ---

            {/* ---- AUTO-GENERATED: tab badges ----
              Managed by scripts/knowledgebase-nav/generate_tags.py from keywords in front matter.
              Do not edit between these markers by hand.
            ---- */}
            <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/alpha)</Badge>
            {/* ---- END AUTO-GENERATED: tab badges ---- */}""")
        p = tmp_path / "a.mdx"
        p.write_text(content, encoding="utf-8")
        assert generate_tags.sync_support_article_footer(p, "widgets") is False

    def test_sync_preserves_blank_line_after_end_marker(self, tmp_path):
        """EOF after the end marker (for example a trailing blank line) is kept."""
        p = tmp_path / "a.mdx"
        p.write_text(
            textwrap.dedent("""\
                ---
                title: "T"
                keywords: ["Alpha"]
                ---

                Body.

                ---

                {/* ---- AUTO-GENERATED: tab badges ----
                  Managed by scripts/knowledgebase-nav/generate_tags.py from keywords in front matter.
                  Do not edit between these markers by hand.
                ---- */}
                <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/alpha)</Badge>
                {/* ---- END AUTO-GENERATED: tab badges ---- */}

                """),
            encoding="utf-8",
        )
        assert generate_tags.sync_support_article_footer(p, "widgets") is False
        out = p.read_text()
        assert out.endswith("---- */}\n\n")

    def test_sync_fixes_wrong_href(self, tmp_path):
        """Wrong tag slug in a tab-page Badge href is replaced (path must match tab pattern)."""
        p = tmp_path / "a.mdx"
        p.write_text(
            textwrap.dedent("""\
                ---
                title: "T"
                keywords: ["Alpha"]
                ---

                Body.

                ---

                <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/not-alpha)</Badge>
                """),
            encoding="utf-8",
        )
        assert generate_tags.sync_support_article_footer(p, "widgets") is True
        assert "/support/widgets/tags/alpha" in p.read_text()

    def test_sync_fixes_wrong_href_preserves_eof_after_badges(self, tmp_path):
        """Badge href changes are wrapped in markers; trailing newlines are kept."""
        p = tmp_path / "a.mdx"
        p.write_text(
            textwrap.dedent("""\
                ---
                title: "T"
                keywords: ["Alpha"]
                ---

                Body.

                ---

                <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/not-alpha)</Badge>
                """).rstrip("\n") + "\n\n",
            encoding="utf-8",
        )
        assert generate_tags.sync_support_article_footer(p, "widgets") is True
        out = p.read_text()
        assert "/support/widgets/tags/alpha" in out
        assert out.endswith("---- */}\n\n")

    def test_sync_without_horizontal_rule_updates_tab_badges(self, tmp_path):
        """Tab Badges are synced even when the --- line was removed."""
        p = tmp_path / "a.mdx"
        p.write_text(
            textwrap.dedent("""\
                ---
                title: "T"
                keywords: ["Alpha"]
                ---

                Body line.

                <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/not-alpha)</Badge>
                """),
            encoding="utf-8",
        )
        assert generate_tags.sync_support_article_footer(p, "widgets") is True
        out = p.read_text()
        assert "/support/widgets/tags/alpha" in out
        body_after_fm = out.split("---", 2)[2]
        assert "\n\n---\n\n" not in body_after_fm

    def test_sync_preserves_non_tab_badge(self, tmp_path):
        """Badges that do not link to /support/<product>/tags/ are not removed."""
        p = tmp_path / "a.mdx"
        p.write_text(
            textwrap.dedent("""\
                ---
                title: "T"
                keywords: ["Alpha"]
                ---

                Text.

                <Badge stroke shape="pill" color="orange" size="md">[Other](https://example.com/doc)</Badge>

                ---

                {/* ---- AUTO-GENERATED: tab badges ----
                  Managed by scripts/knowledgebase-nav/generate_tags.py from keywords in front matter.
                  Do not edit between these markers by hand.
                ---- */}
                <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/alpha)</Badge>
                {/* ---- END AUTO-GENERATED: tab badges ---- */}
                """),
            encoding="utf-8",
        )
        assert generate_tags.sync_support_article_footer(p, "widgets") is False
        out = p.read_text()
        assert "https://example.com/doc" in out
        assert "/support/widgets/tags/alpha" in out

    def test_sync_removes_footer_when_keywords_empty(self, tmp_path):
        """keywords: [] removes the auto-managed footer."""
        p = tmp_path / "a.mdx"
        p.write_text(
            textwrap.dedent("""\
                ---
                title: "T"
                keywords: []
                ---

                Body.

                ---

                <Badge stroke shape="pill" color="orange" size="md">[Alpha](/support/widgets/tags/alpha)</Badge>
                """),
            encoding="utf-8",
        )
        assert generate_tags.sync_support_article_footer(p, "widgets") is True
        out = p.read_text()
        assert "<Badge" not in out
        assert "Body." in out

    def test_sync_warns_when_keywords_not_a_list(self, tmp_path):
        """A string ``keywords`` value warns and is coerced to one tag."""
        p = tmp_path / "a.mdx"
        p.write_text(
            textwrap.dedent("""\
                ---
                title: "T"
                keywords: "Alpha"
                ---

                Body.
                """),
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert generate_tags.sync_support_article_footer(p, "widgets") is True
        msgs = [str(x.message).lower() for x in w]
        assert any("keywords" in m for m in msgs)
        assert any("single tag" in m for m in msgs)
        out = p.read_text()
        assert "/support/widgets/tags/alpha" in out


# ===========================================================================
# Tests: full pipeline
# ===========================================================================

class TestFullPipeline:
    """End-to-end test using run_pipeline with a mock repo."""

    def test_full_pipeline(self, sample_repo, sample_config):
        """
        Running the full pipeline should create tag pages, a product
        index page, and update docs.json navigation in one pass.

        This test verifies the orchestration works end-to-end with a
        minimal mock repo.
        """
        generate_tags.run_pipeline(sample_repo, sample_config)

        # Tag pages should exist
        assert (sample_repo / "support" / "widgets" / "tags" / "alpha.mdx").exists()
        assert (sample_repo / "support" / "widgets" / "tags" / "beta.mdx").exists()

        # Product index should exist
        assert (sample_repo / "support" / "widgets.mdx").exists()
        index_content = (sample_repo / "support" / "widgets.mdx").read_text()
        assert "Browse by category" in index_content
        assert "Featured articles" in index_content

        # docs.json should have the support tab
        docs = json.loads((sample_repo / "docs.json").read_text())
        en_tabs = docs["navigation"]["languages"][0]["tabs"]
        support_tab = next(
            (t for t in en_tabs if t.get("tab") == "Support: W&B Widgets"),
            None,
        )
        assert support_tab is not None
        assert support_tab["hidden"] is True
        assert "support/widgets" in support_tab["pages"]
        assert "support/widgets/tags/alpha" in support_tab["pages"]
        assert "support/widgets/tags/beta" in support_tab["pages"]

        # support.mdx should have updated counts
        support_content = (sample_repo / "support.mdx").read_text()
        assert "3 articles &middot; 2 tags" in support_content

        # Article footers should list every keyword (article-one has Alpha and Beta)
        one = (sample_repo / "support" / "widgets" / "articles" / "article-one.mdx").read_text()
        assert "/support/widgets/tags/alpha" in one
        assert "/support/widgets/tags/beta" in one

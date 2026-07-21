#!/usr/bin/env python3
"""
Generate TypeScript SDK reference documentation for Mintlify.

This script generates TypeScript documentation using typedoc and converts
it to Mintlify's format.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_node_dependencies():
    """Check if Node.js and npm are available."""
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        print("✓ Node.js is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Node.js is not installed", file=sys.stderr)
        print("  Please install Node.js 18+ from https://nodejs.org/")
        sys.exit(1)
    
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("✓ npm is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ npm is not installed", file=sys.stderr)
        sys.exit(1)


def download_weave_source(version="main"):
    """Download Weave source code for a specific version using shallow clone."""
    print(f"\nDownloading Weave source code (version: {version})...")
    
    # Handle "latest" by fetching the latest release tag
    if version == "latest":
        # Use git ls-remote to get latest tag without API calls
        try:
            result = subprocess.run(
                ["git", "ls-remote", "--tags", "--sort=-v:refname", "https://github.com/wandb/weave.git"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the output to get the latest version tag
            for line in result.stdout.strip().split('\n'):
                if '\trefs/tags/' in line and not line.endswith('^{}'):
                    tag = line.split('\trefs/tags/')[-1]
                    if tag.startswith('v') and not 'dev' in tag and not 'rc' in tag:
                        version = tag
                        print(f"  Using latest release: {version}")
                        break
            else:
                print("  Warning: Could not determine latest release, using main branch")
                version = "main"
        except Exception as e:
            print(f"  Warning: Could not fetch latest release, using main branch: {e}")
            version = "main"
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    weave_dir = Path(temp_dir) / "weave"
    
    try:
        # Use shallow clone with single branch
        print(f"  Cloning Weave repository (shallow clone)...")
        clone_cmd = [
            "git", "clone",
            "--depth", "1",
            "--single-branch"
        ]
        
        # Add branch/tag specification
        if version != "main":
            clone_cmd.extend(["--branch", version])
        
        clone_cmd.extend([
            "https://github.com/wandb/weave.git",
            str(weave_dir)
        ])
        
        # Run the clone command
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
        
        print(f"  ✓ Successfully cloned Weave {version} to {weave_dir}")
        
        return weave_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning Weave repository: {e}", file=sys.stderr)
        if e.stderr:
            print(f"  stderr: {e.stderr}", file=sys.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading Weave source: {e}", file=sys.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)


def setup_typescript_project(weave_source):
    """Set up the TypeScript project and install dependencies."""
    sdk_path = Path(weave_source) / "sdks" / "node"
    
    if not sdk_path.exists():
        print(f"TypeScript SDK not found at {sdk_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nSetting up TypeScript project...")
    os.chdir(sdk_path)
    
    try:
        # Install dependencies
        print("  Installing dependencies...")
        subprocess.run(["npm", "install", "--legacy-peer-deps"], check=True)

        # Install typedoc and markdown plugin with compatible versions
        print("  Installing typedoc...")
        subprocess.run([
            "npm", "install", "--save-dev", "--legacy-peer-deps",
            "typedoc@0.28.20",
            "typedoc-plugin-markdown@4.12.0"
        ], check=True)
        
        print("  ✓ Dependencies installed")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}", file=sys.stderr)
        sys.exit(1)
    
    return sdk_path


def generate_typedoc(sdk_path, output_path):
    """Generate TypeScript documentation using typedoc."""
    print(f"\nGenerating TypeScript documentation...")
    
    # Create typedoc config
    config = {
        "entryPoints": ["src/index.ts"],
        "tsconfig": "./tsconfig.esm.json",
        "out": str(output_path),
        "plugin": ["typedoc-plugin-markdown"],
        "readme": "none",
        "githubPages": False,
        "excludePrivate": True,
        "excludeProtected": True,
        "excludeInternal": True,
        "disableSources": False,
        "cleanOutputDir": True,
        "hideBreadcrumbs": True,
        # The Weave repo currently ships a type error in the googleAdk
        # integration (duplicate @google/genai versions in its dependency
        # tree). Docs generation doesn't require the project to type-check,
        # so don't let that abort the build.
        "skipErrorChecking": True,
        # typedoc-plugin-markdown v4 also prepends a bold package-name
        # header to every page; Mintlify provides its own page chrome.
        "hidePageHeader": True
    }
    
    config_path = sdk_path / "typedoc.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run typedoc
        subprocess.run(["npx", "typedoc"], check=True, cwd=sdk_path)
        print("  ✓ Documentation generated")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate documentation: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up config file
        if config_path.exists():
            config_path.unlink()


def _escape_mdx_hostile_chars(content):
    """Escape raw `<`, `{`, and `}` in prose so MDX doesn't parse them as JSX.

    TypeDoc escapes these in the output it generates itself, but comment text
    inherited from third-party .d.ts files passes through verbatim (e.g.
    @google/adk ships a corrupted doc comment containing raw code, which
    TypeDoc 0.28 inherits onto WeaveAdkPlugin's members via its `implements`
    clause). Fenced code blocks and inline code spans are left untouched;
    already-escaped characters are not double-escaped.
    """
    out_lines = []
    in_fence = False
    for line in content.split('\n'):
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        # Even indices are prose; odd indices are inline code spans.
        parts = re.split(r'(``[^`]*``|`[^`]*`)', line)
        for i in range(0, len(parts), 2):
            p = re.sub(r'(?<!\\)<(?=[A-Za-z/])', r'\\<', parts[i])
            p = re.sub(r'(?<!\\)([{}])', r'\\\1', p)
            parts[i] = p
        out_lines.append(''.join(parts))
    return '\n'.join(out_lines)


def convert_to_mintlify_format(docs_dir):
    """Convert TypeDoc markdown to Mintlify MDX format."""
    print(f"\nConverting to Mintlify format...")
    
    docs_path = Path(docs_dir)
    md_files = list(docs_path.rglob("*.md"))
    
    for md_file in md_files:
        content = md_file.read_text()
        
        # Skip if already has frontmatter
        if content.startswith("---"):
            continue

        # Remove TypeDoc's in-page breadcrumb line. Mintlify manages breadcrumbs already.
        content = re.sub(r'^\[weave\]\([^)]+\)(?: / [^\n]+)+\n+', '', content, flags=re.MULTILINE)
        content = re.sub(r'\Aweave(?: / [^\n]+)*\n+', '', content)

        # Extract title from content
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else md_file.stem
        
        # Remove the title line if it exists (Mintlify uses frontmatter title)
        if title_match:
            content = content.replace(title_match.group(0), '', 1).strip()
        
        # Fix escaped angle brackets in title (TypeDoc escapes them as \< and \>)
        title_fixed = title.replace('\\<', '<').replace('\\>', '>')

        # typedoc-plugin-markdown v4 wraps deprecated symbols' titles in
        # strikethrough (`# ~~Variable: startSession~~`); the deprecation is
        # surfaced via the hoisted <Warning> callout instead.
        title_fixed = title_fixed.replace('~~', '')

        # Strip TypeDoc's reflection-kind prefix ("Class: LLM" → "LLM") so the
        # left nav shows bare symbol names; the nav group already conveys the kind.
        # v4 writes "Type Alias"; v3 wrote "Type alias".
        title_fixed = re.sub(
            r'^(?:Class|Interface|Function|Type [Aa]lias|Enumeration|Namespace|Variable|Module):\s+',
            '',
            title_fixed,
        )

        # v4 suffixes function/method titles with call parens ("login()");
        # drop them to keep bare symbol names in the nav.
        title_fixed = re.sub(r'\(\)$', '', title_fixed)

        # Escape MDX-hostile characters in the body before the frontmatter is
        # prepended (the quoted YAML title must keep its bare < and >).
        content = _escape_mdx_hostile_chars(content)
        
        # Add Mintlify frontmatter
        frontmatter = f"""---
title: "{title_fixed}"
description: "TypeScript SDK reference"
---

"""
        content = frontmatter + content
        
        # Fix TypeDoc specific formatting
        # Convert **`code`** to just `code`
        content = re.sub(r'\*\*`([^`]+)`\*\*', r'`\1`', content)
        
        # Fix parameter tables
        content = re.sub(r'\|\s*:--\s*\|', '| --- |', content)

        # Enable Mintlify Twoslash on TS code blocks for IDE-style hover types.
        # `// @noErrors` suppresses type errors from non-self-contained snippets.
        content = re.sub(
            r'^```(ts|typescript)([^\n]*)$',
            r'```\1\2 twoslash\n// @noErrors',
            content,
            flags=re.MULTILINE,
        )

        # Fix internal links to use relative paths with lowercase filenames
        # TypeDoc generates links like ../classes/WeaveObject.md which need fixing
        
        # 1. Remove .md extensions from all links
        content = re.sub(r'\.md(#[^)]+)?\)', r'\1)', content)
        
        # 2. Fix links to README/index to point to landing page
        content = re.sub(r'\]\(\.\./README', '](../', content)
        content = re.sub(r'\]\(\.\./index', '](../', content)
        # Don't convert self-referential README anchor links - they'll be fixed later in extract_members_to_separate_files()
        # Only convert non-anchor README links
        content = re.sub(r'\]\(README\)', '](typescript-sdk)', content)
        
        # 3. Fix all TypeDoc-generated links to lowercase
        # Paths that already have directory structure (../classes/, ../interfaces/, etc.)
        content = re.sub(r'\]\(\.\./classes/([^)#]+)(#[^)]+)?\)', lambda m: f'](../classes/{m.group(1).lower()}{m.group(2) or ""})', content)
        content = re.sub(r'\]\(\.\./interfaces/([^)#]+)(#[^)]+)?\)', lambda m: f'](../interfaces/{m.group(1).lower()}{m.group(2) or ""})', content)
        content = re.sub(r'\]\(\.\./functions/([^)#]+)(#[^)]+)?\)', lambda m: f'](../functions/{m.group(1).lower()}{m.group(2) or ""})', content)
        content = re.sub(r'\]\(\.\./type-aliases/([^)#]+)(#[^)]+)?\)', lambda m: f'](../type-aliases/{m.group(1).lower()}{m.group(2) or ""})', content)
        content = re.sub(r'\]\(\.\./variables/([^)#]+)(#[^)]+)?\)', lambda m: f'](../variables/{m.group(1).lower()}{m.group(2) or ""})', content)

        # 4. Fix relative links without ../ prefix (same directory or subdirectory)
        content = re.sub(r'\]\(classes/([^)#]+)(#[^)]+)?\)', lambda m: f'](../classes/{m.group(1).lower()}{m.group(2) or ""})', content)
        content = re.sub(r'\]\(interfaces/([^)#]+)(#[^)]+)?\)', lambda m: f'](../interfaces/{m.group(1).lower()}{m.group(2) or ""})', content)
        content = re.sub(r'\]\(functions/([^)#]+)(#[^)]+)?\)', lambda m: f'](../functions/{m.group(1).lower()}{m.group(2) or ""})', content)
        content = re.sub(r'\]\(type-aliases/([^)#]+)(#[^)]+)?\)', lambda m: f'](../type-aliases/{m.group(1).lower()}{m.group(2) or ""})', content)
        content = re.sub(r'\]\(variables/([^)#]+)(#[^)]+)?\)', lambda m: f'](../variables/{m.group(1).lower()}{m.group(2) or ""})', content)

        # 5. Fix same-directory class/interface links (start with capital letter, no path separator)
        # Allow digits and hyphens for TypeDoc's name-collision suffixes (e.g. Session-1)
        content = re.sub(r'\]\(([A-Z][a-zA-Z0-9-]+)(#[^)]+)?\)', lambda m: f'](./{m.group(1).lower()}{m.group(2) or ""})', content)
        
        # 6. Special fix for README/landing page - it becomes typescript-sdk.mdx at parent level
        if md_file.name == 'README.md':
            # The landing page will be at weave/reference/typescript-sdk.mdx
            # so links to subdirectories need ./typescript-sdk/ prefix
            content = re.sub(r'\]\(\.\./classes/([^)#]+)(#[^)]+)?\)', lambda m: f'](./typescript-sdk/classes/{m.group(1).lower()}{m.group(2) or ""})', content)
            content = re.sub(r'\]\(\.\./interfaces/([^)#]+)(#[^)]+)?\)', lambda m: f'](./typescript-sdk/interfaces/{m.group(1).lower()}{m.group(2) or ""})', content)
            content = re.sub(r'\]\(\.\./functions/([^)#]+)(#[^)]+)?\)', lambda m: f'](./typescript-sdk/functions/{m.group(1).lower()}{m.group(2) or ""})', content)
            content = re.sub(r'\]\(\.\./type-aliases/([^)#]+)(#[^)]+)?\)', lambda m: f'](./typescript-sdk/type-aliases/{m.group(1).lower()}{m.group(2) or ""})', content)
            content = re.sub(r'\]\(\.\./variables/([^)#]+)(#[^)]+)?\)', lambda m: f'](./typescript-sdk/variables/{m.group(1).lower()}{m.group(2) or ""})', content)

            # Fix self-referential anchor links like (README#anchor) that appear in Table of Contents
            # We'll scan the content to determine if each anchor refers to a function or type alias
            # First, extract all type alias names (they have the Ƭ symbol)
            type_alias_names = set()
            type_alias_matches = re.findall(r'###\s+([A-Za-z][A-Za-z0-9]*)\n\nƬ', content)
            for name in type_alias_matches:
                type_alias_names.add(name.lower())
            
            # Now convert README anchor links to proper paths based on whether they're type aliases or functions
            def fix_readme_anchor(match):
                anchor = match.group(1).lower()
                if anchor in type_alias_names:
                    return f'](./typescript-sdk/type-aliases/{anchor})'
                else:
                    # Assume it's a function if not a type alias
                    return f'](./typescript-sdk/functions/{anchor})'
            
            content = re.sub(r'\]\(README#([a-zA-Z][a-zA-Z0-9-]*)\)', fix_readme_anchor, content)
        
        # Special handling for index files (README.md)
        # These files are at the root level and link directly to subdirectories
        # Keep these as relative links too
        if md_file.name == "README.md":
            # Links like [Dataset](classes/Dataset.md) are already relative
            # Just ensure .md extension is removed (already done above)
            pass
        
        # Write as .mdx file with lowercase filename (avoid Git case sensitivity issues)
        lowercase_stem = md_file.stem.lower()
        mdx_file = md_file.parent / f"{lowercase_stem}.mdx"
        mdx_file.write_text(content)
        
        # Remove original .md file
        md_file.unlink()
        
        print(f"  ✓ Converted {md_file.name} → {mdx_file.name}")


# Matches anchor links that point at the README/landing page in any of the
# forms TypeDoc + earlier conversion steps leave behind: `](README#X)`,
# `](./README#X)`, `](readme#X)`, `](./readme#X)`, `](typescript-sdk#X)`,
# `](./typescript-sdk#X)`.
_LANDING_ANCHOR_LINK_RE = re.compile(
    r'\]\((?:\./)?(?:readme|README|typescript-sdk)#([A-Za-z][A-Za-z0-9-]*)\)'
)


_HEADING_KIND_RE = re.compile(r'^### ([A-Za-z][A-Za-z0-9]*)\n\n(\S)', re.MULTILINE)


def _build_landing_anchor_map(landing_content):
    """Walk every `### Name` heading on the consolidated README/landing page
    and return a dict mapping each heading's original anchor slug (with the
    `-N` disambiguator TypeDoc uses for duplicates) to a (kind, segment)
    tuple, where kind is 'function' | 'type-alias' | 'other'.

    For headings that will be extracted (functions, type aliases), `segment`
    is the lowercased base name — that's the extracted file's stem. For
    headings that survive on the landing page, `segment` is the
    **post-extraction** slug: once same-base function/type-alias siblings
    are pulled out, the surviving 'other' heading collapses back to the
    plain base slug. That's why `[X](./readme#session-1)`, where `session`
    is the extracted type alias and `session-1` is the surviving Variable,
    must resolve to `#session`, not `#session-1`.
    """
    heads = []
    orig_counter = {}
    for m in _HEADING_KIND_RE.finditer(landing_content):
        name = m.group(1)
        first = m.group(2)
        if first == '▸':
            kind = 'function'
        elif first == 'Ƭ':
            kind = 'type-alias'
        else:
            kind = 'other'
        base = name.lower()
        n = orig_counter.get(base, 0)
        orig_slug = base if n == 0 else f'{base}-{n}'
        orig_counter[base] = n + 1
        heads.append((base, kind, orig_slug))

    target_by_slug = {}
    post_counter = {}
    for base, kind, orig_slug in heads:
        if kind in ('function', 'type-alias'):
            target_by_slug[orig_slug] = (kind, base)
        else:
            n = post_counter.get(base, 0)
            post_slug = base if n == 0 else f'{base}-{n}'
            post_counter[base] = n + 1
            target_by_slug[orig_slug] = ('other', post_slug)
    return target_by_slug


def _make_landing_anchor_resolver(target_by_slug, functions_prefix,
                                   type_aliases_prefix, fallback_prefix):
    """Return a `re.sub` callback that rewrites a landing-page anchor link
    (e.g. `](README#getCurrentConversation)`) to the right cross-file or
    in-page link.

    Pre-0.53.0 the extractor could assume every README anchor target was a
    type alias. v0.53.0 added Conversation-replacement *functions* and a
    backwards-compat `Session` Variable export — so we now consult the
    pre-built kind map and route accordingly. Unknown anchors are left
    untouched so the broken-link checker surfaces them rather than having
    us silently route them to a wrong path.
    """
    def resolve(match):
        anchor = match.group(1).lower()
        info = target_by_slug.get(anchor)
        if info is None:
            return match.group(0)
        kind, segment = info
        if kind == 'function':
            return f']({functions_prefix}{segment})'
        if kind == 'type-alias':
            return f']({type_aliases_prefix}{segment})'
        return f']({fallback_prefix}{segment})'

    return resolve


def extract_members_to_separate_files(docs_path):
    """Extract function and type-alias documentation from typescript-sdk.mdx landing page to separate files."""
    # Check for the landing page
    landing_file = docs_path.parent / "typescript-sdk.mdx"
    if landing_file.exists():
        index_file = landing_file
    else:
        # Fallback to index.mdx if it exists (shouldn't happen with new generation)
        index_file = docs_path / "index.mdx"
        if not index_file.exists():
            return

    content = index_file.read_text()

    # Check if we need to extract anything
    has_functions = "### Functions" in content
    has_type_aliases = "### Type Aliases" in content or "### Type aliases" in content

    if not has_functions and not has_type_aliases:
        return

    print("  Extracting consolidated members to separate files...")

    # Pre-scan once so the per-file extraction loops and the post-extraction
    # landing-page rewrite below can all disambiguate README anchor links
    # against the same set of symbol kinds (and post-extraction slugs).
    landing_anchor_map = _build_landing_anchor_map(content)

    resolver_for_function_pages = _make_landing_anchor_resolver(
        landing_anchor_map,
        functions_prefix='./',
        type_aliases_prefix='../type-aliases/',
        fallback_prefix='../typescript-sdk#',
    )
    resolver_for_type_alias_pages = _make_landing_anchor_resolver(
        landing_anchor_map,
        functions_prefix='../functions/',
        type_aliases_prefix='./',
        fallback_prefix='../typescript-sdk#',
    )
    resolver_for_landing_page = _make_landing_anchor_resolver(
        landing_anchor_map,
        functions_prefix='./typescript-sdk/functions/',
        type_aliases_prefix='./typescript-sdk/type-aliases/',
        fallback_prefix='#',
    )
    
    # Extract functions
    if has_functions:
        functions_dir = docs_path / "functions"
        functions_dir.mkdir(exist_ok=True)
        
        # Find and extract each function section
        # Pattern to match function sections (from ### functionName followed by ▸ symbol to the next ### or end)
        # The ▸ symbol (U+25B8) is used by TypeDoc to mark function signatures
        function_pattern = re.compile(
            r'(### ([a-zA-Z][a-zA-Z0-9]*)\n\n▸.*?)(?=\n### |\Z)',
            re.DOTALL
        )
        
        functions_found = []
        for match in function_pattern.finditer(content):
            func_content = match.group(1)
            func_name = match.group(2)
            
            # Fix links when moving from landing page to functions subdirectory
            # ./typescript-sdk/classes/X -> ../classes/X
            # ./typescript-sdk/interfaces/X -> ../interfaces/X
            func_content = func_content.replace('./typescript-sdk/classes/', '../classes/')
            func_content = func_content.replace('./typescript-sdk/interfaces/', '../interfaces/')
            func_content = func_content.replace('./typescript-sdk/type-aliases/', '../type-aliases/')
            func_content = func_content.replace('./typescript-sdk/functions/', './')

            # Rewrite anchor links to README/landing-page headings based on
            # what each anchor actually refers to (function, type alias, or
            # surviving landing-page symbol).
            func_content = _LANDING_ANCHOR_LINK_RE.sub(resolver_for_function_pages, func_content)
            
            # Create the function file content
            func_title = func_name
            func_file_content = f"""---
title: "{func_title}"
description: "TypeScript SDK reference"
---

{func_content.replace(f'### {func_name}', f'# {func_name}')}"""
            
            # Write the function file with lowercase filename (avoid Git case sensitivity issues)
            func_filename = func_name.lower()
            func_file = functions_dir / f"{func_filename}.mdx"
            func_file.write_text(func_file_content)
            functions_found.append((func_name, func_filename))
            print(f"    ✓ Extracted {func_filename}.mdx")

        if functions_found:
            # Remove the detailed function documentation from index
            content = function_pattern.sub('', content)

            # Update the Functions section with links (functions_found already has lowercase names)
            # The landing page is at typescript-sdk.mdx, so links need ./typescript-sdk/ prefix
            functions_section = "\n### Functions\n\n"
            for func_label, func_filename in functions_found:
                functions_section += f"- [{func_label}](./typescript-sdk/functions/{func_filename})\n"
            
            # Replace existing Functions section with links
            content = re.sub(
                r'### Functions.*?(?=\n### |\n## |\Z)',
                functions_section,
                content,
                flags=re.DOTALL
            )
    
    # Extract type aliases
    if has_type_aliases:
        type_aliases_dir = docs_path / "type-aliases"
        type_aliases_dir.mkdir(exist_ok=True)
        
        # Pattern to match all type alias sections
        # Look for ### TypeAliasName pattern
        type_alias_pattern = re.compile(
            r'(### ([A-Za-z][A-Za-z0-9]*)\n\nƬ .*?)(?=\n### |\Z)',
            re.DOTALL
        )
        
        type_aliases = type_alias_pattern.findall(content)
        for alias_content, alias_name in type_aliases:
            # Skip if it's not a type alias (e.g., if it's a function or class)
            if not alias_content.startswith(f"### {alias_name}\n\nƬ "):
                continue
            
            # Fix links when moving from landing page to type-aliases subdirectory
            # ./typescript-sdk/classes/X -> ../classes/X
            # ./typescript-sdk/interfaces/X -> ../interfaces/X
            alias_content = alias_content.replace('./typescript-sdk/classes/', '../classes/')
            alias_content = alias_content.replace('./typescript-sdk/interfaces/', '../interfaces/')
            alias_content = alias_content.replace('./typescript-sdk/functions/', '../functions/')
            alias_content = alias_content.replace('./typescript-sdk/type-aliases/', './')

            # Same disambiguating rewrite as for function pages — except a
            # same-kind reference here means another type alias (sibling
            # file), and a different-kind reference is a function.
            alias_content = _LANDING_ANCHOR_LINK_RE.sub(resolver_for_type_alias_pages, alias_content)
                
            # Create the type alias file content
            alias_file_content = f"""---
title: "{alias_name}"
description: "TypeScript SDK reference"
---

{alias_content.replace(f'### {alias_name}', f'# {alias_name}')}"""
            
            # Write the type alias file with lowercase filename (avoid Git case sensitivity issues)
            alias_filename = alias_name.lower()
            alias_file = type_aliases_dir / f"{alias_filename}.mdx"
            alias_file.write_text(alias_file_content)
            print(f"    ✓ Extracted {alias_filename}.mdx")
        
        # Remove all extracted type aliases from index
        content = type_alias_pattern.sub('', content)
            
        # Update Type Aliases section with links to all extracted type aliases (use lowercase filenames)
        # The landing page is at typescript-sdk.mdx, so links need ./typescript-sdk/ prefix
        if type_aliases:
            type_aliases_links = [f"- [{name}](./typescript-sdk/type-aliases/{name.lower()})" for _, name in type_aliases if _.startswith(f"### {name}\n\nƬ ")]
            if type_aliases_links:
                type_aliases_section = "\n### Type Aliases\n\n" + "\n".join(sorted(type_aliases_links)) + "\n"
                
                # Replace existing Type Aliases section
                content = re.sub(
                    r'### Type [Aa]liases.*?(?=\n### |\n## |\Z)',
                    type_aliases_section,
                    content,
                    flags=re.DOTALL
                )
    
    # Rewrite any remaining README/landing-page anchor links on the landing
    # page itself. The earlier `fix_readme_anchor` pass in
    # convert_to_mintlify_format only catches uppercase `](README#X)`, but
    # the general lowercasing rule rewrites those to `](./readme#X)` before
    # it runs — so links like `[Session](./readme#session-1)` survive until
    # here. Using the post-extraction symbol-kind sets also lets us route
    # surviving landing-page headings (e.g. backwards-compat Variables) to
    # an in-page anchor rather than a non-existent extracted file.
    content = _LANDING_ANCHOR_LINK_RE.sub(resolver_for_landing_page, content)

    # Write updated content back
    index_file.write_text(content)
    if index_file.name == "typescript-sdk.mdx":
        print(f"  ✓ Updated typescript-sdk.mdx landing page")
    else:
        print(f"  ✓ Updated index.mdx")


def organize_for_mintlify(temp_output, final_output):
    """Organize the generated docs according to Mintlify structure."""
    print(f"\nOrganizing documentation structure...")
    
    temp_path = Path(temp_output)
    final_path = Path(final_output)
    
    # Instead of deleting everything, selectively update
    # This preserves files that might not be regenerated
    if not final_path.exists():
        final_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from temp to final, overwriting existing ones
    for item in temp_path.iterdir():
        dest = final_path / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # Move readme.mdx to typescript-sdk.mdx landing page if it exists
    # Check for both uppercase and lowercase versions (case sensitivity)
    readme_path = final_path / "readme.mdx"
    readme_path_upper = final_path / "README.mdx"
    
    if readme_path.exists():
        landing_path = final_path.parent / "typescript-sdk.mdx"
        readme_path.rename(landing_path)
        print("  ✓ Created typescript-sdk.mdx landing page from readme.mdx")
    elif readme_path_upper.exists():
        landing_path = final_path.parent / "typescript-sdk.mdx"
        readme_path_upper.rename(landing_path)
        print("  ✓ Created typescript-sdk.mdx landing page from README.mdx")
    
    # Extract functions and type aliases to separate files if they're consolidated
    extract_members_to_separate_files(final_path)

    # Promote `@deprecated` JSDoc tags into prominent Mintlify <Warning>
    # callouts hoisted under each symbol's heading. Must run AFTER extraction
    # so the `### name\n\n▸` pattern that extraction relies on is still intact.
    hoist_deprecation_callouts(final_path)

    print("  ✓ Documentation organized")


# Matches TypeDoc's @deprecated output. typedoc-plugin-markdown v4 renders it
# as a heading section whose level depends on nesting depth (`## Deprecated`
# at page level, `#### Deprecated` under a member, `##### Deprecated` under an
# accessor signature):
#   ## Deprecated
#
#   <message paragraph, possibly multiline>
#
# (v3 rendered an inline `` `Deprecated` `` label instead; that form is also
# still matched.) Stops at the next heading or `***`/`___` separator (or end
# of file). The non-greedy `.+?` plus the lookahead lets the message itself
# span multiple soft-wrapped lines without swallowing the section that
# follows it.
_DEPRECATED_BLOCK_RE = re.compile(
    r'\n\n(?:(#{2,5}) Deprecated|`Deprecated`)\n\n(.+?)(?=\n\n(?:#{1,6} |\*\*\*|___)|\n*\Z)',
    re.DOTALL,
)

# Symbol-level headings only. H4 (`#### Returns`, `#### Defined in`) is a
# *sub*section of a symbol and is deliberately excluded — anchoring the
# warning there would put it back where TypeDoc already placed it. The
# `Deprecated` headings themselves are excluded too: they belong to the very
# blocks being removed, so anchoring a later warning to one would insert
# text into a deleted span.
_SYMBOL_HEADING_RE = re.compile(r'^#{1,3} (?!Deprecated$).+$', re.MULTILINE)

# Frontmatter block at the very start of a converted .mdx file.
_FRONTMATTER_RE = re.compile(r'\A---\n.*?\n---\n', re.DOTALL)


def hoist_deprecation_callouts(docs_root):
    """Convert inline `Deprecated` markers into hoisted Mintlify <Warning>s.

    TypeDoc renders `@deprecated` as a plain inline label after the Returns
    block, which is easy to overlook. For each `Deprecated` marker we:
      1. Replace it with a Mintlify <Warning> callout for visual weight.
      2. Move it directly under the nearest preceding symbol heading
         (H1-H3) so it's the first thing a reader sees for that symbol.
    """
    docs_root = Path(docs_root)
    targets = list(docs_root.rglob("*.mdx"))
    landing = docs_root.parent / "typescript-sdk.mdx"
    if landing.exists():
        targets.append(landing)

    updated = 0
    for mdx_file in targets:
        original = mdx_file.read_text()
        new_content = _hoist_deprecations_in_text(original)
        if new_content != original:
            mdx_file.write_text(new_content)
            updated += 1

    if updated:
        print(f"  ✓ Hoisted deprecation callouts in {updated} file(s)")


def _hoist_deprecations_in_text(content):
    deprecations = list(_DEPRECATED_BLOCK_RE.finditer(content))
    if not deprecations:
        return content

    headings = list(_SYMBOL_HEADING_RE.finditer(content))

    # A page-level deprecation can precede every heading (the H1 was already
    # moved into frontmatter during conversion), so the fallback anchor must
    # sit after the frontmatter, never at position 0.
    fm = _FRONTMATTER_RE.match(content)
    default_anchor = fm.end() if fm else 0

    edits = []
    inserts_by_pos = {}

    for dep in deprecations:
        message = dep.group(2).strip()
        warning = f'\n\n<Warning>\n  **Deprecated.** {message}\n</Warning>'

        # An `## Deprecated` (H2) section deprecates the page's own symbol —
        # its H1 heading is already gone (moved into frontmatter), and any H2
        # headings preceding it (`## Parameters`, `## Examples`, ...) are
        # sections of the same symbol, not other symbols. Hoist it to the top
        # of the page. Deeper levels (H4/H5) belong to a member documented
        # under an H2/H3 heading, so those anchor to the nearest one.
        if dep.group(1) == '##':
            anchor = default_anchor
        else:
            anchor = default_anchor
            for h in headings:
                if h.start() < dep.start():
                    anchor = h.end()
                else:
                    break

        edits.append((dep.start(), dep.end(), ''))
        # Heading-anchored inserts land before the blank line that already
        # follows the heading; a top-of-page insert lands right before the
        # first content line, so it needs its own trailing blank line to
        # keep the following markdown block separate from the JSX element.
        if anchor == default_anchor:
            warning += '\n'
        inserts_by_pos.setdefault(anchor, []).append(warning)

    for pos, warns in inserts_by_pos.items():
        edits.append((pos, pos, ''.join(warns)))

    # Reverse-order application keeps earlier offsets valid.
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)

    for start, end, replacement in edits:
        content = content[:start] + replacement + content[end:]

    return content


def cleanup_temp_dirs(*paths):
    """Clean up temporary directories."""
    for path in paths:
        if path and Path(path).exists():
            shutil.rmtree(path, ignore_errors=True)


def main():
    """Main function."""
    # Check dependencies
    check_node_dependencies()
    
    # Get Weave version from environment or use latest
    weave_version = os.environ.get("WEAVE_VERSION", "latest")
    
    # Store the original working directory
    original_cwd = os.getcwd()
    
    # Download Weave source
    weave_source = download_weave_source(weave_version)
    
    try:
        # Set up TypeScript project
        sdk_path = setup_typescript_project(weave_source)
        
        # Generate documentation to a temporary directory
        temp_output = tempfile.mkdtemp()
        generate_typedoc(sdk_path, temp_output)
        
        # Convert to Mintlify format
        convert_to_mintlify_format(temp_output)
        
        # Organize for Mintlify - use absolute path from original directory
        final_output = os.path.join(original_cwd, "weave/reference/typescript-sdk")
        organize_for_mintlify(temp_output, final_output)
        
        print("\n✓ TypeScript SDK documentation generation complete!")
        
    finally:
        # Clean up temporary directories
        cleanup_temp_dirs(weave_source.parent if 'weave_source' in locals() else None,
                         temp_output if 'temp_output' in locals() else None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Detect chained redirects in Mintlify docs.json configuration.

A chained redirect occurs when a redirect destination is itself a redirect source,
causing multiple hops to reach the final destination. This can impact performance
and create maintenance issues.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_redirects(docs_json_path: Path) -> List[Dict[str, str]]:
    """Load redirects from docs.json file."""
    try:
        with open(docs_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('redirects', [])
    except FileNotFoundError:
        print(f"Error: Could not find {docs_json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {docs_json_path}: {e}", file=sys.stderr)
        sys.exit(1)


def build_redirect_map(redirects: List[Dict[str, str]]) -> Dict[str, str]:
    """Build a map of source -> destination from redirects list."""
    redirect_map = {}
    for r in redirects:
        source = r.get('source')
        destination = r.get('destination')
        if source and destination:
            redirect_map[source] = destination
    return redirect_map


def find_chained_redirects(redirect_map: Dict[str, str]) -> List[List[str]]:
    """
    Find all chained redirects in the redirect map.

    Returns a list of chains, where each chain is a list of URLs
    showing the redirect path from source to final destination.
    """
    chains = []

    for source, destination in redirect_map.items():
        # Check if the destination is also a redirect source
        if destination in redirect_map:
            chain = [source, destination]
            current = destination
            visited = set(chain)

            # Follow the chain to its end
            while current in redirect_map:
                next_dest = redirect_map[current]

                # Detect circular redirects
                if next_dest in visited:
                    chain.append(next_dest)
                    chain.append("(CIRCULAR REDIRECT!)")
                    break

                chain.append(next_dest)
                visited.add(next_dest)
                current = next_dest

            chains.append(chain)

    return chains


def check_docs_slug_chains(redirects: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Check for redirects that chain through the /docs/:slug* pattern.

    The /docs/:slug* -> /:slug* redirect can create implicit chains.
    """
    docs_chains = []

    for r in redirects:
        source = r.get('source', '')
        destination = r.get('destination', '')

        # Skip the /docs/:slug* redirect itself
        if source == '/docs/:slug*':
            continue

        # Check if destination starts with /docs/ and is not external
        if destination.startswith('/docs/') and not destination.startswith('http'):
            docs_chains.append({
                'source': source,
                'intermediate': destination,
                'final': destination.replace('/docs/', '/', 1)
            })

    return docs_chains


def format_chain_output(chains: List[List[str]]) -> str:
    """Format chains for human-readable output."""
    output = []

    # Group chains by hop count
    chains_by_hops = {}
    for chain in chains:
        # Calculate hops (exclude circular marker if present)
        hops = len(chain) - 1
        if "(CIRCULAR REDIRECT!)" in chain:
            hops -= 1

        if hops not in chains_by_hops:
            chains_by_hops[hops] = []
        chains_by_hops[hops].append(chain)

    # Sort by hop count (longest first)
    for hops in sorted(chains_by_hops.keys(), reverse=True):
        chains_list = chains_by_hops[hops]
        output.append(f"\n{'='*80}")
        output.append(f"{len(chains_list)} chain(s) with {hops} hop(s):")
        output.append('='*80)

        for i, chain in enumerate(chains_list, 1):
            output.append(f"\n{i}.")
            for j, step in enumerate(chain):
                if step == "(CIRCULAR REDIRECT!)":
                    output.append(f"   {'   ' * j}⚠️  {step}")
                else:
                    indent = '   ' * j
                    arrow = '→' if j > 0 else ' '
                    output.append(f"   {indent}{arrow} {step}")

    return '\n'.join(output)


def main():
    """Main entry point for the script."""
    # Determine docs.json path
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    docs_json_path = repo_root / 'docs.json'

    # Allow path override via command line
    if len(sys.argv) > 1:
        docs_json_path = Path(sys.argv[1])

    print(f"Checking redirects in: {docs_json_path}")
    print()

    # Load and process redirects
    redirects = load_redirects(docs_json_path)
    print(f"Found {len(redirects)} total redirect(s)")

    redirect_map = build_redirect_map(redirects)
    chains = find_chained_redirects(redirect_map)
    docs_chains = check_docs_slug_chains(redirects)

    total_chains = len(chains) + len(docs_chains)

    if total_chains == 0:
        print("\n✅ No chained redirects found!")
        return 0

    # Report findings
    print(f"\n⚠️  Found {total_chains} chained redirect(s)!\n")

    if chains:
        print(f"Direct chains: {len(chains)}")
        print(format_chain_output(chains))

    if docs_chains:
        print(f"\n{'='*80}")
        print(f"Chains through /docs/:slug* pattern: {len(docs_chains)}")
        print('='*80)
        for i, chain in enumerate(docs_chains, 1):
            print(f"\n{i}.")
            print(f"    → {chain['source']}")
            print(f"       → {chain['intermediate']}")
            print(f"          → {chain['final']} (via /docs/:slug* redirect)")

    # Summary and recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print('='*80)
    print("\nChained redirects should be consolidated to point directly to the")
    print("final destination to improve performance and avoid potential issues.")
    print("\nFor each chain above, update the initial source to redirect directly")
    print("to the final destination, then remove intermediate redirect entries.")

    # Exit with error code to support CI/CD integration
    return 1


if __name__ == '__main__':
    sys.exit(main())

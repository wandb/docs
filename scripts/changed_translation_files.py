#!/usr/bin/env python3
"""
Script to find English content files that have changed since the last translation commit.
This helps identify which files need retranslation for Japanese and Korean.

Usage:
    python scripts/changed_translation_files.py --since_commit <commit_hash> --language <ja|ko>
    python scripts/changed_translation_files.py --since_commit abc1234 --language ja
    python scripts/changed_translation_files.py --list_recent_commits  # Helper to find commit hashes
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass

import simple_parsing as sp


@dataclass
class Config:
    """Configuration for finding changed translation files."""
    
    since_commit: Optional[str] = sp.field(
        default=None,
        help="Git commit hash of the last translation"
    )
    
    language: Optional[str] = sp.field(
        default=None,
        help="Target language (ja for Japanese, ko for Korean)",
        choices=["ja", "ko"]
    )
    
    list_recent_commits: bool = sp.field(
        default=False,
        help="List recent commits to help find translation commit hashes"
    )
    
    verbose: bool = sp.field(
        default=False,
        help="Show detailed information about each file"
    )
    
    output_format: str = sp.field(
        default="list",
        help="Output format",
        choices=["list", "json"]
    )


def run_git_command(cmd: List[str]) -> str:
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            cwd=Path(__file__).parent.parent  # Run from repo root
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def get_changed_files_since_commit(since_commit: str) -> List[str]:
    """Get list of files that have changed since the given commit."""
    cmd = ["git", "diff", "--name-only", f"{since_commit}..HEAD"]
    output = run_git_command(cmd)
    
    if not output:
        return []
    
    return [line.strip() for line in output.split('\n') if line.strip()]


def filter_english_content_files(files: List[str]) -> List[str]:
    """Filter to only include English content files that might need translation."""
    english_content_files = []
    
    for file in files:
        # Only include files in content/en/ directory
        if file.startswith('content/en/') and file.endswith('.md'):
            english_content_files.append(file)
    
    return english_content_files


def check_translation_exists(en_file: str, language: str) -> Tuple[bool, str]:
    """Check if a translation already exists for the given file."""
    # Convert content/en/path/file.md to content/{language}/path/file.md
    translated_file = en_file.replace('content/en/', f'content/{language}/')
    
    repo_root = Path(__file__).parent.parent
    translated_path = repo_root / translated_file
    
    return translated_path.exists(), translated_file


def get_file_status_since_commit(file_path: str, since_commit: str) -> str:
    """Get the git status of a file (added, modified, deleted, renamed)."""
    cmd = ["git", "diff", "--name-status", f"{since_commit}..HEAD", "--", file_path]
    output = run_git_command(cmd)
    
    if not output:
        return "unknown"
    
    # Output format is like "M\tfile_path" or "A\tfile_path"
    status_line = output.split('\t')[0]
    
    status_map = {
        'A': 'added',
        'M': 'modified', 
        'D': 'deleted',
        'R': 'renamed',
        'C': 'copied'
    }
    
    return status_map.get(status_line[0], status_line)


def list_recent_commits(count: int = 20) -> None:
    """List recent commits to help find translation commit hashes."""
    cmd = ["git", "log", "--oneline", f"-{count}"]
    output = run_git_command(cmd)
    
    print(f"Last {count} commits:")
    print("=" * 50)
    for line in output.split('\n'):
        if line.strip():
            print(line)


def main():
    config = sp.parse(Config)
    
    # Handle list recent commits
    if config.list_recent_commits:
        list_recent_commits()
        return
    
    # Validate required arguments
    if not config.since_commit or not config.language:
        print("Error: --since_commit and --language are required (unless using --list_recent_commits)")
        print("Use --help for usage information")
        sys.exit(1)
    
    print(f"Finding English files changed since commit: {config.since_commit}")
    print(f"Target language: {config.language}")
    print("=" * 60)
    
    # Get all changed files since the commit
    all_changed_files = get_changed_files_since_commit(config.since_commit)
    
    if not all_changed_files:
        print("No files have changed since the specified commit.")
        return
    
    # Filter to English content files
    english_files = filter_english_content_files(all_changed_files)
    
    if not english_files:
        print("No English content files have changed since the specified commit.")
        return
    
    print(f"Found {len(english_files)} English content files that have changed:")
    print()
    
    files_needing_translation = []
    files_with_existing_translation = []
    
    for en_file in english_files:
        file_status = get_file_status_since_commit(en_file, config.since_commit)
        translation_exists, translated_file = check_translation_exists(en_file, config.language)
        
        if config.verbose:
            print(f"📄 {en_file}")
            print(f"   Status: {file_status}")
            print(f"   Translation exists: {'✅' if translation_exists else '❌'}")
            if translation_exists:
                print(f"   Translation path: {translated_file}")
            print()
        
        if translation_exists:
            files_with_existing_translation.append({
                'english_file': en_file,
                'translated_file': translated_file,
                'status': file_status
            })
        else:
            files_needing_translation.append({
                'english_file': en_file,
                'status': file_status
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if files_needing_translation:
        print(f"\n🔴 Files needing NEW translation ({len(files_needing_translation)}):")
        for file_info in files_needing_translation:
            print(f"   {file_info['english_file']} ({file_info['status']})")
    
    if files_with_existing_translation:
        print(f"\n🟡 Files needing RE-translation ({len(files_with_existing_translation)}):")
        for file_info in files_with_existing_translation:
            print(f"   {file_info['english_file']} ({file_info['status']})")
            if config.verbose:
                print(f"      → {file_info['translated_file']}")
    
    # Output in different formats
    if config.output_format == "json":
        import json
        result = {
            'since_commit': config.since_commit,
            'language': config.language,
            'files_needing_new_translation': files_needing_translation,
            'files_needing_retranslation': files_with_existing_translation
        }
        print(f"\n{json.dumps(result, indent=2)}")
    
    # Final statistics
    total_changed = len(english_files)
    total_new_translation = len(files_needing_translation)
    total_retranslation = len(files_with_existing_translation)
    total_requiring_attention = total_new_translation + total_retranslation
    
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"📊 Total English files changed since commit:     {total_changed}")
    print(f"🔴 Files needing NEW translation:               {total_new_translation}")
    print(f"🟡 Files needing RE-translation:                {total_retranslation}")
    print(f"⚠️  Total files requiring attention:            {total_requiring_attention}")
    
    if total_changed > 0:
        print(f"\n📈 Translation coverage: {total_retranslation}/{total_changed} files already have translations ({total_retranslation/total_changed*100:.1f}%)")
    
    if total_requiring_attention == 0:
        print("🎉 All files are up to date! No translation work needed.")


if __name__ == '__main__':
    main()

---
description: Reference guide for keyboard shortcuts available in the W&B App UI
menu:
  default:
    identifier: keyboard-shortcuts
    parent: app
title: Keyboard shortcuts
weight: 60
---

The W&B App UI supports keyboard shortcuts to help you navigate and interact with experiments, workspaces, and data more efficiently. This reference guide covers all available keyboard shortcuts organized by functional area.

## Workspace Management

| Shortcut | Action | Context |
|----------|--------|---------|
| **Cmd+Z** (macOS) / **Ctrl+Z** (Windows/Linux) | Undo workspace changes | Workspace panels |

Use this shortcut to quickly undo changes you've made to your workspace layout, panel configurations, or other workspace modifications.

## Run Management

| Shortcut | Action | Context |
|----------|--------|---------|
| **Ctrl+D** | Stop/kill running scripts | Terminal (all platforms) |

Press `Ctrl+D` in your terminal or command line interface to stop a script that's instrumented with W&B. This shortcut works consistently across macOS, Windows, and Linux.

## Navigation

| Shortcut | Action | Context |
|----------|--------|---------|
| **Left/Right arrows** | Navigate between panels | Full-screen mode |
| **Command+Left/Right** (macOS) / **Ctrl+Left/Right** (Windows/Linux) | Move step slider | Full-screen mode |

When viewing a panel in full-screen mode, use these shortcuts to efficiently navigate through your data:
- **Arrow keys** let you step through panels without first clicking on the step slider
- **Command/Ctrl + arrows** move the step slider to navigate through different steps or time periods in your data

## Content Management

| Shortcut | Action | Context |
|----------|--------|---------|
| **Delete** or **Backspace** | Delete panel grids | Reports |
| **Enter** | Insert Markdown block | Reports (after typing "/mark") |

### Reports and Documentation
- **Delete/Backspace**: Select a panel grid by clicking its drag handle in the top-right corner, then press Delete or Backspace to remove it
- **Markdown blocks**: Type "/mark" anywhere in a report document and press Enter to insert a Markdown block for rich text editing

## Standard Navigation

| Shortcut | Action | Context |
|----------|--------|---------|
| **Tab** | Navigate between elements | Standard accessibility navigation |
| **Enter** | Submit searches/forms | Query panels, search fields |

These shortcuts follow standard web accessibility patterns and work throughout the W&B App UI for consistent navigation.

## Tips for Efficient Use

- **Full-screen mode shortcuts**: The Left/Right arrow and Command/Ctrl+Left/Right shortcuts are particularly useful when analyzing media panels, time series data, or stepping through experimental results
- **Workspace organization**: Use Cmd/Ctrl+Z frequently when experimenting with different workspace layouts to quickly revert unwanted changes
- **Cross-platform consistency**: Most shortcuts work identically across platforms, with the main difference being Cmd (macOS) vs Ctrl (Windows/Linux) for modifier keys

## Browser Compatibility

These keyboard shortcuts work in all modern web browsers that support the W&B App UI. Some shortcuts may interact with browser-specific features, so if you experience conflicts, check your browser's keyboard shortcut settings. 
---
title: How should I run sweeps on SLURM?
menu:
  support:
    identifier: ja-support-kb-articles-run_sweeps_slurm
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

[SLURM スケジューリングシステム](https://slurm.schedmd.com/documentation.html) で Sweeps を使用する場合、スケジュールされた各ジョブで `wandb agent --count 1 SWEEP_ID` を実行します。このコマンドは、単一のトレーニングジョブを実行して終了し、ハイパーパラメーター検索の並列処理を活用しながら、リソース要求のランタイム予測を容易にします。

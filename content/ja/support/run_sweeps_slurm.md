---
title: How should I run sweeps on SLURM?
menu:
  support:
    identifier: ja-support-run_sweeps_slurm
tags:
- sweeps
toc_hide: true
type: docs
---

スイープを [SLURM scheduling system](https://slurm.schedmd.com/documentation.html) で使用する場合、各スケジュールされたジョブで `wandb agent --count 1 SWEEP_ID` コマンドを実行します。このコマンドは単一のトレーニングジョブを実行し、その後終了します。これにより、ハイパーパラメーター検索の並列性を活用しながら、リソース要求のランタイムの予測を容易にします。
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

[SLURM scheduling system](https://slurm.schedmd.com/documentation.html) で Sweeps を使用する場合、スケジュールされた各ジョブで `wandb agent --count 1 SWEEP_ID` を実行します。この コマンド は、単一の トレーニング ジョブを実行して終了し、ハイパー パラメーター 探索の並列処理を活用しながら、リソース要求の ランタイム 予測を容易にします。

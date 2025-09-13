---
title: SLURM 上で Sweeps をどのように実行すればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-run_sweeps_slurm
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

Sweeps を [SLURM スケジューリング システム](https://slurm.schedmd.com/documentation.html) と併用する場合は、各スケジュール済みジョブで `wandb agent --count 1 SWEEP_ID` を実行してください。このコマンドは単一のトレーニング ジョブを実行して終了し、ハイパーパラメーター探索の並列性を活用しつつ、リソース要求に対する実行時間の予測を容易にします。
---
title: SLURM でスイープを実行するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-run_sweeps_slurm
support:
- スイープ
toc_hide: true
type: docs
url: /support/:filename
---

[SLURM スケジューリングシステム](https://slurm.schedmd.com/documentation.html) で Sweeps を使用する場合は、スケジューリングされた各ジョブ内で `wandb agent --count 1 SWEEP_ID` を実行してください。このコマンドは 1 回のトレーニングジョブを実行して終了し、リソース要求のランタイム予測を容易にしつつ、ハイパーパラメーター探索の並列化も活用できます。
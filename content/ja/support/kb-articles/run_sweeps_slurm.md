---
title: SLURM で sweeps を実行するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-run_sweeps_slurm
support:
  - sweeps
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: run_sweeps_slurm
---
スイープを [SLURM スケジューリングシステム](https://slurm.schedmd.com/documentation.html) で使用する場合、各スケジュールされたジョブで `wandb agent --count 1 SWEEP_ID` を実行します。このコマンドは、1つのトレーニングジョブを実行してから終了し、ハイパーパラメーター探索の並列性を活用しながら、リソース要求のランタイム予測を容易にします。
---
title: SLURM で Sweeps を実行するにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
---

SLURM スケジューリングシステム（[公式ドキュメント](https://slurm.schedmd.com/documentation.html)）で Sweeps を使用する場合、スケジューリングされた各ジョブで `wandb agent --count 1 SWEEP_ID` を実行してください。このコマンドは 1 回のトレーニングジョブを実行して終了するため、リソースリクエストの実行時予測を容易にしつつ、ハイパーパラメーター探索の並列性を活用できます。
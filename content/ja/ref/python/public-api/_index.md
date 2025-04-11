---
title: インポート＆エクスポート API
menu:
  reference:
    identifier: ja-ref-python-public-api-_index
---

## クラス

[`class Api`](./api.md): wandb サーバーをクエリするために使用されます。

[`class File`](./file.md): File は wandb によって保存されたファイルに関連付けられているクラスです。

[`class Files`](./files.md): `File` オブジェクトの反復可能なコレクション。

[`class Job`](./job.md)

[`class Project`](./project.md): プロジェクトは、run の名前空間です。

[`class Projects`](./projects.md): `Project` オブジェクトの反復可能なコレクション。

[`class QueuedRun`](./queuedrun.md): entity と project に関連付けられた単一のキューされた run。`run = queued_run.wait_until_running()` または `run = queued_run.wait_until_finished()` を呼び出して run にアクセスします。

[`class Run`](./run.md): entity と project に関連付けられた単一の run。

[`class RunQueue`](./runqueue.md)

[`class Runs`](./runs.md): プロジェクトとオプションのフィルタに関連付けられたいくつかの run の反復可能なコレクション。

[`class Sweep`](./sweep.md): sweep に関連付けられた一連の runs。
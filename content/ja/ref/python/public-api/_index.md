---
title: Import & Export API
menu:
  reference:
    identifier: ja-ref-python-public-api-_index
---

## クラス

[`class Api`](./api.md): wandb サーバー のクエリに使用されます。

[`class File`](./file.md): File は wandb で保存されたファイルに関連付けられたクラスです。

[`class Files`](./files.md): `File` オブジェクトの反復可能なコレクション。

[`class Job`](./job.md)

[`class Project`](./project.md): Project は Runs の名前空間です。

[`class Projects`](./projects.md): `Project` オブジェクトの反復可能なコレクション。

[`class QueuedRun`](./queuedrun.md): エンティティと project に関連付けられた単一のキューされた run。`run = queued_run.wait_until_running()` または `run = queued_run.wait_until_finished()` を呼び出して、run にアクセスします。

[`class Run`](./run.md): エンティティと project に関連付けられた単一の run。

[`class RunQueue`](./runqueue.md)

[`class Runs`](./runs.md): project に関連付けられた runs の反復可能なコレクションとオプションのフィルター。

[`class Sweep`](./sweep.md): sweep に関連付けられた runs のセット。

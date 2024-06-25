
# Import & Export API

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

## クラス

[`class Api`](./api.md): wandb サーバーをクエリするために使用されます。

[`class File`](./file.md): Fileはwandbによって保存されたファイルに関連するクラスです。

[`class Files`](./files.md): `File` オブジェクトの反復可能なコレクション。

[`class Job`](./job.md)

[`class Project`](./project.md): Project は runs の名前空間です。

[`class Projects`](./projects.md): `Project` オブジェクトの反復可能なコレクション。

[`class QueuedRun`](./queuedrun.md): エンティティとプロジェクトに関連する単一のキュー済みrun。 `run = queued_run.wait_until_running()` または `run = queued_run.wait_until_finished()` を呼び出して run にアクセスします。

[`class Run`](./run.md): エンティティとプロジェクトに関連する単一の run。

[`class RunQueue`](./runqueue.md)

[`class Runs`](./runs.md): プロジェクトおよびオプションのフィルタに関連する runs の反復可能なコレクション。

[`class Sweep`](./sweep.md): sweep に関連する runs のセット。

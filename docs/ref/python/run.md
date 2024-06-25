
# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L461-L4183' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandbによって記録される計算の単位。通常はML実験です。

```python
Run(
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    sweep_config: Optional[Dict[str, Any]] = None,
    launch_config: Optional[Dict[str, Any]] = None
) -> None
```

`wandb.init()`でrunを作成します:

```python
import wandb

run = wandb.init()
```

どのプロセスでも一度にアクティブな`wandb.Run`は最大1つのみであり、`wandb.run`としてアクセス可能です:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

`wandb.log`でログするものはすべてそのrunに送信されます。

同じスクリプトやノートブックで複数のrunを開始したい場合、現在進行中のrunを終了する必要があります。runは`wandb.finish`を使うか`with`ブロックで使用して終了することができます:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # ここでデータをログします

assert wandb.run is None
```

runの作成に関する詳細は`wandb.init`のドキュメントをご覧ください。または[こちらのガイド](https://docs.wandb.ai/guides/track/launch)をご覧ください。

分散トレーニングでは、ランク0プロセスで単一のrunを作成し、そのプロセスからのみ情報をログする方法と、各プロセスでrunを作成し、各プロセスから別々にログを取る方法があります。結果をまとめるには、`wandb.init`の`group`引数を使用します。W&Bを使用した分散トレーニングの詳細は[こちらのガイド](https://docs.wandb.ai/guides/track/log/distributed-training)をご覧ください。

現在、`wandb.Api`には並行する`Run`オブジェクトがありますが、最終的にはこれらのオブジェクトは統合される予定です。

| 属性 |  |
| :--- | :--- |
|  `summary` |  (Summary) 各`wandb.log()`キーに設定された単一の値。デフォルトでは、summaryは最後にログされた値に設定されます。summaryを手動で最高値に設定することもできます。例えば、最終値ではなく最高精度などです。 |
|  `config` |  このrunに関連付けられたConfigオブジェクト。 |
|  `dir` |  runに関連するファイルが保存されるディレクトリー。 |
|  `entity` |  runに関連付けられたW&Bエンティティの名前。エンティティはユーザー名やチームや組織の名前になります。 |
|  `group` |  runに関連付けられたグループ名。グループを設定すると、W&B UIが合理的な方法でrunを整理します。分散トレーニングを行う場合は、トレーニングのすべてのrunに同じグループ名を付けるべきです。交差検証を行う場合は、すべての交差検証フォールドに同じグループ名を付けるべきです。 |
|  `id` |  このrunの識別子。 |
|  `mode` |  `0.9.x`およびそれ以前との互換性のためであり、最終的には廃止されます。 |
|  `name` |  runの表示名。表示名は一意であることは保証されず、説明的である場合があります。デフォルトではランダムに生成されます。 |
|  `notes` |  runに関連付けられたメモ（存在する場合）。メモは複数行の文字列であり、マークダウンや`$$`内のlatex数式を使用することもできます。例: `$x + 3$`。 |
|  `path` |  runへのパス。runパスにはエンティティ、プロジェクト、およびrun IDが含まれ、形式は`entity/project/run_id`です。 |
|  `project` |  runに関連付けられたW&Bプロジェクトの名前。 |
|  `resumed` |  runが再開された場合はTrue、それ以外の場合はFalse。 |
|  `settings` |  runのSettingsオブジェクトのフローズンコピー。 |
|  `start_time` |  runが開始された時刻のUnixタイムスタンプ（秒）です。 |
|  `starting_step` |  runの最初のステップ。 |
|  `step` |  ステップの現在値。このカウンターは`wandb.log`によってインクリメントされます。 |
|  `sweep_id` |  runに関連付けられたsweepのID（存在する場合）。 |
|  `tags` |  runに関連付けられたタグ（存在する場合）。 |
|  `url` |  runに関連付けられたW&BのURL。 |

## メソッド

### `alert`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3480-L3513)

```python
alert(
    title: str,
    text: str,
    level: Optional[Union[str, 'AlertLevel']] = None,
    wait_duration: Union[int, float, timedelta, None] = None
) -> None
```

指定されたタイトルとテキストでアラートを起動します。

| 引数 |  |
| :--- | :--- |
|  `title` |  (str) アラートのタイトル。64文字未満である必要があります。 |
|  `text` |  (str) アラートの本文。 |
|  `level` |  (strまたはwandb.AlertLevel、オプション) 使用するアラートレベル。`INFO`、`WARN`、または`ERROR`のいずれかです。 |
|  `wait_duration` |  (int, float、またはtimedelta、オプション) このタイトルで別のアラートを送信する前に待つ時間（秒）。 |

### `define_metric`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2681-L2715)

```python
define_metric(
    name: str,
    step_metric: Union[str, wandb_metric.Metric, None] = None,
    step_sync: Optional[bool] = None,
    hidden: Optional[bool] = None,
    summary: Optional[str] = None,
    goal: Optional[str] = None,
    overwrite: Optional[bool] = None,
    **kwargs
) -> wandb_metric.Metric
```

後で`wandb.log()`を使用してログされるメトリクスのプロパティを定義します。

| 引数 |  |
| :--- | :--- |
|  `name` |  メトリクスの名前。 |
|  `step_metric` |  メトリクスに関連付けられた独立変数。 |
|  `step_sync` |  必要に応じて`step_metric`を自動的に履歴に追加。step_metricが指定されている場合はデフォルトでTrueです。 |
|  `hidden` |  このメトリクスを自動プロットから非表示にします。 |
|  `summary` |  summaryに追加される集計メトリクスを指定します。サポートされている集計方法は次の通り: "min,max,mean,best,last,none"。デフォルトの集計方法は `copy` で、集計 `best` のデフォルトは `goal`==`minimize` です。 |
|  `goal` |  メトリクスの最適化方向を指定します。サポートされている方向は次の通り: "minimize,maximize"。 |

| 戻り値 |  |
| :--- | :--- |
|  さらに指定できるメトリクスオブジェクトを返します。 |

### `detach`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2848-L2849)

```python
detach() -> None
```

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1349-L1357)

```python
display(
    height: int = 420,
    hidden: bool = (False)
) -> bool
```

このrunをjupyterで表示します。

### `finish`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2086-L2100)

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

runを終了としてマークし、すべてのデータをアップロードし終えます。

これは同じプロセス内で複数のrunを作成する場合に使用されます。スクリプトが終了したときやrunコンテキストマネージャを使用した場合に、このメソッドが自動的に呼び出されます。

| 引数 |  |
| :--- | :--- |
|  `exit_code` |  0以外の値を設定するとrunが失敗したとマークされます |
|  `quiet` |  ログ出力を最小限に抑える場合はtrueを設定します |

### `finish_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3098-L3150)

```python
finish_artifact(
    artifact_or_path: Union[Artifact, str],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    distributed_id: Optional[str] = None
) -> Artifact
```

未確定のアーティファクトをrunの出力として終了します。

同じ分散IDを持つ後続のアップデートは新しいバージョンとなります。

| 引数 |  |
| :--- | :--- |
|  `artifact_or_path` |  (strまたはArtifact) このアーティファクトの内容へのパス。次の形式で指定できます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` `wandb.Artifact`を呼び出して作成されたArtifactオブジェクトを渡すこともできます。 |
|  `name` |  (str, オプション) アーティファクト名。エンティティ/プロジェクトであらかじめ指定されることがあります。次の形式が有効です: - name:version - name:alias - digest 指定されない場合、デフォルトでパスのベース名と現在のrun IDが先行されます。 |
|  `type` |  (str) ログするアーティファクトのタイプ。例: `dataset`, `model` |
|  `aliases` |  (list, オプション) このアーティファクトに適用されるエイリアス。デフォルトは `["latest"]` です。 |
|  `distributed_id` |  (string, オプション) すべての分散ジョブが共有する一意の文字列。指定がない場合、デフォルトでrunのグループ名となります。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact`オブジェクト。 |

### `get_project_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1231-L1239)

```python
get_project_url() -> Optional[str]
```

runに関連付けられたW&BプロジェクトのURLを返します（存在する場合）。

オフラインrunにはプロジェクトURLはありません。

### `get_sweep_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1241-L1246)

```python
get_sweep_url() -> Optional[str]
```

runに関連付けられたsweepのURLを返します（存在する場合）。

### `get_url`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L1221-L1229)

```python
get_url() -> Optional[str]
```

W&B runのURLを返します（存在する場合）。

オフラインrunにはURLはありません。

### `join`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2134-L2144)

```python
join(
    exit_code: Optional[int] = None
) -> None
```

`finish()`の古いエイリアス - 代わりにfinishを使用してください。

### `link_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L2851-L2897)

```python
link_artifact(
    artifact: Artifact,
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

指定されたアーティファクトをポートフォリオにリンクします（アーティファクトの昇格コレクション）。

リンクされたアーティファクトは、指定されたポートフォリオのUIで表示されます。

| 引数 |  |
| :--- | :--- |
|  `artifact` |  リンクされる（公開またはローカルの）アーティファクト |
|  `target_path` |  `str` - 次の形式: {portfolio}, {project}/{portfolio}, または {entity}/{project}/{portfolio} |
|  `aliases` |  `List[str]` - このリンクされたアーティファクト内でのみ適用されるエイリアス（オプション）。エイリアス "latest" はリンクされたアーティファクトの最新バージョンに常に適用されます。 |

| 戻り値 |  |
| :--- | :--- |
|  なし |

### `link_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_run.py#L3386-L3478)

```python
link_model(
    path: StrPath,
    registered_model_name: str,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None
) -> None
```

モデルのアーティファクトバージョンを記録し、モデルレジストリ内の登録されたモデルにリンクします。

リンクされたモデルバージョンは、指定された登録モデルのUIで表示されます。

#### ステップ:

- 'name' モデルアーティファクトがログされているか確認します。ログされている場合、ファイルのある場所に一致するアーティファクトバージョンを使用するか、新しいバージョンをログします。そうでない場合、パスにあるファイルを新しいモデルアーティファクト 'name' としてログし、タイプ 'model' とします。
- モデルレジストリプロジェクト内に 'registered_model_name' という名前の登録モデルが存在するか確認します。存在しない場合、新しい登録モデル 'registered_model_name' を作成します。
- 'name' のモデルアーティファクトバージョンを 'registered_model_name' にリンクします。
- 'aliases' リストからのエイリアスを新しくリンクされたモデルアーティファクトバージョンに付加します。

| 引数 |  |
| :--- | :--- |
|  `path` |  (str) このモデルの内容へのパス。次の形式で指定できます: - `/local/directory` - `/local/directory/file.txt` - `s3://bucket/path` |
|  `registered_model_name` |  (str) - モデルをリンクする登録モデルの名前。登録モデルはモデルレジストリにリンクされたモデルバージョンのコレクションであり、通常はチームの特定のMLタスクを表します。この登録モデルが属するエンティティはrun名から派生します。 |
|  `aliases` |  (List[str], オプション) - この
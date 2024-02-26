# プロジェクト

## チェイン可能なOps
<h3 id="project-artifact"><code>project-artifact</code></h3>

指定された名前の[アーティファクト](https://docs.wandb.ai/ref/weave/artifact)を、[プロジェクト](https://docs.wandb.ai/ref/weave/project)内で返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |
| `artifactName` | [アーティファクト](https://docs.wandb.ai/ref/weave/artifact)の名前 |

#### 返り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)内の指定された名前の[アーティファクト](https://docs.wandb.ai/ref/weave/artifact)

<h3 id="project-artifactType"><code>project-artifactType</code></h3>

指定された名前の[artifactType](https://docs.wandb.ai/ref/weave/artifact-type)を、[プロジェクト](https://docs.wandb.ai/ref/weave/project)内で返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |
| `artifactType` | [artifactType](https://docs.wandb.ai/ref/weave/artifact-type)の名前 |

#### 返り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)内の指定された名前の[artifactType](https://docs.wandb.ai/ref/weave/artifact-type)

<h3 id="project-artifactTypes"><code>project-artifactTypes</code></h3>

[プロジェクト](https://docs.wandb.ai/ref/weave/project)の[artifactTypes](https://docs.wandb.ai/ref/weave/artifact-type)を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |

#### 返り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)の[artifactTypes](https://docs.wandb.ai/ref/weave/artifact-type)

<h3 id="project-artifactVersion"><code>project-artifactVersion</code></h3>

指定された名前とバージョンの[artifactVersion](https://docs.wandb.ai/ref/weave/artifact-version)を、[プロジェクト](https://docs.wandb.ai/ref/weave/project)内で返します。
| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |
| `artifactName` | [artifactVersion](https://docs.wandb.ai/ref/weave/artifact-version)の名前 |
| `artifactVersionAlias` | [artifactVersion](https://docs.wandb.ai/ref/weave/artifact-version)のエイリアスバージョン |

#### 返り値
指定された名前とバージョンのある[プロジェクト](https://docs.wandb.ai/ref/weave/project)内の[artifactVersion](https://docs.wandb.ai/ref/weave/artifact-version)

<h3 id="project-createdAt"><code>project-createdAt</code></h3>

[プロジェクト](https://docs.wandb.ai/ref/weave/project)の作成時間を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |

#### 返り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)の作成時間

<h3 id="project-name"><code>project-name</code></h3>

[プロジェクト](https://docs.wandb.ai/ref/weave/project)の名前を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |

#### 返り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)の名前

<h3 id="project-runs"><code>project-runs</code></h3>

[プロジェクト](https://docs.wandb.ai/ref/weave/project)からの[runs](https://docs.wandb.ai/ref/weave/run)を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |
#### 戻り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)からの[runs](https://docs.wandb.ai/ref/weave/run)

## リスト操作
<h3 id="project-artifact"><code>project-artifact</code></h3>

指定された名前の[プロジェクト](https://docs.wandb.ai/ref/weave/project)内の[アーティファクト](https://docs.wandb.ai/ref/weave/artifact)を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |
| `artifactName` | [アーティファクト](https://docs.wandb.ai/ref/weave/artifact)の名前 |

#### 戻り値
指定された名前の[プロジェクト](https://docs.wandb.ai/ref/weave/project)内の[アーティファクト](https://docs.wandb.ai/ref/weave/artifact)

<h3 id="project-artifactType"><code>project-artifactType</code></h3>

指定された名前の[プロジェクト](https://docs.wandb.ai/ref/weave/project)内の[artifactType](https://docs.wandb.ai/ref/weave/artifact-type)を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |
| `artifactType` | [artifactType](https://docs.wandb.ai/ref/weave/artifact-type)の名前 |

#### 戻り値
指定された名前の[プロジェクト](https://docs.wandb.ai/ref/weave/project)内の[artifactType](https://docs.wandb.ai/ref/weave/artifact-type)

<h3 id="project-artifactTypes"><code>project-artifactTypes</code></h3>

[プロジェクト](https://docs.wandb.ai/ref/weave/project)の[artifactTypes](https://docs.wandb.ai/ref/weave/artifact-type)を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |

#### 戻り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)の[artifactTypes](https://docs.wandb.ai/ref/weave/artifact-type)

<h3 id="project-artifactVersion"><code>project-artifactVersion</code></h3>
指定された名前とバージョンに対応する[artifactVersion](https://docs.wandb.ai/ref/weave/artifact-version)を[プロジェクト](https://docs.wandb.ai/ref/weave/project)内で返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |
| `artifactName` | [artifactVersion](https://docs.wandb.ai/ref/weave/artifact-version)の名前 |
| `artifactVersionAlias` | [artifactVersion](https://docs.wandb.ai/ref/weave/artifact-version)のバージョンエイリアス |

#### 返り値
指定された名前とバージョンに対応する[artifactVersion](https://docs.wandb.ai/ref/weave/artifact-version)を[プロジェクト](https://docs.wandb.ai/ref/weave/project)内で返します。

<h3 id="project-createdAt"><code>project-createdAt</code></h3>

[プロジェクト](https://docs.wandb.ai/ref/weave/project)の作成時刻を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |

#### 返り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)の作成時刻

<h3 id="project-name"><code>project-name</code></h3>

[プロジェクト](https://docs.wandb.ai/ref/weave/project)の名前を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |

#### 返り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)の名前

<h3 id="project-runs"><code>project-runs</code></h3>

[プロジェクト](https://docs.wandb.ai/ref/weave/project)から[runs](https://docs.wandb.ai/ref/weave/run)を返します。

| 引数 |  |
| :--- | :--- |
| `project` | [プロジェクト](https://docs.wandb.ai/ref/weave/project) |
#### 返り値
[プロジェクト](https://docs.wandb.ai/ref/weave/project)からの[runs](https://docs.wandb.ai/ref/weave/run)
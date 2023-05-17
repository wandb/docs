# WandbRun.Builder

### 概要

Builderパターンを使用すると、WandbRunを設定するための読みやすいコードを書くことができます。ビルダーには、これらの値を初期化するために使用されるいくつかの関数が含まれています。

* **builder.build()** — runを表すWandbRunインスタンスを返します
* **builder.withName(String name)** — このrunの表示名で、UIに表示され編集可能で、一意である必要はありません
* **builder.withConfig(JSONObject data)** — 初期設定値が含まれるJava JSONオブジェクト
* **builder.withProject(String project)** — このrunが所属するプロジェクトの名前
* **builder.withNotes(String notes)** — runに関連付けられた説明
* **builder.setTags(List String tags)** — runで使用されるタグの配列
* **builder.setJobType(String type)** — ログしているジョブのタイプ（例：eval、worker、ps（デフォルト：training））
* **builder.withGroup(String group)** — 他のrunとグループ化するための文字列；[Grouping](../../guides/runs/grouping.md)を参照してください。

これらの設定のほとんどは、[環境変数](../../guides/track/environment-variables.md)を介して制御することもできます。これは、クラスター上でジョブを実行している場合に便利です。

### 例

デフォルトのrunを初期化する

```java
WandbRun run = new WandbRun.Builder().build();
```

configオブジェクトと名前を使ってrunを初期化する

```java
// JSONObject configを作成
JSONObject config = new JSONOBject();
config.add("property", true);

// ビルダーを使用してrunのオプションをカスタマイズ

WandbRun run = new WandbRun.Builder()

    .withConfig(config)

    .withName("A Java Run")

    .build();

```
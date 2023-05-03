# WandbRun

### 概要

実行は、[WandbRun Builder](wandbrun-builder.md)を使用して作成できます。このオブジェクトは、runs のトラッキングに使用されます。

* **run.log(JSONObject data)** — 実行のデータをログする。[wand.log()](../../guides/track/log/intro.md)と同等。
* **run.log(int step, JSONObject data)** — 特定のステップで実行のデータをログする。[wand.log()](../../guides/track/log/intro.md)と同等。
* **run.finish(int exitCode)** — exitCode（デフォルト: 0）を持つ実行を終了する。
### 例

Javaクライアントでsin波をプロットする

```java
// runを初期化する
WandbRun run = new WandbRun.Builder().build();
// 各sin値を計算し、ログに記録する
for (double i = 0.0; i < 2 * Math.PI; i += 0.1) {
    JSONObject data = new JSONObject();
    data.put("value", Math.sin(i));
    run.log(data);
}
以下は翻訳していただきたいMarkdownのテキストです。日本語に翻訳してください。それ以外のことは何も言わずに、翻訳されたテキストだけを返してください。テキスト:

// 完了時にrunを終了する。
run.done();
```
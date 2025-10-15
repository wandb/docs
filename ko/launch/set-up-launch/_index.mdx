---
title: JavaScript ライブラリ
description: TypeScript、Node、最新のWebブラウザー向けの W&B SDK
menu:
  reference:
    identifier: ja-ref-js-_index
---

似たように、Pythonライブラリと同様に、JavaScript/TypeScriptにおいても実験管理をトラッキングするためのクライアントを提供しています。

- Nodeサーバーからメトリクスをログし、それをW&B上のインタラクティブなプロットで表示
- インタラクティブなトレースを用いてLLMアプリケーションのデバッグ
- [LangChain.js](https://github.com/hwchase17/langchainjs) の使用デバッグ

このライブラリはNodeおよびモダンなJSランタイムに対応しています。

JavaScriptクライアントのソースコードは[Githubリポジトリ](https://github.com/wandb/wandb-js)で見つけることができます。

{{% alert %}}
私たちのJavaScriptインテグレーションはまだBeta版です。問題が発生した場合は知らせてください。
{{% /alert %}}

### インストール

```shell
npm install @wandb/sdk
# あるいは ...
yarn add @wandb/sdk
```

### 使用法

TypeScript/ESM:

```typescript
import wandb from '@wandb/sdk'

async function track() {
    await wandb.init({config: {test: 1}});
    wandb.log({acc: 0.9, loss: 0.1});
    wandb.log({acc: 0.91, loss: 0.09});
    await wandb.finish();
}

await track()
```

{{% alert color="secondary" %}}
全てのAPIコールを非同期で処理するために、別のMessageChannelを生成します。これにより、`await wandb.finish()`を呼ばないとスクリプトが停止します。
{{% /alert %}}

Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

現在、Python SDKで見つかる多くの機能が不足していますが、基本的なログ機能は利用可能です。[Tables]({{< relref path="/guides/models/tables/?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme" lang="ja" >}})など、追加の機能をすぐに追加予定です。

### 認証と設定

Node環境では`process.env.WANDB_API_KEY`を探し、TTYがある場合は入力を促します。非Node環境では`sessionStorage.getItem("WANDB_API_KEY")`を探します。追加の設定は[こちら](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts)で確認できます。

# インテグレーション

私たちの[Pythonインテグレーション]({{< relref path="/guides/integrations/" lang="ja" >}})はコミュニティで広く利用されており、より多くのJavaScriptインテグレーションを構築し、LLMアプリビルダーが任意のツールを活用できるようにすることを希望しています。

もし追加のインテグレーションのリクエストがあれば、リクエストの詳細と共にissueを開くことをお勧めします。

## LangChain.js

このライブラリは、LLMアプリケーションを構築するための人気のライブラリである[LangChain.js](https://github.com/hwchase17/langchainjs) バージョン >= 0.0.75 に統合されています。

### 使用法

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// Langchainのワークロードを実行...
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

{{% alert color="secondary" %}}
全てのAPIコールを非同期で処理するために、別のMessageChannelを生成します。これにより、`await WandbTracer.finish()`を呼ばないとスクリプトが停止します。
{{% /alert %}}

より詳細な例については[こちらのテスト](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts)をご覧ください。
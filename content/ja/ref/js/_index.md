---
title: JavaScript Library
description: TypeScript、Node、および最新のWebブラウザ向けのW&B SDK
menu:
  reference:
    identifier: ja-ref-js-_index
---

Python ライブラリと同様に、JavaScript/TypeScript での Experiments を追跡するクライアントも提供しています。

- Node サーバーからメトリクスをログに記録し、W&B のインタラクティブなプロットで表示します
- インタラクティブな Traces で LLM アプリケーションをデバッグします
- [LangChain.js](https://github.com/hwchase17/langchainjs) の使用をデバッグします

このライブラリは、Node および最新の JS ランタイムと互換性があります。

JavaScript クライアントのソースコードは、[Github リポジトリ](https://github.com/wandb/wandb-js) にあります。

{{% alert %}}
JavaScript のインテグレーションはまだベータ版です。問題が発生した場合は、お知らせください。
{{% /alert %}}

### インストール

```shell
npm install @wandb/sdk
# or ...
yarn add @wandb/sdk
```

### 使用方法

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
すべての API 呼び出しを非同期で処理するために、個別の MessageChannel を生成します。`await wandb.finish()` を呼び出さないと、スクリプトがハングアップします。
{{% /alert %}}

Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

現在、Python SDK にある機能の多くが欠けていますが、基本的なログ機能は利用可能です。 [Tables]({{< relref path="/guides/models/tables/?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme" lang="ja" >}}) のような機能もまもなく追加する予定です。

### 認証と設定

Node 環境では、`process.env.WANDB_API_KEY` を探し、TTY がある場合はその入力を求めます。非 Node 環境では、`sessionStorage.getItem("WANDB_API_KEY")` を探します。追加の settings は[こちら](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts)にあります。

# インテグレーション

当社の [Python integrations]({{< relref path="/guides/integrations/" lang="ja" >}}) はコミュニティで広く使用されており、LLM アプリケーションビルダーがあらゆる tool を活用できるように、より多くの JavaScript integrations を構築したいと考えています。

追加の integrations のリクエストがある場合は、リクエストの詳細を記載した issue をオープンしてください。

## LangChain.js

このライブラリは、LLM アプリケーションを構築するための一般的なライブラリである [LangChain.js](https://github.com/hwchase17/langchainjs) バージョン >= 0.0.75 と統合されています。

### 使用方法

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// run your langchain workloads...
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

{{% alert color="secondary" %}}
すべての API 呼び出しを非同期で処理するために、別の MessageChannel を生成します。`await WandbTracer.finish()` を呼び出さないと、スクリプトがハングアップします。
{{% /alert %}}

詳細な例については、[このテスト](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts) を参照してください。

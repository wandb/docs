---
title: JavaScript Library
description: TypeScript, Node, 최신 웹 브라우저용 W&B SDK
menu:
  reference:
    identifier: ko-ref-js-_index
---

Python 라이브러리와 유사하게 JavaScript/TypeScript에서 Experiments를 추적할 수 있는 클라이언트를 제공합니다.

- Node 서버에서 메트릭을 기록하고 W&B의 대화형 플롯에 표시합니다.
- 대화형 추적으로 LLM 애플리케이션을 디버깅합니다.
- [LangChain.js](https://github.com/hwchase17/langchainjs) 사용을 디버깅합니다.

이 라이브러리는 Node 및 최신 JS 런타임과 호환됩니다.

JavaScript 클라이언트의 소스 코드는 [Github repository](https://github.com/wandb/wandb-js)에서 찾을 수 있습니다.

{{% alert %}}
JavaScript 통합은 아직 베타 버전이므로 문제가 발생하면 알려주세요.
{{% /alert %}}

### 설치

```shell
npm install @wandb/sdk
# or ...
yarn add @wandb/sdk
```

### 사용법

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
모든 API 호출을 비동기적으로 처리하기 위해 별도의 MessageChannel을 생성합니다. `await wandb.finish()`를 호출하지 않으면 스크립트가 멈추게 됩니다.
{{% /alert %}}

Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

현재 Python SDK에서 제공되는 많은 기능이 누락되었지만 기본 로깅 기능은 사용할 수 있습니다. 곧 [Tables]({{< relref path="/guides/models/tables/?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme" lang="ko" >}}) 와 같은 추가 기능을 추가할 예정입니다.

### 인증 및 설정

Node 환경에서는 `process.env.WANDB_API_KEY`를 찾고 TTY가 있는 경우 입력을 요청합니다. Non-Node 환경에서는 `sessionStorage.getItem("WANDB_API_KEY")`를 찾습니다. 추가 설정은 [여기](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts)에서 찾을 수 있습니다.

# 통합

[Python integrations]({{< relref path="/guides/integrations/" lang="ko" >}})는 우리 커뮤니티에서 널리 사용되고 있으며, LLM 앱 빌더가 원하는 툴을 활용할 수 있도록 더 많은 JavaScript integrations를 구축하기를 바랍니다.

추가 integrations에 대한 요청이 있으시면 요청에 대한 자세한 내용을 issue로 열어주시면 감사하겠습니다.

## LangChain.js

이 라이브러리는 LLM 애플리케이션을 구축하기 위한 널리 사용되는 라이브러리인 [LangChain.js](https://github.com/hwchase17/langchainjs) 버전 >= 0.0.75와 통합됩니다.

### 사용법

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// run your langchain workloads...
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

{{% alert color="secondary" %}}
모든 API 호출을 비동기적으로 처리하기 위해 별도의 MessageChannel을 생성합니다. `await WandbTracer.finish()`를 호출하지 않으면 스크립트가 멈추게 됩니다.
{{% /alert %}}

자세한 예는 [this test](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts)를 참조하십시오.

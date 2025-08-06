---
title: JavaScript 라이브러리
description: TypeScript, Node, 그리고 최신 웹 브라우저를 위한 W&B SDK
menu:
  reference:
    identifier: ko-ref-js-_index
---

Python 라이브러리와 마찬가지로, JavaScript/TypeScript 에서도 실험을 추적할 수 있는 클라이언트를 제공합니다.

- Node 서버에서 메트릭을 로그하고, W&B 의 인터랙티브 플롯에서 시각화
- 인터랙티브 트레이스로 LLM 애플리케이션 디버깅
- [LangChain.js](https://github.com/hwchase17/langchainjs) 사용 디버깅

이 라이브러리는 Node 및 최신 JS 런타임과 호환됩니다.

JavaScript 클라이언트의 소스 코드는 [Github 저장소](https://github.com/wandb/wandb-js)에서 확인할 수 있습니다.

{{% alert %}}
JavaScript 인테그레이션은 아직 베타 단계입니다. 문제가 있다면 꼭 알려주세요.
{{% /alert %}}

## 설치

```shell
npm install @wandb/sdk
# 또는 ...
yarn add @wandb/sdk
```

## 사용법

### TypeScript/ESM:

```typescript
import wandb from '@wandb/sdk'

// 실험 추적 함수 예시
async function track() {
    await wandb.init({config: {test: 1}});
    wandb.log({acc: 0.9, loss: 0.1});
    wandb.log({acc: 0.91, loss: 0.09});
    await wandb.finish();
}

await track()
```

{{% alert color="secondary" %}}
모든 api 호출을 비동기로 처리하기 위해 별도의 MessageChannel 을 생성합니다. `await wandb.finish()` 를 호출하지 않으면 스크립트가 종료되지 않을 수 있습니다.
{{% /alert %}}

### Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

현재는 Python SDK 에 있는 많은 기능들이 아직 구현되지 않았지만, 기본 로그 기능은 사용할 수 있습니다. [Tables]({{< relref path="/guides/models/tables/?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme" lang="ko" >}}) 와 같은 추가 기능도 곧 지원할 예정입니다.

## 인증 및 설정

Node 환경에서는 `process.env.WANDB_API_KEY` 를 먼저 확인하고, TTY 가 있으면 입력을 요청합니다. Node 환경이 아닐 경우에는 `sessionStorage.getItem("WANDB_API_KEY")` 를 확인합니다. 추가 설정은 [여기서 확인할 수 있습니다](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts).

## 인테그레이션

[Python 인테그레이션]({{< relref path="/guides/integrations/" lang="ko" >}}) 은 커뮤니티에서 폭넓게 사용되고 있으며, LLM 앱 개발자가 원하는 툴을 자유롭게 활용할 수 있도록 JavaScript 인테그레이션도 계속 확대해 나갈 계획입니다.

추가로 필요한 인테그레이션이 있다면, 요청 사항을 이슈로 남겨주시면 적극 반영하겠습니다.

## LangChain.js

이 라이브러리는 LLM 애플리케이션 구축을 위한 인기 라이브러리인 [LangChain.js](https://github.com/hwchase17/langchainjs) 버전 >= 0.0.75 와 연동할 수 있습니다.

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// langchain 작업을 실행하세요...
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

{{% alert color="secondary" %}}
모든 api 호출을 비동기로 처리하기 위해 별도의 MessageChannel 을 생성합니다. `await WandbTracer.finish()` 를 호출하지 않으면 스크립트가 종료되지 않을 수 있습니다.
{{% /alert %}}

더 자세한 예시는 [이 테스트 코드](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts) 를 참고하세요.
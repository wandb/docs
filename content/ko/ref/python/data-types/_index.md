---
title: Data Types
menu:
  reference:
    identifier: ko-ref-python-data-types-_index
---

이 모듈은 풍부하고 인터랙티브한 시각화 자료를 W&B에 로깅하기 위한 데이터 유형을 정의합니다.

데이터 유형에는 이미지, 오디오 및 비디오와 같은 일반적인 미디어 유형, 테이블 및 HTML과 같은 정보에 대한 유연한 컨테이너 등이 있습니다.

미디어 로깅에 대한 자세한 내용은 [가이드](https://docs.wandb.com/guides/track/log/media)를 참조하세요.

인터랙티브한 데이터셋 및 모델 분석을 위한 구조화된 데이터 로깅에 대한 자세한 내용은 [W&B Tables 가이드](https://docs.wandb.com/guides/models/tables/)를 참조하세요.

이러한 모든 특수 데이터 유형은 WBValue의 서브클래스입니다. 모든 데이터 유형은 JSON으로 직렬화됩니다. 이는 wandb가 오브젝트를 로컬에 저장하고 W&B 서버에 업로드하는 데 사용하는 형식이기 때문입니다.

## 클래스

[`class Audio`](./audio.md): 오디오 클립을 위한 Wandb 클래스입니다.

[`class BoundingBoxes2D`](./boundingboxes2d.md): W&B에 로깅하기 위해 2D 바운딩 박스 오버레이가 있는 이미지 포맷입니다.

[`class Graph`](./graph.md): 그래프를 위한 Wandb 클래스입니다.

[`class Histogram`](./histogram.md): 히스토그램을 위한 wandb 클래스입니다.

[`class Html`](./html.md): 임의의 html을 위한 Wandb 클래스입니다.

[`class Image`](./image.md): W&B에 로깅하기 위한 이미지 포맷입니다.

[`class ImageMask`](./imagemask.md): W&B에 로깅하기 위한 이미지 마스크 또는 오버레이 포맷입니다.

[`class Molecule`](./molecule.md): 3D 분자 데이터를 위한 Wandb 클래스입니다.

[`class Object3D`](./object3d.md): 3D 포인트 클라우드를 위한 Wandb 클래스입니다.

[`class Plotly`](./plotly.md): Plotly 플롯을 위한 Wandb 클래스입니다.

[`class Table`](./table.md): 테이블 형식의 데이터를 표시하고 분석하는 데 사용되는 Table 클래스입니다.

[`class Video`](./video.md): W&B에 로깅하기 위한 비디오 포맷입니다.

[`class WBTraceTree`](./wbtracetree.md): 추적 트리 데이터를 위한 미디어 오브젝트입니다.

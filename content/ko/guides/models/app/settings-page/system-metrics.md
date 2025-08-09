---
title: 시스템 메트릭
description: W&B에서 자동으로 로그되는 메트릭.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-system-metrics
    parent: settings
weight: 70
---

이 페이지에서는 W&B SDK에서 추적하는 시스템 메트릭에 대한 자세한 정보를 제공합니다.

{{% alert %}}
`wandb`는 15초마다 시스템 메트릭을 자동으로 기록합니다.
{{% /alert %}}

## CPU

### 프로세스 CPU 사용률 (CPU)
프로세스가 사용하는 CPU 사용률을, 사용 가능한 CPU 수로 정규화한 백분율입니다.

W&B는 이 메트릭에 `cpu` 태그를 지정합니다.

### 프로세스 CPU 스레드 수
프로세스에서 사용 중인 스레드 개수입니다.

W&B는 이 메트릭에 `proc.cpu.threads` 태그를 지정합니다.



## 디스크

기본적으로 `/` 경로에 대한 사용량 메트릭이 수집됩니다. 모니터링할 경로를 설정하려면 다음 설정을 사용하세요:

```python
run = wandb.init(
    settings=wandb.Settings(
        x_stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### 디스크 사용률 (%)
지정된 경로의 전체 시스템 디스크 사용률(%)를 나타냅니다.

W&B는 이 메트릭에 `disk.{path}.usagePercent` 태그를 지정합니다.

### 디스크 사용량
지정된 경로의 전체 시스템 디스크 사용량(GB)을 나타냅니다.
엑세스 가능한 경로를 샘플링하며, 각 경로의 디스크 사용량(GB)이 샘플에 추가됩니다.

W&B는 이 메트릭에 `disk.{path}.usageGB` 태그를 지정합니다.

### Disk In
전체 시스템 디스크 읽기량(MB)을 나타냅니다.
최초 샘플링 시 디스크 읽기 바이트가 기록되며, 이후 샘플들은 현재 읽기 바이트에서 초기 값을 뺀 차이를 계산합니다.

W&B는 이 메트릭에 `disk.in` 태그를 지정합니다.

### Disk Out
전체 시스템 디스크 쓰기량(MB)을 나타냅니다.
[Disk In]({{< relref path="#disk-in" lang="ko" >}})과 마찬가지로, 최초 샘플링 시 디스크 쓰기 바이트가 기록되고, 이후 샘플들은 현재 쓰기 바이트에서 초기 값을 뺀 차이를 계산합니다.

W&B는 이 메트릭에 `disk.out` 태그를 지정합니다.



## 메모리

### 프로세스 메모리 RSS
프로세스의 메모리 RSS(Resident Set Size)를 MB 단위로 나타냅니다. RSS는 프로세스가 실제 주기억장치(RAM)에 점유하고 있는 메모리의 양을 의미합니다.

W&B는 이 메트릭에 `proc.memory.rssMB` 태그를 지정합니다.

### 프로세스 메모리 사용률
프로세스의 메모리 사용량을 전체 사용 가능한 메모리에 대한 백분율로 나타냅니다.

W&B는 이 메트릭에 `proc.memory.percent` 태그를 지정합니다.

### 전체 메모리 사용률
전체 시스템 메모리 사용률을 전체 사용 가능한 메모리에 대한 백분율로 나타냅니다.

W&B는 이 메트릭에 `memory_percent` 태그를 지정합니다.

### 사용 가능한 메모리
시스템에서 사용 가능한 전체 메모리(MB)를 나타냅니다.

W&B는 이 메트릭에 `proc.memory.availableMB` 태그를 지정합니다.



## 네트워크

### Network Sent
네트워크를 통해 전송된 전체 바이트 수를 나타냅니다.
메트릭이 처음 초기화될 때 전송된 바이트가 기록되고, 이후 샘플에서 현재 바이트에서 초기 값을 뺀 차이가 계산됩니다.

W&B는 이 메트릭에 `network.sent` 태그를 지정합니다.

### Network Received

네트워크를 통해 수신된 전체 바이트 수를 나타냅니다.
[Network Sent]({{< relref path="#network-sent" lang="ko" >}})과 마찬가지로, 초기 수신 바이트가 기록되고, 이후 샘플에서 현재 값과 초기 값의 차이가 계산됩니다.

W&B는 이 메트릭에 `network.recv` 태그를 지정합니다.



## NVIDIA GPU

아래 설명된 메트릭 이외에도, 프로세스 또는 그 하위 프로세스가 특정 GPU를 사용할 때, W&B는 대응하는 메트릭을 `gpu.process.{gpu_index}.{metric_name}` 으로 캡처합니다.

### GPU 메모리 사용률
각 GPU의 메모리 사용률(%)을 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memory` 태그를 지정합니다.

### GPU 메모리 할당률
각 GPU의 전체 사용 가능 메모리에 대한 할당된 메모리의 백분율을 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 지정합니다.

### GPU 메모리 할당 바이트
각 GPU의 할당된 GPU 메모리를 바이트 단위로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocatedBytes` 태그를 지정합니다.

### GPU 사용률
각 GPU의 전체 사용률(%)을 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 지정합니다.

### GPU 온도
각 GPU의 온도를 섭씨 단위로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 지정합니다.

### GPU 전력 사용량 (Watt)
각 GPU의 전력 사용량을 Watt 단위로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 지정합니다.

### GPU 전력 사용량 퍼센트

각 GPU의 전력 용량 대비 사용률(%)을 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 지정합니다.

### GPU SM 클럭 속도
GPU의 Streaming Multiprocessor (SM)의 클럭 속도(MHz)를 나타냅니다. 이 메트릭은 GPU 코어 내부의 연산 처리 속도를 의미합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.smClock` 태그를 지정합니다.

### GPU 메모리 클럭 속도
GPU 메모리의 클럭 속도(MHz)를 나타내며, GPU 메모리와 처리 코어 간 데이터 전송 속도에 영향을 미칩니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryClock` 태그를 지정합니다.

### GPU 그래픽 클럭 속도

GPU의 그래픽 렌더링 작업용 기본 클럭 속도(MHz)를 나타냅니다. 이 메트릭은 시각화나 렌더링 작업시 성능을 반영합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.graphicsClock` 태그를 지정합니다.

### GPU 교정 메모리 에러

GPU에서 오류 검출 프로토콜을 통해 W&B가 자동으로 교정한 메모리 에러의 수를 추적합니다. 이는 복구 가능한 하드웨어 이슈를 의미합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.correctedMemoryErrors` 태그를 지정합니다.

### GPU 미교정 메모리 에러
GPU에서 W&B가 교정하지 못한 메모리 에러 수를 추적합니다. 이는 비복구성 에러로 처리 신뢰도에 영향을 줄 수 있습니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.unCorrectedMemoryErrors` 태그를 지정합니다.

### GPU 인코더 사용률

GPU의 비디오 인코더 사용률(%)을 나타내며, 인코딩 작업(예: 비디오 렌더링) 중 인코더의 부하를 의미합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.encoderUtilization` 태그를 지정합니다.



## AMD GPU
W&B는 AMD에서 제공하는 `rocm-smi` 툴(`rocm-smi -a --json`)의 출력에서 메트릭을 추출합니다.

ROCm [6.x (최신)](https://rocm.docs.amd.com/en/latest/) 및 [5.x](https://rocm.docs.amd.com/en/docs-5.6.0/) 포맷을 지원합니다. 다양한 ROCm 포맷은 [AMD ROCm 공식 문서](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)에서 확인할 수 있습니다. 최신 포맷에는 더 많은 상세 정보가 포함됩니다.

### AMD GPU 사용률
각 AMD GPU 디바이스의 GPU 사용률(%)을 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 지정합니다.

### AMD GPU 메모리 할당률
각 AMD GPU 디바이스의 전체 사용 가능 메모리 대비 할당된 메모리 비율을 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 지정합니다.

### AMD GPU 온도
각 AMD GPU 디바이스의 온도(섭씨)를 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 지정합니다.

### AMD GPU 전력 사용량 (Watt)
각 AMD GPU 디바이스에서 소비된 전력(W)을 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 지정합니다.

### AMD GPU 전력 사용량 퍼센트
각 AMD GPU 디바이스의 최대 전력 대비 사용 비율(%)를 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 지정합니다.



## Apple ARM Mac GPU

### Apple GPU 사용률
ARM Mac에서 Apple GPU 디바이스의 GPU 사용률(%)을 나타냅니다.

W&B는 이 메트릭에 `gpu.0.gpu` 태그를 지정합니다.

### Apple GPU 메모리 할당률
ARM Mac에서 Apple GPU 디바이스의 전체 사용 가능 메모리 대비 할당된 메모리 비율을 나타냅니다.

W&B는 이 메트릭에 `gpu.0.memoryAllocated` 태그를 지정합니다.

### Apple GPU 온도
ARM Mac에서 Apple GPU 디바이스의 온도(섭씨)를 나타냅니다.

W&B는 이 메트릭에 `gpu.0.temp` 태그를 지정합니다.

### Apple GPU 전력 사용량 (Watt)
ARM Mac에서 Apple GPU 디바이스의 전력 사용량을 Watt로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.powerWatts` 태그를 지정합니다.

### Apple GPU 전력 사용량 퍼센트
ARM Mac에서 Apple GPU 디바이스의 최대 전력 대비 사용률(%)를 나타냅니다.

W&B는 이 메트릭에 `gpu.0.powerPercent` 태그를 지정합니다.



## Graphcore IPU
Graphcore IPU(Intelligence Processing Unit)는 기계 지능 작업에 특화된 고유 하드웨어 가속기입니다.

### IPU 디바이스 메트릭
이 메트릭들은 특정 IPU 디바이스의 다양한 통계를 나타냅니다. 각 메트릭은 디바이스 ID(`device_id`)와 메트릭 키(`metric_key`)로 구분됩니다. W&B는 이 메트릭에 `ipu.{device_id}.{metric_key}` 태그를 지정합니다.

메트릭은 Graphcore의 `gcipuinfo` 바이너리와 연동되는 전용 `gcipuinfo` 라이브러리를 통해 추출됩니다. `sample` 메소드는 프로세스 ID(`pid`)와 연결된 각 IPU 디바이스에서 이 메트릭들을 가져옵니다. 시간이 지나면서 값이 변경되는 메트릭, 또는 디바이스별로 메트릭이 처음 가져온 경우에만 로그로 기록되어 중복 기록을 방지합니다.

각 메트릭에 대해, `parse_metric` 메소드가 원본 문자열에서 값을 추출하는 데 사용됩니다. 여러 샘플에서 메트릭을 집계할 때는 `aggregate` 메소드가 활용됩니다.

아래는 사용 가능한 메트릭과 단위입니다:

- **평균 보드 온도** (`average board temp (C)`): IPU 보드의 온도(섭씨)
- **평균 다이 온도** (`average die temp (C)`): IPU 다이의 온도(섭씨)
- **클럭 속도** (`clock (MHz)`): IPU의 클럭 속도(MHz)
- **IPU 전력** (`ipu power (W)`): IPU의 전력 사용량(W)
- **IPU 사용률** (`ipu utilisation (%)`): IPU의 전체 사용률(%)
- **IPU 세션별 사용률** (`ipu utilisation (session) (%)`): 현재 세션에서의 IPU 사용률(%)
- **데이터 링크 속도** (`speed (GT/s)`): 초당 기가 전송(GT/s) 기준 데이터 전송 속도



## Google Cloud TPU
Tensor Processing Unit(TPU)는 Google에서 기계학습 워크로드 가속을 위해 개발한 전용 ASIC(Application Specific Integrated Circuit)입니다.

### TPU 메모리 사용량
TPU 코어별로 현재 High Bandwidth Memory 사용량(바이트)을 나타냅니다.

W&B는 이 메트릭에 `tpu.{tpu_index}.memoryUsageBytes` 태그를 지정합니다.

### TPU 메모리 사용률(%)
TPU 코어별로 현재 High Bandwidth Memory 사용량(%)을 나타냅니다.

W&B는 이 메트릭에 `tpu.{tpu_index}.memoryUsageBytes` 태그를 지정합니다.

### TPU Duty cycle
TPU 디바이스별 TensorCore duty cycle 비율(%)을 나타냅니다. 샘플 기간 동안 Accelerator TensorCore가 활성 처리 중이었던 시간의 비율로, 값이 클수록 TensorCore 활용도가 높음을 의미합니다.

W&B는 이 메트릭에 `tpu.{tpu_index}.dutyCycle` 태그를 지정합니다.



## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)은 AWS에서 제공하는 기계학습 워크로드 가속에 특화된 하드웨어 플랫폼입니다. AWS의 `neuron-monitor` 툴을 사용해 AWS Trainium 메트릭을 수집합니다.

### Trainium Neuron Core 사용률
각 NeuronCore의 사용률(%)을 코어별로 보고합니다.

W&B는 이 메트릭에 `trn.{core_index}.neuroncore_utilization` 태그를 지정합니다.

### Trainium 호스트 전체 메모리 사용량
호스트의 전체 메모리 사용량(바이트)을 나타냅니다.

W&B는 이 메트릭에 `trn.host_total_memory_usage` 태그를 지정합니다.

### Trainium Neuron 디바이스 전체 메모리 사용량
Neuron 디바이스의 전체 메모리 사용량(바이트)을 나타냅니다.

W&B는 이 메트릭에 `trn.neuron_device_total_memory_usage)` 태그를 지정합니다.

### Trainium 호스트 메모리 사용량 세부 정보

호스트의 메모리 사용량은 다음과 같이 분류됩니다:

- **Application Memory** (`trn.host_total_memory_usage.application_memory`): 애플리케이션에서 사용하는 메모리
- **Constants** (`trn.host_total_memory_usage.constants`): 상수 저장용 메모리
- **DMA Buffers** (`trn.host_total_memory_usage.dma_buffers`): DMA(Direct Memory Access) 버퍼용 메모리
- **Tensors** (`trn.host_total_memory_usage.tensors`): 텐서 데이터 저장용 메모리

### Trainium Neuron Core 메모리 사용 세부 정보
각 NeuronCore에 대한 상세 메모리 사용 정보를 제공합니다:

- **Constants** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **Model Code** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **Model Shared Scratchpad** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **Runtime Memory** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **Tensors** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
OpenMetrics / Prometheus 호환 데이터를 제공하는 외부 엔드포인트에서 메트릭을 수집 및 기록할 수 있으며, 커스텀 정규식 기반 메트릭 필터도 지원합니다.

GPU 클러스터 성능을 모니터링하는 대표적인 사용 예시는 [Monitoring GPU cluster performance in W&B](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1)와 [NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html) 관련 문서에서 확인하실 수 있습니다.
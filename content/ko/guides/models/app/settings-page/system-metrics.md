---
title: System metrics
description: wandb에서 자동으로 기록되는 메트릭
menu:
  default:
    identifier: ko-guides-models-app-settings-page-system-metrics
    parent: settings
weight: 70
---

이 페이지에서는 W&B SDK에서 추적하는 시스템 메트릭에 대한 자세한 정보를 제공합니다.

{{% alert %}}
`wandb`는 시스템 메트릭을 10초마다 자동으로 기록합니다.
{{% /alert %}}

## CPU

### 프로세스 CPU 사용률 (CPU)
사용 가능한 CPU 수로 정규화된 프로세스의 CPU 사용률입니다.

W&B는 이 메트릭에 `cpu` 태그를 할당합니다.

### CPU 사용률
코어별 시스템 CPU 사용률입니다.

W&B는 이 메트릭에 `cpu.{i}.cpu_percent` 태그를 할당합니다.

### 프로세스 CPU 스레드
프로세스에서 사용하는 스레드 수입니다.

W&B는 이 메트릭에 `proc.cpu.threads` 태그를 할당합니다.

## Disk

기본적으로 사용량 메트릭은 `/` 경로에 대해 수집됩니다. 모니터링할 경로를 구성하려면 다음 설정을 사용하세요.

```python
run = wandb.init(
    settings=wandb.Settings(
        _stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### 디스크 사용률
지정된 경로에 대한 총 시스템 디스크 사용률을 백분율로 나타냅니다.

W&B는 이 메트릭에 `disk.{path}.usagePercen` 태그를 할당합니다.

### 디스크 사용량
지정된 경로에 대한 총 시스템 디스크 사용량을 기가바이트(GB) 단위로 나타냅니다.
접근 가능한 경로가 샘플링되고 각 경로에 대한 디스크 사용량(GB)이 샘플에 추가됩니다.

W&B는 이 메트릭에 `disk.{path}.usageGB)` 태그를 할당합니다.

### 디스크 In
총 시스템 디스크 읽기(메가바이트(MB))를 나타냅니다.
첫 번째 샘플을 가져올 때 초기 디스크 읽기 바이트가 기록됩니다. 후속 샘플은 현재 읽기 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `disk.in` 태그를 할당합니다.

### 디스크 Out
총 시스템 디스크 쓰기(메가바이트(MB))를 나타냅니다.
[디스크 In]({{< relref path="#disk-in" lang="ko" >}})과 유사하게, 첫 번째 샘플을 가져올 때 초기 디스크 쓰기 바이트가 기록됩니다. 후속 샘플은 현재 쓰기 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `disk.out` 태그를 할당합니다.

## Memory

### 프로세스 Memory RSS
프로세스의 Memory Resident Set Size (RSS)를 메가바이트(MB) 단위로 나타냅니다. RSS는 메인 메모리(RAM)에 보관된 프로세스가 사용하는 메모리 부분입니다.

W&B는 이 메트릭에 `proc.memory.rssMB` 태그를 할당합니다.

### 프로세스 Memory 사용률
사용 가능한 총 메모리의 백분율로 프로세스의 메모리 사용량을 나타냅니다.

W&B는 이 메트릭에 `proc.memory.percent` 태그를 할당합니다.

### Memory 사용률
사용 가능한 총 메모리의 백분율로 총 시스템 메모리 사용량을 나타냅니다.

W&B는 이 메트릭에 `memory` 태그를 할당합니다.

### Memory Available
사용 가능한 총 시스템 메모리를 메가바이트(MB) 단위로 나타냅니다.

W&B는 이 메트릭에 `proc.memory.availableMB` 태그를 할당합니다.

## Network

### Network Sent
네트워크를 통해 전송된 총 바이트 수를 나타냅니다.
메트릭이 처음 초기화될 때 초기 바이트가 기록됩니다. 후속 샘플은 현재 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `network.sent` 태그를 할당합니다.

### Network Received

네트워크를 통해 수신된 총 바이트 수를 나타냅니다.
[Network Sent]({{< relref path="#network-sent" lang="ko" >}})와 유사하게, 메트릭이 처음 초기화될 때 초기 바이트가 기록됩니다. 후속 샘플은 현재 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `network.recv` 태그를 할당합니다.

## NVIDIA GPU

아래에 설명된 메트릭 외에도 프로세스 및/또는 해당 하위 프로세스가 특정 GPU를 사용하는 경우 W&B는 해당 메트릭을 `gpu.process.{gpu_index}...`로 캡처합니다.

### GPU Memory 사용률
각 GPU의 GPU 메모리 사용률을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memory` 태그를 할당합니다.

### GPU Memory Allocated
각 GPU에 대해 할당된 GPU 메모리를 사용 가능한 총 메모리의 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 할당합니다.

### GPU Memory Allocated Bytes
각 GPU에 대해 할당된 GPU 메모리를 바이트 단위로 지정합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocatedBytes` 태그를 할당합니다.

### GPU 사용률
각 GPU의 GPU 사용률을 백분율로 반영합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 할당합니다.

### GPU 온도
각 GPU의 GPU 온도를 섭씨로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 할당합니다.

### GPU 전력 사용량(와트)
각 GPU의 GPU 전력 사용량을 와트 단위로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 할당합니다.

### GPU 전력 사용률(%)

각 GPU의 전력 용량의 백분율로 GPU 전력 사용량을 반영합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 할당합니다.

### GPU SM 클럭 속도
GPU의 스트리밍 멀티프로세서(SM)의 클럭 속도를 MHz 단위로 나타냅니다. 이 메트릭은 계산 작업을 담당하는 GPU 코어 내의 처리 속도를 나타냅니다.

W&B는 `gpu.{gpu_index}.smClock` 태그를 이 메트릭에 할당합니다.

### GPU 메모리 클럭 속도
GPU 메모리와 처리 코어 간의 데이터 전송 속도에 영향을 미치는 GPU 메모리의 클럭 속도를 MHz 단위로 나타냅니다.

W&B는 `gpu.{gpu_index}.memoryClock` 태그를 이 메트릭에 할당합니다.

### GPU 그래픽 클럭 속도

MHz 단위로 표시되는 GPU의 그래픽 렌더링 작업을 위한 기본 클럭 속도를 나타냅니다. 이 메트릭은 시각화 또는 렌더링 작업 중 성능을 반영하는 경우가 많습니다.

W&B는 `gpu.{gpu_index}.graphicsClock` 태그를 이 메트릭에 할당합니다.

### GPU 수정된 메모리 오류

W&B가 오류 검사 프로토콜을 통해 자동으로 수정하는 GPU의 메모리 오류 수를 추적하여 복구 가능한 하드웨어 문제를 나타냅니다.

W&B는 `gpu.{gpu_index}.correctedMemoryErrors` 태그를 이 메트릭에 할당합니다.

### GPU 수정되지 않은 메모리 오류
W&B에서 수정하지 않은 GPU의 메모리 오류 수를 추적하여 처리 안정성에 영향을 줄 수 있는 복구 불가능한 오류를 나타냅니다.

W&B는 `gpu.{gpu_index}.unCorrectedMemoryErrors` 태그를 이 메트릭에 할당합니다.

### GPU 인코더 사용률

GPU의 비디오 인코더 사용률을 백분율로 나타내며 인코딩 작업(예: 비디오 렌더링)이 실행 중일 때의 부하를 나타냅니다.

W&B는 `gpu.{gpu_index}.encoderUtilization` 태그를 이 메트릭에 할당합니다.

## AMD GPU
W&B는 AMD에서 제공하는 `rocm-smi` 툴의 출력에서 메트릭을 추출합니다(`rocm-smi -a --json`).

### AMD GPU 사용률
각 AMD GPU 장치의 GPU 사용률을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 할당합니다.

### AMD GPU Memory Allocated
각 AMD GPU 장치에 대해 할당된 GPU 메모리를 사용 가능한 총 메모리의 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 할당합니다.

### AMD GPU 온도
각 AMD GPU 장치의 GPU 온도를 섭씨로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 할당합니다.

### AMD GPU 전력 사용량(와트)
각 AMD GPU 장치의 GPU 전력 사용량을 와트 단위로 나타냅니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 할당합니다.

### AMD GPU 전력 사용률(%)
각 AMD GPU 장치의 전력 용량의 백분율로 GPU 전력 사용량을 반영합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 할당합니다.

## Apple ARM Mac GPU

### Apple GPU 사용률
특히 ARM Mac에서 Apple GPU 장치의 GPU 사용률을 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.gpu` 태그를 할당합니다.

### Apple GPU Memory Allocated
ARM Mac의 Apple GPU 장치에 대해 할당된 GPU 메모리를 사용 가능한 총 메모리의 백분율로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.memoryAllocated` 태그를 할당합니다.

### Apple GPU 온도
ARM Mac의 Apple GPU 장치의 GPU 온도를 섭씨로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.temp` 태그를 할당합니다.

### Apple GPU 전력 사용량(와트)
ARM Mac의 Apple GPU 장치의 GPU 전력 사용량을 와트 단위로 나타냅니다.

W&B는 이 메트릭에 `gpu.0.powerWatts` 태그를 할당합니다.

### Apple GPU 전력 사용률(%)
ARM Mac의 Apple GPU 장치의 전력 용량의 백분율로 GPU 전력 사용량을 나타냅니다.

W&B는 이 메트릭에 `gpu.0.powerPercent` 태그를 할당합니다.

## Graphcore IPU
Graphcore IPU(Intelligence Processing Unit)는 기계 학습 작업을 위해 특별히 설계된 고유한 하드웨어 가속기입니다.

### IPU 장치 메트릭
이러한 메트릭은 특정 IPU 장치에 대한 다양한 통계를 나타냅니다. 각 메트릭에는 장치 ID(`device_id`)와 메트릭 키(`metric_key`)가 있어 식별할 수 있습니다. W&B는 이 메트릭에 `ipu.{device_id}.{metric_key}` 태그를 할당합니다.

메트릭은 Graphcore의 `gcipuinfo` 바이너리와 상호 작용하는 독점 `gcipuinfo` 라이브러리를 사용하여 추출됩니다. `sample` 메소드는 프로세스 ID(`pid`)와 연결된 각 IPU 장치에 대해 이러한 메트릭을 가져옵니다. 중복 데이터 로깅을 방지하기 위해 시간이 지남에 따라 변경되는 메트릭 또는 장치의 메트릭을 처음 가져오는 경우에만 기록됩니다.

각 메트릭에 대해 `parse_metric` 메소드는 원시 문자열 표현에서 메트릭의 값을 추출하는 데 사용됩니다. 그런 다음 `aggregate` 메소드를 사용하여 여러 샘플에서 메트릭을 집계합니다.

다음은 사용 가능한 메트릭 및 해당 단위를 나열합니다.

- **평균 보드 온도**(`average board temp (C)`): IPU 보드의 온도를 섭씨로 나타냅니다.
- **평균 다이 온도**(`average die temp (C)`): IPU 다이의 온도를 섭씨로 나타냅니다.
- **클럭 속도**(`clock (MHz)`): IPU의 클럭 속도를 MHz 단위로 나타냅니다.
- **IPU 전력**(`ipu power (W)`): IPU의 전력 소비를 와트 단위로 나타냅니다.
- **IPU 사용률**(`ipu utilisation (%)`): IPU 사용률을 백분율로 나타냅니다.
- **IPU 세션 사용률**(`ipu utilisation (session) (%)`): 현재 세션에 특정한 IPU 사용률을 백분율로 나타냅니다.
- **데이터 링크 속도**(`speed (GT/s)`): 데이터 전송 속도를 초당 기가 전송 단위로 나타냅니다.

## Google Cloud TPU
TPU(Tensor Processing Unit)는 기계 학습 워크로드를 가속화하는 데 사용되는 Google의 맞춤 개발 ASIC(Application Specific Integrated Circuits)입니다.

### TPU 메모리 사용량
TPU 코어당 현재 High Bandwidth Memory 사용량(바이트)입니다.

W&B는 이 메트릭에 `tpu.{tpu_index}.memoryUsageBytes` 태그를 할당합니다.

### TPU 메모리 사용률
TPU 코어당 현재 High Bandwidth Memory 사용률(%)입니다.

W&B는 이 메트릭에 `tpu.{tpu_index}.memoryUsageBytes` 태그를 할당합니다.

### TPU Duty cycle
TPU 장치당 TensorCore Duty cycle 백분율입니다. 가속기 TensorCore가 활발하게 처리하는 샘플 기간 동안의 시간 백분율을 추적합니다. 값이 클수록 TensorCore 사용률이 높다는 의미입니다.

W&B는 이 메트릭에 `tpu.{tpu_index}.dutyCycle` 태그를 할당합니다.

## AWS Trainium
[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)은 AWS에서 제공하는 특수 하드웨어 플랫폼으로, 기계 학습 워크로드 가속화에 중점을 둡니다. AWS의 `neuron-monitor` 툴은 AWS Trainium 메트릭을 캡처하는 데 사용됩니다.

### Trainium Neuron Core 사용률
코어별로 보고되는 각 NeuronCore의 사용률(%)입니다.

W&B는 이 메트릭에 `trn.{core_index}.neuroncore_utilization` 태그를 할당합니다.

### Trainium 호스트 메모리 사용량, 총계
호스트의 총 메모리 소비량(바이트)입니다.

W&B는 이 메트릭에 `trn.host_total_memory_usage` 태그를 할당합니다.

### Trainium Neuron 장치 총 메모리 사용량
Neuron 장치의 총 메모리 사용량(바이트)입니다.

W&B는 이 메트릭에 `trn.neuron_device_total_memory_usage)` 태그를 할당합니다.

### Trainium 호스트 메모리 사용량 분석:

다음은 호스트의 메모리 사용량 분석입니다.

- **애플리케이션 메모리**(`trn.host_total_memory_usage.application_memory`): 애플리케이션에서 사용하는 메모리입니다.
- **상수**(`trn.host_total_memory_usage.constants`): 상수에 사용되는 메모리입니다.
- **DMA 버퍼**(`trn.host_total_memory_usage.dma_buffers`): Direct Memory Access 버퍼에 사용되는 메모리입니다.
- **텐서**(`trn.host_total_memory_usage.tensors`): 텐서에 사용되는 메모리입니다.

### Trainium Neuron Core 메모리 사용량 분석
각 NeuronCore에 대한 자세한 메모리 사용량 정보입니다.

- **상수**(`trn.{core_index}.neuroncore_memory_usage.constants`)
- **모델 코드**(`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **모델 공유 스크래치패드**(`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **런타임 메모리**(`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **텐서**(`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
소비된 엔드포인트에 적용할 사용자 지정 정규식 기반 메트릭 필터를 지원하여 OpenMetrics / Prometheus 호환 데이터를 노출하는 외부 엔드포인트에서 메트릭을 캡처하고 기록합니다.

[이 리포트](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1)에서 [NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)를 사용하여 GPU 클러스터 성능을 모니터링하는 특정 경우에 이 기능을 사용하는 방법에 대한 자세한 예제를 참조하세요.

# Data Description

원활한 코드 이해를 돕기 위해 README에서 데이터의 차원(Shape)과 변수 구성을 상세히 설명합니다. 실제 데이터 파일(`inputs.npy`, `outputs.npy`)은 원자력 안전해석 시뮬레이션 기반의 비공개 데이터이므로 공개하지 않습니다.  

---

## Files

- `inputs.npy`  
  사고 시나리오를 나타내는 입력 feature 배열

- `outputs.npy`  
  각 시나리오에 대한 열수력(TH) 시계열 출력 배열

---

## Data Shape

- `inputs.npy`: `(N, S)`
- `outputs.npy`: `(N, T, C)`

### Dimensions

- `N = 10,000`  
  사고 시나리오 수

- `S = 27`  
  사고 시나리오 feature 수

- `T = 500`  
  다운샘플링된 시간 step 수

- `C = 4`  
  예측 대상 변수 수

각 시나리오는 72시간 MAAP 시뮬레이션 결과를 기반으로 하며, 원 시계열은 대표 시간점 500개로 다운샘플링되어 저장됩니다.

---

## Scenario Features

| Index | Description | Index | Description |
|------:|-------------|------:|-------------|
| 01-02 | AC power 1 on/off time | 18 | SDS open time |
| 03-04 | AC power 2 on/off time | 19-20 | CSS on/off time |
| 05-06 | AC power 3 on/off time | 21 | Recirculation off time |
| 07-08 | AC power 4 on/off time | 22 | Seal LOCA time |
| 09-10 | HPSI on/off time | 23 | Seal LOCA area |
| 11-12 | TDAFW on/off time | 24 | ET-LOOP |
| 13-14 | MDAFW on/off time | 25 | SBO-R |
| 15-16 | PLPP on/off time | 26 | SBO-S |
| 17 | Flag PSV Stuck | 27 | TSLOCA |

---

## Target Variables

출력 변수는 아래 4개 열수력 변수의 시계열입니다.

| Index | Variable | Description |
|------:|----------|-------------|
| 1 | PPS | Pressurizer Pressure |
| 2 | ZW_VESSEL | Reactor Vessel Water Level |
| 3 | TCRHOT | Peak Cladding Temperature |
| 4 | ZWPZ | Pressurizer Water Level |

---

## Notes

학습 시에는 각 시나리오 feature에 시간 feature `[t, t², t³]`를 추가하여 time-conditioned input을 구성합니다.  
즉, 실제 모델 입력 차원은 `27 + 3 = 30`입니다.
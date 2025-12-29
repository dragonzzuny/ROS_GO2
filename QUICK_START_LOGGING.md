# 🚀 빠른 시작 - 완벽한 로깅과 함께

**학습 결과를 빠짐없이 기록하면서 훈련하기**

---

## ⚡ 3단계로 시작

### 1단계: TensorBoard 실행 (별도 터미널)
```bash
cd ~/rl_dispatch_mvp
tensorboard --logdir runs
```

브라우저에서 http://localhost:6006 열기

---

### 2단계: 학습 시작
```bash
# 기본 학습 (5M steps, ~1-3시간)
python scripts/train_with_logging.py \
    --total-timesteps 5000000 \
    --seed 42 \
    --cuda \
    --experiment-name my_first_run

# 또는 빠른 테스트 (100K steps, ~5-10분)
python scripts/train_with_logging.py \
    --total-timesteps 100000 \
    --seed 42 \
    --experiment-name quick_test
```

**실시간 확인**:
- TensorBoard에서 `train/entropy`, `episode/return` 등 모니터링
- Console에서 맵별 성능 확인

---

### 3단계: 결과 분석
```bash
# 학습 완료 후 (또는 학습 중에도 가능)
python scripts/analyze_logs.py runs/my_first_run/20251230-123456

# 생성된 파일 확인
ls runs/my_first_run/20251230-123456/analysis/
# - learning_curves.png        (학습 곡선)
# - training_diagnostics.png   (PPO 진단)
# - map_comparison.png         (맵별 비교)
# - training_report.txt        (텍스트 요약)
```

---

## 📊 기록되는 모든 데이터

### ✅ 자동으로 저장됨

| 데이터 | 형식 | 용도 |
|--------|------|------|
| **TensorBoard** | events 파일 | 실시간 모니터링 |
| **Step CSV** | `csv/steps.csv` | 매 스텝 상세 데이터 |
| **Episode CSV** | `csv/episodes.csv` | 에피소드 통계 |
| **Update CSV** | `csv/updates.csv` | PPO 학습 메트릭 |
| **Map CSVs** | `csv/map_*.csv` | 맵별 성능 |
| **Checkpoints** | `checkpoints/*.pth` | 모델 저장 |
| **Config** | `training_config.yaml` | 설정 백업 |
| **Results** | `results.json` | 최종 결과 |

---

## 📈 실시간 모니터링 (TensorBoard)

### 필수 확인 Metrics

| Metric | 위치 | 정상 | 경고 |
|--------|------|------|------|
| **Entropy** | `train/entropy` | 0.03-0.10 | ❌ < 0.02 |
| **Approx KL** | `train/approx_kl` | > 0.001 | ❌ ≈ 0 |
| **Value Loss** | `train/value_loss` | < 100K | ❌ > 100K |
| **Success Rate** | `episode/event_success_rate` | 상승 | ❌ 하락 |
| **Return** | `episode/return` | 상승 | ❌ 하락 |

### 경고 발생 시

#### Entropy < 0.02 (Policy Collapse)
```bash
# 학습 중단 후 재시작 (hyperparameter 조정)
python scripts/train_with_logging.py \
    --entropy-coef 0.15 \
    --learning-rate 5e-5 \
    ...
```

#### Value Loss > 100K (Critic 폭주)
```bash
# Reward scale 조정
python scripts/train_with_logging.py \
    --value-loss-coef 2.0 \
    ...
```

---

## 📁 로그 파일 위치

```
runs/my_first_run/20251230-123456/
├── tensorboard/              ← 실시간 모니터링
├── csv/                      ← 상세 데이터 분석
│   ├── steps.csv            (매 스텝)
│   ├── episodes.csv         (에피소드)
│   ├── updates.csv          (PPO)
│   └── map_*.csv            (맵별)
├── checkpoints/              ← 모델
│   ├── update_50.pth
│   └── final.pth
├── analysis/                 ← 분석 결과 (analyze_logs 실행 후)
│   ├── learning_curves.png
│   ├── training_diagnostics.png
│   ├── map_comparison.png
│   └── training_report.txt
└── results.json              ← 최종 요약
```

---

## 🎯 학습 진행 체크리스트

### 시작 전 (5분)
- [ ] TensorBoard 실행 (`tensorboard --logdir runs`)
- [ ] 디스크 공간 확인 (2GB 이상 여유)
- [ ] Experiment name 설정

### 학습 중 (매 30분-1시간)
- [ ] TensorBoard 확인
  - [ ] `train/entropy` > 0.02 ✅
  - [ ] `train/value_loss` < 100K ✅
  - [ ] `episode/return` 상승 추세 ✅
- [ ] Console warning 없음 확인 ✅

### 학습 후
- [ ] `python scripts/analyze_logs.py runs/...` 실행
- [ ] `analysis/training_report.txt` 읽기
- [ ] 플롯 확인 (`learning_curves.png` 등)
- [ ] 최종 모델 백업 (`checkpoints/final.pth`)

---

## 🔍 결과 분석 예시

### 학습 성공 패턴

```
📈 learning_curves.png 확인:
  - Return: -5000 → 1500 (상승) ✅
  - Success Rate: 40% → 78% (상승) ✅
  - Coverage: 30% → 85% (상승) ✅

📊 training_diagnostics.png 확인:
  - Entropy: 0.08 → 0.03 (천천히 감소) ✅
  - Value Loss: 500K → 50K (감소) ✅
  - Explained Var: 0.1 → 0.7 (증가) ✅

📝 training_report.txt 확인:
  평균 Return: 1234.56 ± 456.78
  Event Success Rate: 78.9%
  Patrol Coverage: 85.3%
  ⚠️  경고 사항: 없음 - 학습 정상 진행 중 ✅
```

**결론**: 학습 성공! ✅

---

### 학습 실패 패턴 (Policy Collapse)

```
📈 learning_curves.png 확인:
  - Return: -2000 → -8000 (하락) ❌
  - Success Rate: 55% → 29% (하락) ❌

📊 training_diagnostics.png 확인:
  - Entropy: 0.08 → 0.001 (급락) ❌
  - Approx KL: 0.01 → 0.0000 (0으로) ❌
  - Value Loss: 2.5M (폭주) ❌

📝 training_report.txt 확인:
  ⚠️  경고 사항:
    - Entropy 너무 낮음 (0.001 < 0.02) - 탐색 부족
    - Value Loss 너무 높음 (2500000 > 100K) - Critic 학습 실패
```

**해결**: `POLICY_COLLAPSE_FIX.md` 참고

---

## 💡 Pro Tips

### 1. 실험 관리
```bash
# 실험마다 의미있는 이름
python scripts/train_with_logging.py \
    --experiment-name entropy_0.15_lr_1e4 \
    --entropy-coef 0.15 \
    --learning-rate 1e-4
```

### 2. 중단 후 재개 (향후 지원 예정)
```bash
# Checkpoint에서 재개
python scripts/train_with_logging.py \
    --resume runs/my_run/20251230-123456/checkpoints/update_500.pth
```

### 3. CSV 데이터 직접 분석
```python
import pandas as pd

# Episode 데이터 로드
df = pd.read_csv('runs/.../csv/episodes.csv')

# 최근 100 에피소드 평균
recent = df.tail(100)
print(f"Return: {recent['return'].mean():.2f}")
print(f"Success: {recent['event_success_rate'].mean()*100:.1f}%")

# 맵별 필터링
df_map = df[df['map_name'] == 'map_large_square']
print(f"Map episodes: {len(df_map)}")
```

### 4. TensorBoard 비교
```bash
# 여러 실험 비교
tensorboard --logdir runs \
    --logdir_spec run1:runs/exp1,run2:runs/exp2
```

---

## 📞 트러블슈팅

### "No module named 'matplotlib'"
```bash
pip install matplotlib pandas
```

### TensorBoard 실행 안 됨
```bash
pip install tensorboard
tensorboard --logdir runs --port 6007  # 다른 포트 시도
```

### CSV 파일 너무 큼 (> 1GB)
```python
# 샘플링해서 분석
import pandas as pd
df = pd.read_csv('steps.csv')
df_sampled = df.iloc[::100]  # 100 step마다 1개
df_sampled.to_csv('steps_sampled.csv', index=False)
```

### 디스크 공간 부족
```bash
# Step CSV 삭제 (에피소드/업데이트 데이터만 유지)
rm runs/.../csv/steps.csv

# 또는 압축
gzip runs/.../csv/steps.csv
```

---

## 🎓 추가 학습 자료

| 문서 | 내용 |
|------|------|
| `LOGGING_SYSTEM.md` | 전체 로깅 시스템 상세 설명 |
| `POLICY_COLLAPSE_FIX.md` | Policy collapse 해결 방법 |
| `ACTION_MASKING_FIX.md` | Action masking 구현 |
| `README.md` | 프로젝트 전체 설명 |

---

## ✅ 최종 체크

학습 전:
- [ ] `pip install matplotlib pandas tensorboard` 완료
- [ ] TensorBoard 실행 확인
- [ ] 실험 이름 설정

학습 시작:
- [ ] `python scripts/train_with_logging.py ...` 실행
- [ ] Console에 "🚀 Multi-Map Training with COMPREHENSIVE LOGGING" 출력
- [ ] TensorBoard에서 metrics 보임

학습 후:
- [ ] `python scripts/analyze_logs.py ...` 실행
- [ ] 4개 플롯 생성 확인
- [ ] `training_report.txt` 읽고 이해

---

**🎉 이제 완벽한 로깅과 함께 학습을 시작하세요!**

```bash
# 지금 바로 시작
tensorboard --logdir runs &
python scripts/train_with_logging.py \
    --total-timesteps 5000000 \
    --seed 42 \
    --cuda \
    --experiment-name my_first_run
```

**학습 중 확인**: http://localhost:6006

**작성자**: Reviewer 박용준
**작성일**: 2025-12-30

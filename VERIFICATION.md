# 멀티맵 시스템 검증 가이드

## 🧪 검증 순서

### 1단계: Config 로딩 테스트

```bash
cd rl_dispatch_mvp
source venv/bin/activate  # 또는 .venv\Scripts\activate (Windows)
python test_config_load.py
```

**예상 출력:**
```
Testing EnvConfig.load_yaml()...

Loading: .../configs/map_large_square.yaml
✅ EnvConfig loaded successfully!
   Map size: 100.0×100.0m
   Patrol points: 12
   Robot velocity: 2.0 m/s

✅ RewardConfig loaded successfully!
   w_event: 1.0
   w_patrol: 0.6
   w_safety: 2.0

============================================================
✅ All config loading tests passed!
============================================================
```

---

### 2단계: MultiMapEnv 테스트

```bash
python test_multimap.py
```

**예상 출력:**
```
================================================================================
Multi-Map System Verification Test
================================================================================

[1/5] Testing imports...
✅ Imports successful

[2/5] Loading map configurations...
✅ Found all 6 map configs

[3/5] Creating MultiMapPatrolEnv...
✅ Environment created
   Maps loaded: 6
   - map_large_square: 100.0×100.0m, 12 points
   - map_corridor: 120.0×30.0m, 10 points
   - map_l_shaped: 80.0×80.0m, 10 points
   - map_office_building: 90.0×70.0m, 14 points
   - map_campus: 150.0×120.0m, 16 points
   - map_warehouse: 140.0×100.0m, 12 points

[4/5] Testing environment reset...
✅ Reset successful
   Observation shape: (77,)
   Selected map: map_campus
   Episode count: 1

[5/5] Running test episodes...
   Episode 1: map_large_square, 10 steps, reward=-5.23
   Episode 2: map_corridor, 10 steps, reward=-3.12
   Episode 3: map_office_building, 10 steps, reward=-8.45
✅ Episodes ran successfully

[6/6] Checking coverage tracking...
✅ Coverage tracking works
   map_large_square: (50, 50), 142 total visits
   map_corridor: (15, 60), 87 total visits
   ...

================================================================================
✅ ALL TESTS PASSED!
================================================================================

Multi-Map System is working correctly!

Next steps:
  1. Train: python scripts/train_multi_map.py --total-timesteps 100000
  2. Visualize: python scripts/visualize_coverage.py --episodes-per-map 5
================================================================================
```

---

### 3단계: 짧은 학습 테스트 (선택사항)

```bash
# 100K steps만 학습 (약 5-10분)
python scripts/train_multi_map.py \
    --total-timesteps 100000 \
    --seed 42
```

**확인 사항:**
- 에러 없이 실행됨
- TensorBoard 로그 생성됨
- 맵별 성능 출력됨

---

## ❌ 에러 발생 시

### "KeyError: 'env'" 또는 "__init__() got unexpected keyword"

→ `config.py`의 수정이 제대로 반영되지 않음

**해결:**
```bash
# 1. 패키지 재설치
pip install -e . --force-reinstall --no-deps

# 2. Python 캐시 삭제
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 3. 다시 테스트
python test_config_load.py
```

---

### "ModuleNotFoundError: No module named 'rl_dispatch'"

→ 패키지가 설치되지 않음

**해결:**
```bash
pip install -e .
```

---

### "FileNotFoundError: configs/map_*.yaml"

→ 설정 파일 경로 문제

**해결:**
```bash
# 파일 존재 확인
ls configs/map_*.yaml

# 절대 경로로 테스트
python test_multimap.py
```

---

## ✅ 검증 완료 후

모든 테스트가 통과하면 다음 단계로 진행:

### 1. 본격 학습

```bash
# 5M steps 멀티맵 학습 (약 1-2시간, GPU)
python scripts/train_multi_map.py \
    --total-timesteps 5000000 \
    --seed 42 \
    --cuda
```

### 2. 성능 평가

```bash
# 일반화 성능 평가
python scripts/evaluate_generalization.py \
    --model runs/multi_map_ppo/*/checkpoints/final.pth \
    --episodes 50 \
    --save-json
```

### 3. 커버리지 확인

```bash
# 커버리지 히트맵
python scripts/visualize_coverage.py \
    --model runs/multi_map_ppo/*/checkpoints/final.pth \
    --episodes-per-map 20
```

---

## 📊 기대 결과

### 학습 진행
- 맵별 리턴이 점진적으로 증가
- 모든 맵에서 양수 리턴 달성
- 맵 간 성능 분산 감소

### 일반화 성능
- 평균 이벤트 성공률: **75%+**
- 평균 순찰 커버리지: **85%+**
- 최악 맵 리턴: **> 0**

### 커버리지
- 맵의 **85%+ 영역** 방문
- 순찰 포인트 주변 높은 밀도
- 사각지대 최소화

---

**작성일**: 2025-12-29
**작성자**: 박용준 (YJP)

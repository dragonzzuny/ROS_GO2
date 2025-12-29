A) 기본/거리 기반

Keep-Order (No-Replan)

Nearest-Next (Greedy NN)

k-Nearest Lookahead (kNN-horizon)

Shortest-ETA First (Nav2 ETA 기반)

Shortest-Path-Length First (Nav2 path length 기반)

B) 공백(gap) 기반 커버리지

Most-Overdue First (Max-Gap)

Top-K Overdue First

Overdue-Threshold First (gᵢ > G_th 우선)

Minimize Max-Gap (Minimax Gap)

Gap-Variance Minimization (Fairness / Var(g) 최소)

C) 거리×공백 균형(혼합 스코어)

Overdue + Distance Balance (α·gap − β·dist)

Overdue / Distance Ratio (gap/dist)

Softmax-Score Sampling (확률적 균형 선택)

Regret-based Balance (2nd best 대비 후회 최소)

D) 위험도/중요도 가중

Risk-Weighted Gap (wᵢ·gap)

Risk-Weighted Balance (wᵢ·(α·gap − β·dist))

Critical-Zone First (핵심구역 우선 루프)

Incident-History Boosted (사고이력 가중)

E) 이벤트/출동 회복 특화

Event-First then Resume (출동→원루트 복귀)

Event-First + Nearest Recovery

Event-First + Overdue Recovery

Event-First + Balanced Coverage Recovery

Event-Cluster Sweep (이벤트 주변 포인트 스윕)

F) 최소 변경(안정성 우선)

Minimal-Deviation Insert (현재 루트에 event 삽입)

Swap-2 (2-point swap)

Local 2-opt Improvement

Windowed Replan (앞 H개만 재정렬)

Anchor-Preserve (앵커 포인트 고정 재정렬)

G) 안전/성공률 우선 (Nav2 실패 고려)

Feasibility-First (Nav2 feasible만)

Low-Risk Corridor First (장애물/혼잡 낮은 구간)

Abort-Avoiding (최근 실패 방향 회피)

Energy-Saving (회전량/경사 최소)

H) 랜덤/탐색(학습 안정화용)

Uniform Random Reorder (제한된 랜덤)

ε-Greedy Mix (best 후보 + 랜덤)

Boltzmann Sampling over candidates

Noisy Score (score + noise)

너가 “학습용 후보”로 우선 뽑으면 좋은 세트 (추천)

최소 8~12개로 시작하는 게 안정적이라, 아래는 강추 조합(명단만):

Keep-Order

Nearest-Next

Shortest-ETA First

Most-Overdue First

Overdue-Threshold First

Overdue + Distance Balance

Risk-Weighted Gap

Balanced Coverage (Var(g) 최소)

Minimal-Deviation Insert

Windowed Replan (앞 H개 재정렬)
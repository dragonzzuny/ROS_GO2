"""
Event type definitions for industrial safety monitoring.

산업안전 평가 기준에 따른 이벤트 분류:
- 위험도 1-9 (1=낮음, 9=치명적)
- 실제 발생 가능한 안전 이벤트 목록
"""

from typing import Dict, Tuple
from enum import Enum


class EventCategory(Enum):
    """Event categories for industrial safety."""
    FIRE_SAFETY = "화재/폭발"
    INTRUSION = "침입/보안"
    FALLING_OBJECT = "낙하/추락"
    LEAK = "누수/누출"
    EQUIPMENT_FAILURE = "설비고장"
    HAZMAT = "위험물질"
    OBSTRUCTION = "통로차단"
    ABNORMAL_BEHAVIOR = "이상행동"
    ENVIRONMENTAL = "환경이상"


# 산업안전 이벤트 정의: (event_name, risk_level, category)
# 위험도 기준:
# 1-3: 저위험 (일상 점검, 경미한 이상)
# 4-6: 중위험 (조사 필요, 잠재적 위험)
# 7-9: 고위험 (즉시 대응, 중대 재해 가능)

INDUSTRIAL_SAFETY_EVENTS: Dict[str, Tuple[int, EventCategory]] = {
    # 화재/폭발 (7-9: 매우 위험)
    "화재감지": (9, EventCategory.FIRE_SAFETY),
    "연기감지": (8, EventCategory.FIRE_SAFETY),
    "과열감지": (7, EventCategory.FIRE_SAFETY),
    "전기화재위험": (8, EventCategory.FIRE_SAFETY),
    "가스누출": (9, EventCategory.HAZMAT),

    # 침입/보안 (4-8: 중-고위험)
    "무단침입": (8, EventCategory.INTRUSION),
    "비인가구역접근": (7, EventCategory.INTRUSION),
    "배회": (4, EventCategory.ABNORMAL_BEHAVIOR),
    "도난의심": (6, EventCategory.INTRUSION),

    # 낙하/추락 (6-9: 중-고위험)
    "낙하물감지": (8, EventCategory.FALLING_OBJECT),
    "추락위험": (9, EventCategory.FALLING_OBJECT),
    "구조물손상": (7, EventCategory.EQUIPMENT_FAILURE),
    "바닥파손": (6, EventCategory.EQUIPMENT_FAILURE),

    # 누수/누출 (4-8: 중-고위험)
    "누수감지": (5, EventCategory.LEAK),
    "화학물질누출": (9, EventCategory.HAZMAT),
    "유류누출": (8, EventCategory.HAZMAT),
    "배관파열": (7, EventCategory.LEAK),

    # 설비 고장 (3-7: 저-중위험)
    "설비이상음": (5, EventCategory.EQUIPMENT_FAILURE),
    "전력이상": (6, EventCategory.EQUIPMENT_FAILURE),
    "환기시스템고장": (5, EventCategory.ENVIRONMENTAL),
    "조명고장": (3, EventCategory.EQUIPMENT_FAILURE),

    # 통로/환경 (2-6: 저-중위험)
    "통로차단": (5, EventCategory.OBSTRUCTION),
    "비상구차단": (7, EventCategory.OBSTRUCTION),
    "미끄러움위험": (4, EventCategory.ENVIRONMENTAL),
    "온도이상": (4, EventCategory.ENVIRONMENTAL),
    "소음이상": (3, EventCategory.ENVIRONMENTAL),

    # 이상 행동 (3-8: 저-고위험)
    "쓰러짐감지": (8, EventCategory.ABNORMAL_BEHAVIOR),
    "싸움감지": (6, EventCategory.ABNORMAL_BEHAVIOR),
    "폭력의심": (7, EventCategory.ABNORMAL_BEHAVIOR),
    "이상행동": (5, EventCategory.ABNORMAL_BEHAVIOR),
    "응급상황": (9, EventCategory.ABNORMAL_BEHAVIOR),

    # 기타 (1-3: 저위험)
    "청결이상": (2, EventCategory.ENVIRONMENTAL),
    "정상순찰": (1, EventCategory.ENVIRONMENTAL),
    "점검필요": (3, EventCategory.EQUIPMENT_FAILURE),
}


def get_event_risk_level(event_name: str) -> int:
    """
    Get risk level for an event name.

    Args:
        event_name: Name of the event

    Returns:
        risk_level: 1-9 risk level
    """
    if event_name in INDUSTRIAL_SAFETY_EVENTS:
        return INDUSTRIAL_SAFETY_EVENTS[event_name][0]
    else:
        # Default to medium risk if unknown
        return 5


def get_event_category(event_name: str) -> EventCategory:
    """
    Get category for an event name.

    Args:
        event_name: Name of the event

    Returns:
        category: Event category
    """
    if event_name in INDUSTRIAL_SAFETY_EVENTS:
        return INDUSTRIAL_SAFETY_EVENTS[event_name][1]
    else:
        return EventCategory.ENVIRONMENTAL


def get_random_event_name(np_random) -> str:
    """
    Sample a random event name with probability weighted by risk level.
    Higher risk events are less frequent.

    Args:
        np_random: NumPy random number generator

    Returns:
        event_name: Randomly selected event name
    """
    import numpy as np

    events = list(INDUSTRIAL_SAFETY_EVENTS.keys())
    risk_levels = [INDUSTRIAL_SAFETY_EVENTS[e][0] for e in events]

    # 위험도가 높을수록 발생 확률 낮게 (역가중치)
    # P(risk=r) ~ 1/(r^2)  : 고위험 이벤트는 훨씬 드물게
    inv_risk = [1.0 / (r ** 2) for r in risk_levels]
    probs = np.array(inv_risk) / np.sum(inv_risk)

    return np_random.choice(events, p=probs)


def get_event_distribution_by_category() -> Dict[EventCategory, int]:
    """Get count of events per category."""
    from collections import defaultdict

    counts = defaultdict(int)
    for event_name, (risk, category) in INDUSTRIAL_SAFETY_EVENTS.items():
        counts[category] += 1

    return dict(counts)


# 위험도별 통계
def get_risk_level_statistics() -> Dict[str, int]:
    """Get statistics on risk levels."""
    from collections import defaultdict

    stats = defaultdict(int)
    for event_name, (risk, _) in INDUSTRIAL_SAFETY_EVENTS.items():
        if 1 <= risk <= 3:
            stats["low_risk"] += 1
        elif 4 <= risk <= 6:
            stats["medium_risk"] += 1
        elif 7 <= risk <= 9:
            stats["high_risk"] += 1

    return dict(stats)


if __name__ == "__main__":
    # Print event statistics
    print("=" * 80)
    print("산업안전 이벤트 목록")
    print("=" * 80)

    print(f"\n총 이벤트 수: {len(INDUSTRIAL_SAFETY_EVENTS)}")

    print("\n위험도별 분포:")
    stats = get_risk_level_statistics()
    print(f"  저위험 (1-3): {stats['low_risk']}개")
    print(f"  중위험 (4-6): {stats['medium_risk']}개")
    print(f"  고위험 (7-9): {stats['high_risk']}개")

    print("\n카테고리별 분포:")
    cat_stats = get_event_distribution_by_category()
    for cat, count in sorted(cat_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat.value}: {count}개")

    print("\n" + "=" * 80)
    print("이벤트 목록 (위험도별 정렬)")
    print("=" * 80)

    sorted_events = sorted(
        INDUSTRIAL_SAFETY_EVENTS.items(),
        key=lambda x: x[1][0],
        reverse=True
    )

    for event_name, (risk, category) in sorted_events:
        risk_label = "🔴" if risk >= 7 else ("🟡" if risk >= 4 else "🟢")
        print(f"{risk_label} [{risk}] {event_name:<20} - {category.value}")

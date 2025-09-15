"""
Target Class to Answer Class 매핑
- Target class: 이미지를 제공하는 클래스
- Answer class: 사용자가 맞혀야 하는 정답 클래스들
"""

# Target Class -> Answer Classes 매핑
TARGET_TO_ANSWER_MAPPING = {
    "금붕어": ["금붕어", "물고기"],
    "웜뱃": ["웜뱃"],
    "공작": ["새", "공작"],
    "긴꼬리흰앵무": ["새", "앵무새"],
    "금화조": ["새"],
    "파랑새류": ["새"],
    "코뿔새": ["새"],
    "까치": ["까치", "새"],
    "검은고니": ["새"],
    "무지개앵무": ["새", "앵무새"],
    "개": ["개", "강아지"],
    "고양이": ["고양이"]
}

# Answer Class -> Target Classes 역매핑 (검증용)
ANSWER_TO_TARGET_MAPPING = {}
for target, answers in TARGET_TO_ANSWER_MAPPING.items():
    for answer in answers:
        if answer not in ANSWER_TO_TARGET_MAPPING:
            ANSWER_TO_TARGET_MAPPING[answer] = []
        ANSWER_TO_TARGET_MAPPING[answer].append(target)

def get_answer_classes(target_class: str) -> list:
    """
    Target class에 대응하는 정답 클래스들을 반환
    
    Args:
        target_class: 이미지를 제공하는 타겟 클래스
        
    Returns:
        list: 정답 클래스 리스트
    """
    return TARGET_TO_ANSWER_MAPPING.get(target_class, [target_class])

def get_target_classes(answer_class: str) -> list:
    """
    정답 클래스에 대응하는 타겟 클래스들을 반환
    
    Args:
        answer_class: 사용자가 맞혀야 하는 정답 클래스
        
    Returns:
        list: 타겟 클래스 리스트
    """
    return ANSWER_TO_TARGET_MAPPING.get(answer_class, [answer_class])

def is_valid_answer(target_class: str, user_answer: str) -> bool:
    """
    사용자 답변이 타겟 클래스에 대해 유효한지 확인
    
    Args:
        target_class: 이미지를 제공한 타겟 클래스
        user_answer: 사용자가 입력한 답변
        
    Returns:
        bool: 유효한 답변인지 여부
    """
    valid_answers = get_answer_classes(target_class)
    return user_answer in valid_answers

def get_all_target_classes() -> list:
    """모든 타겟 클래스 리스트 반환"""
    return list(TARGET_TO_ANSWER_MAPPING.keys())

def get_all_answer_classes() -> list:
    """모든 정답 클래스 리스트 반환 (중복 제거)"""
    all_answers = set()
    for answers in TARGET_TO_ANSWER_MAPPING.values():
        all_answers.update(answers)
    return sorted(list(all_answers))

def print_mapping_info():
    """매핑 정보를 출력"""
    print("🎯 Target Class -> Answer Classes 매핑:")
    print("=" * 50)
    
    for target, answers in TARGET_TO_ANSWER_MAPPING.items():
        print(f"📸 {target:12} -> {', '.join(answers)}")
    
    print("\n📋 모든 Target Classes:")
    print("=" * 30)
    for i, target in enumerate(get_all_target_classes(), 1):
        print(f"{i:2d}. {target}")
    
    print("\n✅ 모든 Answer Classes:")
    print("=" * 30)
    for i, answer in enumerate(get_all_answer_classes(), 1):
        print(f"{i:2d}. {answer}")

if __name__ == "__main__":
    print_mapping_info()
    
    # 테스트
    print("\n🧪 매핑 테스트:")
    print("=" * 20)
    
    test_cases = [
        ("금붕어", "금붕어"),
        ("금붕어", "물고기"),
        ("금붕어", "고양이"),  # 잘못된 답변
        ("공작", "새"),
        ("공작", "공작"),
        ("개", "강아지"),
        ("고양이", "고양이")
    ]
    
    for target, user_answer in test_cases:
        is_valid = is_valid_answer(target, user_answer)
        status = "✅" if is_valid else "❌"
        print(f"{status} Target: {target:12} | User: {user_answer:8} | Valid: {is_valid}")

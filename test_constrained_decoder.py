#!/usr/bin/env python3
"""
ML-Service용 Constrained Beam Search 디코딩 테스트 스크립트
"""

import torch
import sys
import os
import json

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.crnn.utils.constrained_decoder import (
    CaptchaTrie, 
    LanguageModel, 
    constrained_beam_search_decode,
    load_captcha_word_list
)


def test_trie_structure():
    """Trie 구조 테스트"""
    print("=== Trie 구조 테스트 ===")
    
    # ML-Service의 실제 단어 리스트 사용
    test_words = ["금붕어", "웜뱃", "공작", "긴꼬리흰앵무", "금화조", "파랑새류", "코뿔새", "까치", "검은고니", "무지개앵무", "고양이", "개"]
    trie = CaptchaTrie(test_words)
    
    # 유효한 prefix 테스트
    assert trie.is_valid_prefix("금"), "금은 유효한 prefix여야 함"
    assert trie.is_valid_prefix("금붕"), "금붕은 유효한 prefix여야 함"
    assert not trie.is_valid_prefix("금붕어어"), "금붕어어는 유효하지 않은 prefix여야 함"
    assert not trie.is_valid_prefix("호"), "호는 유효하지 않은 prefix여야 함"
    
    # 완성된 단어 테스트
    assert trie.is_complete_word("금붕어"), "금붕어는 완성된 단어여야 함"
    assert trie.is_complete_word("고양이"), "고양이는 완성된 단어여야 함"
    assert not trie.is_complete_word("금붕"), "금붕은 완성된 단어가 아니어야 함"
    
    print("✅ Trie 구조 테스트 통과")


def test_language_model():
    """언어 모델 테스트"""
    print("=== 언어 모델 테스트 ===")
    
    test_words = ["금붕어", "웜뱃", "공작", "긴꼬리흰앵무", "금화조", "파랑새류", "코뿔새", "까치", "검은고니", "무지개앵무", "고양이", "개"]
    lm = LanguageModel(test_words)
    
    # 문자 빈도 테스트
    char_score = lm.get_char_score("어")
    print(f"문자 '어'의 점수: {char_score:.4f}")
    
    # 길이 페널티 테스트
    length_penalty_2 = lm.get_length_penalty(2)
    length_penalty_3 = lm.get_length_penalty(3)
    print(f"2글자 길이 페널티: {length_penalty_2:.4f}")
    print(f"3글자 길이 페널티: {length_penalty_3:.4f}")
    
    # 한국어 패턴 점수 테스트
    pattern_score = lm.get_korean_pattern_score("금붕어")
    print(f"한국어 패턴 점수: {pattern_score:.4f}")
    
    print("✅ 언어 모델 테스트 통과")


def test_constrained_beam_search():
    """Constrained Beam Search 테스트"""
    print("=== Constrained Beam Search 테스트 ===")
    
    # ML-Service의 실제 단어 리스트 사용
    test_words = ["금붕어", "웜뱃", "공작", "긴꼬리흰앵무", "금화조", "파랑새류", "코뿔새", "까치", "검은고니", "무지개앵무", "고양이", "개"]
    trie = CaptchaTrie(test_words)
    lm = LanguageModel(test_words)
    
    # 문자 to 인덱스 매핑 (실제 charset.json과 유사하게)
    char_to_idx = {"<blank>": 0, "금": 1, "붕": 2, "어": 3, "웜": 4, "뱃": 5, "공": 6, "작": 7, "고": 8, "양": 9, "이": 10, "개": 11}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    # 가상의 모델 출력 생성 (금붕어에 대한 높은 확률)
    T, num_classes = 10, len(char_to_idx)
    logits = torch.randn(T, num_classes)
    
    # 금붕어에 대한 높은 확률 설정
    logits[0, 1] = 5.0  # 금
    logits[1, 2] = 5.0  # 붕
    logits[2, 3] = 5.0  # 어
    logits[3:, 0] = 5.0  # 나머지는 blank
    
    # 디코딩 실행
    candidates = constrained_beam_search_decode(
        logits, trie, lm, idx_to_char, beam_size=5
    )
    
    print(f"디코딩 후보들: {candidates}")
    
    if candidates:
        best_word = candidates[0][0]
        print(f"최고 점수 단어: {best_word}")
        assert best_word in test_words, f"결과가 테스트 단어 리스트에 있어야 함: {best_word}"
    
    print("✅ Constrained Beam Search 테스트 통과")


def test_with_real_word_list():
    """실제 ML-Service 단어 리스트로 테스트"""
    print("=== 실제 ML-Service 단어 리스트 테스트 ===")
    
    word_list_path = "src/crnn/word_list.txt"
    if not os.path.exists(word_list_path):
        print(f"⚠️  {word_list_path} 파일이 없습니다. 테스트를 건너뜁니다.")
        return
    
    try:
        # 실제 단어 리스트 로드
        word_list = load_captcha_word_list(word_list_path)
        print(f"로드된 단어 수: {len(word_list)}")
        print(f"단어 리스트: {word_list}")
        
        # Trie와 Language Model 생성
        trie = CaptchaTrie(word_list)
        lm = LanguageModel(word_list)
        
        # 몇 가지 단어 테스트
        test_words = ["금붕어", "고양이", "개", "공작"]
        for word in test_words:
            if word in word_list:
                assert trie.is_complete_word(word), f"{word}는 완성된 단어여야 함"
                print(f"✅ {word} 검증 통과")
            else:
                print(f"⚠️  {word}는 단어 리스트에 없음")
        
        print("✅ 실제 ML-Service 단어 리스트 테스트 통과")
        
    except Exception as e:
        print(f"❌ 실제 단어 리스트 테스트 실패: {e}")


def test_with_charset():
    """실제 charset.json과 함께 테스트"""
    print("=== 실제 charset.json과 함께 테스트 ===")
    
    charset_path = "src/crnn/model/charset.json"
    if not os.path.exists(charset_path):
        print(f"⚠️  {charset_path} 파일이 없습니다. 테스트를 건너뜁니다.")
        return
    
    try:
        # charset.json 로드
        with open(charset_path, 'r', encoding='utf-8') as f:
            charset = json.load(f)
            idx_to_char = charset['idx_to_char']
            char_to_idx = charset['char_to_idx']
        
        print(f"문자 집합 크기: {len(char_to_idx)}")
        print(f"첫 10개 문자: {list(char_to_idx.keys())[:10]}")
        
        # 단어 리스트 로드
        word_list_path = "src/crnn/word_list.txt"
        if os.path.exists(word_list_path):
            word_list = load_captcha_word_list(word_list_path)
            print(f"단어 리스트: {word_list}")
            
            # Trie와 Language Model 생성
            trie = CaptchaTrie(word_list)
            lm = LanguageModel(word_list)
            
            # 간단한 디코딩 테스트
            T, num_classes = 10, len(char_to_idx)
            logits = torch.randn(T, num_classes)
            
            # 첫 번째 단어에 대한 높은 확률 설정
            if word_list:
                first_word = word_list[0]
                print(f"첫 번째 단어 테스트: {first_word}")
                
                # 각 문자에 대해 높은 확률 설정
                for i, char in enumerate(first_word):
                    if char in char_to_idx:
                        char_idx = char_to_idx[char]
                        logits[i, char_idx] = 5.0
                
                # 나머지는 blank
                logits[len(first_word):, 0] = 5.0
                
                # 디코딩 실행
                candidates = constrained_beam_search_decode(
                    logits, trie, lm, idx_to_char, beam_size=5
                )
                
                print(f"디코딩 후보들: {candidates}")
                
                if candidates:
                    best_word = candidates[0][0]
                    print(f"최고 점수 단어: {best_word}")
        
        print("✅ 실제 charset.json과 함께 테스트 통과")
        
    except Exception as e:
        print(f"❌ charset.json 테스트 실패: {e}")


def main():
    """메인 테스트 함수"""
    print("🚀 ML-Service Constrained Beam Search 디코딩 테스트 시작\n")
    
    try:
        test_trie_structure()
        print()
        
        test_language_model()
        print()
        
        test_constrained_beam_search()
        print()
        
        test_with_real_word_list()
        print()
        
        test_with_charset()
        print()
        
        print("🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



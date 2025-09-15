import torch
import math
from typing import List, Dict, Tuple, Union, Sequence, Mapping
import json
import os

class CaptchaTrie:
    """캡차 단어 리스트 기반 Trie 구조"""
    
    def __init__(self, word_list: List[str]):
        self.root = {}
        for word in word_list:
            self._insert(word)
    
    def _insert(self, word: str):
        """단어를 Trie에 삽입"""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['END'] = True  # 단어 완성 표시
    
    def is_valid_prefix(self, prefix: str) -> bool:
        """현재 prefix가 유효한 단어의 시작인지 확인"""
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
    
    def is_complete_word(self, word: str) -> bool:
        """완성된 단어인지 확인"""
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return 'END' in node
    
    def get_valid_next_chars(self, prefix: str) -> List[str]:
        """현재 prefix에서 가능한 다음 문자들 반환"""
        node = self.root
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
        return list(node.keys())


class LanguageModel:
    """캡차 특성에 맞는 언어 모델"""
    
    def __init__(self, word_list: List[str]):
        self.word_list = word_list
        self.char_freq = self._calculate_char_frequency()
        self.length_dist = self._calculate_length_distribution()
    
    def _calculate_char_frequency(self) -> Dict[str, float]:
        """문자 빈도 계산"""
        char_count = {}
        total_chars = 0
        
        for word in self.word_list:
            for char in word:
                char_count[char] = char_count.get(char, 0) + 1
                total_chars += 1
        
        # 정규화
        char_freq = {}
        for char, count in char_count.items():
            char_freq[char] = count / total_chars
        
        return char_freq
    
    def _calculate_length_distribution(self) -> Dict[int, float]:
        """단어 길이별 분포 계산"""
        length_count = {}
        total_words = len(self.word_list)
        
        for word in self.word_list:
            length = len(word)
            length_count[length] = length_count.get(length, 0) + 1
        
        # 정규화
        length_dist = {}
        for length, count in length_count.items():
            length_dist[length] = count / total_words
        
        return length_dist
    
    def get_char_score(self, char: str) -> float:
        """문자 빈도 기반 점수"""
        return self.char_freq.get(char, 0.001)  # 기본값
    
    def get_length_penalty(self, length: int) -> float:
        """길이 기반 페널티"""
        if length in self.length_dist:
            return math.log(self.length_dist[length])
        else:
            return -2.0  # 매우 낮은 점수
    
    def get_korean_pattern_score(self, prefix: str) -> float:
        """한국어 패턴 점수"""
        score = 0.0
        
        # 자음/모음 패턴 점수
        consonants = set('ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ')
        vowels = set('ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ')
        
        for i, char in enumerate(prefix):
            if char in consonants:
                score += 0.01
            elif char in vowels:
                score += 0.01
        
        return score


def constrained_beam_search_decode(
    logits: torch.Tensor,
    trie: CaptchaTrie,
    language_model: LanguageModel,
    idx_to_char: Union[Sequence[str], Mapping[int, str]],
    beam_size: int = 10,
    max_length: int = 6
) -> List[Tuple[str, float]]:
    """
    Constrained Beam Search with Language Model 디코딩
    
    Args:
        logits: [T, num_classes] - CTC 출력
        trie: CaptchaTrie - 캡차 단어 트리
        language_model: LanguageModel - 언어 모델
        idx_to_char: 인덱스 to 문자 매핑
        beam_size: 빔 크기
        max_length: 최대 단어 길이
    
    Returns:
        List[Tuple[str, float]]: (단어, 점수) 리스트
    """
    T, num_classes = logits.shape
    print(f"🔍 [Beam Search] 시작 - 시퀀스 길이: {T}, 클래스 수: {num_classes}, 빔 크기: {beam_size}")
    
    # idx -> char 접근 함수
    if isinstance(idx_to_char, Mapping):
        def _itc(i): return idx_to_char.get(i, "")
    else:
        def _itc(i): return idx_to_char[i] if 0 <= i < len(idx_to_char) else ""
    
    # Beam Search 상태: (prefix, ctc_score, lm_score, last_char, blank_count)
    beams = [("", 0.0, 0.0, -1, 0)]
    print(f"🌱 [Beam Search] 초기 빔 상태: {beams}")
    
    for t in range(T):
        new_beams = []
        char_probs = torch.softmax(logits[t], dim=0)
        
        # 상위 확률 문자들 출력 (디버깅용)
        top_probs, top_indices = torch.topk(char_probs, 5)
        top_chars = [_itc(idx.item()) for idx in top_indices]
        print(f"⏰ [Beam Search] 시간 {t+1}/{T} - 상위 확률: {list(zip(top_chars, top_probs.tolist()))}")
        
        for prefix, ctc_score, lm_score, last_char, blank_count in beams:
            # Blank 토큰 처리 (0번 인덱스)
            blank_prob = char_probs[0].item()
            if blank_prob > 1e-8:  # 수치적 안정성
                new_beams.append((
                    prefix, 
                    ctc_score + math.log(blank_prob), 
                    lm_score, 
                    last_char, 
                    blank_count + 1
                ))
            
            # 문자 토큰 처리
            for char_idx in range(1, num_classes):
                char_prob = char_probs[char_idx].item()
                
                if char_prob < 1e-8:  # 수치적 안정성
                    continue
                
                if char_idx == last_char:
                    # 같은 문자 연속 - CTC 규칙에 따라 무시
                    continue
                
                # 새 문자 추가
                char = _itc(char_idx)
                if not char:  # 유효하지 않은 문자
                    continue
                
                new_prefix = prefix + char
                
                # 길이 제한 확인
                if len(new_prefix) > max_length:
                    continue
                
                # Trie 제약 조건 확인
                if not trie.is_valid_prefix(new_prefix):
                    continue  # 유효하지 않은 경로 조기 제거
                
                # 언어 모델 점수 계산
                char_lm_score = language_model.get_char_score(char)
                length_penalty = language_model.get_length_penalty(len(new_prefix))
                pattern_score = language_model.get_korean_pattern_score(new_prefix)
                
                new_lm_score = lm_score + math.log(char_lm_score) + length_penalty + pattern_score
                
                # 총 점수 = CTC 점수 + 언어 모델 점수
                total_score = ctc_score + math.log(char_prob) + new_lm_score
                
                new_beams.append((
                    new_prefix, 
                    ctc_score + math.log(char_prob), 
                    new_lm_score, 
                    char_idx, 
                    0
                ))
        
        # Top-K 빔 선택
        beams = sorted(new_beams, key=lambda x: x[1] + x[2], reverse=True)[:beam_size]
        print(f"📊 [Beam Search] 시간 {t+1} 후 상위 빔들: {[(beam[0], f'{beam[1]+beam[2]:.2f}') for beam in beams[:3]]}")
    
    # 완성된 단어만 필터링하고 정렬
    valid_candidates = []
    for prefix, ctc_score, lm_score, _, _ in beams:
        if trie.is_complete_word(prefix):
            total_score = ctc_score + lm_score
            valid_candidates.append((prefix, total_score))
            print(f"✅ [Beam Search] 완성된 단어 발견: '{prefix}' (점수: {total_score:.4f})")
    
    print(f"🎯 [Beam Search] 최종 후보 수: {len(valid_candidates)}")
    return sorted(valid_candidates, key=lambda x: x[1], reverse=True)


def load_captcha_word_list(word_list_path: str) -> List[str]:
    """캡차 단어 리스트 로드"""
    with open(word_list_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def create_constrained_decoder(word_list_path: str):
    """Constrained Decoder 생성"""
    word_list = load_captcha_word_list(word_list_path)
    trie = CaptchaTrie(word_list)
    language_model = LanguageModel(word_list)
    return trie, language_model


# 기존 CTC 디코딩 함수 (주석처리)
def decode_prediction_old(pred, idx_to_char):
    """기존 CTC 디코딩 (주석처리)"""
    # pred = pred.detach().permute(1, 0, 2).cpu().numpy()  # [batch_size, seq_length, num_classes]
    # 
    # outputs = []
    # for p in pred:
    #     p = p.argmax(axis=1)  # 각 타임스텝에서 가장 높은 확률의 문자 선택
    #     
    #     # Merge repeated characters and remove blank label
    #     previous = -1
    #     out = []
    #     for c in p:
    #         if c != previous and c != 0:  # 0은 blank label
    #             out.append(idx_to_char[c])
    #         previous = c
    #     outputs.append(''.join(out))
    # 
    # return outputs
    pass


def decode_prediction(pred, idx_to_char, word_list_path: str = None, beam_size: int = 10):
    """
    새로운 Constrained Beam Search 디코딩
    
    Args:
        pred: [seq_length, batch_size, num_classes] - 모델 출력
        idx_to_char: 인덱스 to 문자 매핑
        word_list_path: 캡차 단어 리스트 경로
        beam_size: 빔 크기
    
    Returns:
        List[str]: 디코딩된 단어 리스트
    """
    print("🔍 [Constrained Decoder] 새로운 Constrained Beam Search 디코딩 시작")
    
    if word_list_path is None:
        # 기본 경로
        word_list_path = os.path.join(os.path.dirname(__file__), '../word_list.txt')
    
    print(f"📁 [Constrained Decoder] 단어 리스트 경로: {word_list_path}")
    
    # Constrained Decoder 생성
    trie, language_model = create_constrained_decoder(word_list_path)
    print(f"🌳 [Constrained Decoder] Trie 구조 생성 완료 (단어 수: {len(language_model.word_list)})")
    print(f"📊 [Constrained Decoder] 언어 모델 생성 완료 (문자 수: {len(language_model.char_freq)})")
    
    # pred: [seq_length, batch_size, num_classes]
    pred = pred.detach().permute(1, 0, 2)  # [batch_size, seq_length, num_classes]
    print(f"🔢 [Constrained Decoder] 입력 텐서 형태: {pred.shape}")
    
    outputs = []
    for i in range(pred.shape[0]):
        print(f"🎯 [Constrained Decoder] 배치 {i+1}/{pred.shape[0]} 디코딩 시작")
        
        # 각 배치에 대해 디코딩
        logits = pred[i]  # [seq_length, num_classes]
        
        # Constrained Beam Search 디코딩
        candidates = constrained_beam_search_decode(
            logits, trie, language_model, idx_to_char, beam_size
        )
        
        print(f"🔍 [Constrained Decoder] 배치 {i+1} 후보 수: {len(candidates)}")
        
        if candidates:
            # 가장 높은 점수의 후보 선택
            best_word = candidates[0][0]
            best_score = candidates[0][1]
            print(f"✅ [Constrained Decoder] 배치 {i+1} 최고 후보: '{best_word}' (점수: {best_score:.4f})")
            
            # 상위 3개 후보 출력
            if len(candidates) > 1:
                print(f"📋 [Constrained Decoder] 배치 {i+1} 상위 후보들:")
                for j, (word, score) in enumerate(candidates[:3]):
                    print(f"   {j+1}. '{word}' (점수: {score:.4f})")
            
            outputs.append(best_word)
        else:
            # 후보가 없으면 빈 문자열
            print(f"❌ [Constrained Decoder] 배치 {i+1} 유효한 후보 없음")
            outputs.append("")
    
    print(f"🎉 [Constrained Decoder] 디코딩 완료 - 결과: {outputs}")
    return outputs

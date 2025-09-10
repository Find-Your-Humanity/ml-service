import os
import json
import torch
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from pathlib import Path
import uuid

from .model.crnn import CRNN
from .utils.data_utils import process_image_enhanced, decode_prediction

class HandwritingPredictor:
    def __init__(self, model_path, char_to_idx, idx_to_char, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        
        # 모델 로드
        self.model = CRNN(
            num_channels=1,
            num_classes=len(char_to_idx) + 1  # CTC blank(0) 포함
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def _forward_logits(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """CRNN 순전파하여 원시 logits 반환: [T, B, C]"""
        with torch.no_grad():
            outputs = self.model(img_tensor)
        return outputs

    def _decode_with_pyctc(self, logits_np: np.ndarray, lexicon: list | None = None, beam_width: int = 50):
        """pyctcdecode 기반 빔서치 디코딩 (소형 lexicon 옵션). logits_np: [T, C]."""
        try:
            from pyctcdecode import build_ctcdecoder  # type: ignore
        except Exception as e:
            raise RuntimeError(f"pyctcdecode not available: {e}")

        # labels: idx_to_char를 그대로 사용 (이미 CTC_BLANK 포함됨)
        labels = [str(ch) if ch != "" else "CTC_BLANK" for ch in self.idx_to_char]
        print(f"🔧 [pyctcdecode] labels 생성: {len(labels)}개 문자")
        
        # lexicon 사용 가능성 검사 및 디코더 생성
        if lexicon and len(lexicon) > 0:
            print(f"🔧 [pyctcdecode] lexicon 감지됨: {lexicon}")
            print("🔧 [pyctcdecode] lexicon을 후처리로 활용하여 vocabulary 크기 유지")
            # lexicon 사용 시 pyctcdecode가 vocabulary를 확장하여 크기 불일치 발생
            # 기본 디코더 사용 후 lexicon을 활용한 후처리 수행
            decoder = build_ctcdecoder(labels=labels)
            print("✅ [pyctcdecode] 기본 디코더 생성 완료 (lexicon 후처리 모드)")
        else:
            # lexicon이 없을 때
            print("🔧 [pyctcdecode] lexicon 없음, 기본 디코더 생성")
            decoder = build_ctcdecoder(labels=labels)
            print("✅ [pyctcdecode] 기본 디코더 생성 완료")

        # pyctcdecode expects shape (T, C)
        print(f"🔧 [pyctcdecode] 빔서치 디코딩 시작: beam_width={beam_width}, logits_shape={logits_np.shape}")
        beams = decoder.decode_beams(logits_np, beam_width=beam_width, prune_history=True)
        print(f"📊 [pyctcdecode] 빔서치 결과: {len(beams)}개 빔 생성")
        
        if not beams:
            print("⚠️ [pyctcdecode] 빔서치 결과 없음")
            return "", {"avg_logprob": None, "margin": None}
            
        top1 = beams[0]
        top2 = beams[1] if len(beams) > 1 else None
        text = top1.text or ""
        avg_logprob = float(top1.logit_score) / max(1, len(text))
        margin = float(top1.logit_score - (top2.logit_score if top2 is not None else 0.0))
        
        # lexicon 후처리: 기본 디코딩 결과를 lexicon과 매칭하여 개선
        if lexicon and len(lexicon) > 0:
            print(f"🔧 [lexicon-post] 후처리 시작: 원본='{text}', lexicon={lexicon}")
            improved_text = self._apply_lexicon_postprocessing(text, lexicon)
            if improved_text != text:
                print(f"✅ [lexicon-post] 개선됨: '{text}' → '{improved_text}'")
                text = improved_text
            else:
                print(f"ℹ️ [lexicon-post] 개선 없음: '{text}'")
        
        print(f"✅ [pyctcdecode] 디코딩 완료: text='{text}', avg_logprob={avg_logprob:.4f}, margin={margin:.4f}")
        return text, {"avg_logprob": avg_logprob, "margin": margin}

    def _apply_lexicon_postprocessing(self, text: str, lexicon: list) -> str:
        """lexicon을 활용한 후처리: 기본 디코딩 결과를 lexicon과 매칭하여 개선"""
        if not text or not lexicon:
            return text
            
        # 1. 정확한 매칭 확인
        if text in lexicon:
            print(f"🎯 [lexicon-post] 정확한 매칭 발견: '{text}'")
            return text
            
        # 2. 유사도 기반 매칭 (편집 거리)
        best_match = None
        min_distance = float('inf')
        
        for word in lexicon:
            # 간단한 편집 거리 계산 (Levenshtein distance)
            distance = self._levenshtein_distance(text, word)
            if distance < min_distance:
                min_distance = distance
                best_match = word
                
        # 3. 임계값 기반 매칭 (편집 거리가 너무 크면 원본 유지)
        if best_match and min_distance <= max(1, len(text) // 2):
            print(f"🔧 [lexicon-post] 유사도 매칭: '{text}' → '{best_match}' (거리: {min_distance})")
            return best_match
        else:
            print(f"ℹ️ [lexicon-post] 유사한 단어 없음: '{text}' (최소거리: {min_distance})")
            return text
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """두 문자열 간의 편집 거리 계산"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    def predict(self, image, lexicon: list | None = None):
        """이미지에서 텍스트 예측 (lexicon 제공 시 pyctcdecode 빔서치 사용)."""
        # 이미지 전처리 (백엔드에서 하던 전처리를 이곳으로 이동)
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        # 알파 채널이 있으면 흰 배경에 합성 후 RGB로 변환
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            background.paste(img, (0, 0), img)
            img = background.convert('RGB')
        elif img.mode == 'P':
            img = img.convert('RGB')

        # 그레이스케일로 변환
        img = img.convert('L')

        img_array = process_image_enhanced(img)

        # 디버그: 전처리 후 모델 입력 이미지를 저장 (매 호출)
        try:
            debug_dir = Path(__file__).resolve().parents[2] / "debug_uploads"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"ocr_model_input_pil_{ts}_{uuid.uuid4().hex[:8]}.png"
            debug_img = Image.fromarray((np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8))
            debug_img.save(debug_dir / fname)
        except Exception:
            pass
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # 예측
        outputs = self._forward_logits(img_tensor)  # [T, B, C]

        if lexicon is not None and isinstance(lexicon, list) and len(lexicon) > 0:
            # pyctcdecode 빔서치 (소형 lexicon)
            logits_np = outputs[:, 0, :].detach().cpu().numpy().astype(np.float32)
            text, meta = self._decode_with_pyctc(logits_np, lexicon=lexicon, beam_width=50)
            # 게이팅 파라미터는 호출자에서 활용 가능
            # return text
            return text

        # [기존 Greedy 디코딩 경로]
        # with torch.no_grad():
        #     outputs = self.model(img_tensor)
        #     predictions = decode_prediction(outputs, self.idx_to_char)
        # return predictions[0]

        # Greedy로 폴백
        predictions = decode_prediction(outputs, self.idx_to_char)
        return predictions[0]

def main(args):
    # 문자 집합 로드 (charset_path 우선)
    if args.charset_path:
        with open(args.charset_path, 'r', encoding='utf-8') as f:
            charset = json.load(f)
            idx_to_char = charset['idx_to_char']
            char_to_idx = charset['char_to_idx']
    else:
        idx_to_char = args.idx_to_char
        char_to_idx = args.char_to_idx

    # 예측기 초기화
    predictor = HandwritingPredictor(
        args.model_path,
        char_to_idx,
        idx_to_char
    )
    
    # 이미지에서 텍스트 예측
    prediction = predictor.predict(args.image_path)
    print(f'Predicted text: {prediction}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--charset_path', type=str, default=None, help='models/charset.json 경로')
    # 하위 호환: 직접 dict 전달도 허용 (권장 X)
    parser.add_argument('--char_to_idx', type=dict, default=None)
    parser.add_argument('--idx_to_char', type=dict, default=None)

    args = parser.parse_args()
    main(args)


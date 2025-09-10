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



    def predict(self, image, lexicon: list | None = None):
        """이미지에서 텍스트 예측 (Greedy 디코딩 사용)."""
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

        # Greedy 디코딩만 사용
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


import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset

 

def load_image(image_path):
    """이미지를 로드하고 전처리합니다."""
    image = Image.open(image_path).convert('L')  # 그레이스케일로 변환
    return image

def process_image(image, target_height=32):
    """이미지를 지정된 높이로 리사이즈하고 정규화합니다."""
    w, h = image.size
    new_w = int(w * (target_height / h))
    image = image.resize((new_w, target_height), Image.LANCZOS)
    
    # PIL Image를 numpy array로 변환
    img_array = np.array(image).astype(np.float32)
    
    # 정규화 (0-1 범위로)
    img_array = img_array / 255.0
    
    return img_array

def preprocess_for_ocr(image: Image.Image, fg_threshold: int = 245, pad: int = 18) -> Image.Image:
    """OCR 친화 전처리
    - 연한 획/여백 문제를 줄이기 위해 컨텐츠 바운딩박스 크롭(+패딩)
    - 대비 강화(autocontrast)
    - 얇은 획을 살짝 두껍게(MaxFilter 3x3)
    """
    if image.mode != 'L':
        image = image.convert('L')

    np_img = np.array(image)
    mask = np_img < fg_threshold  # 비백색 픽셀
    if mask.any():
        ys, xs = np.where(mask)
        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        # 패딩 적용
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(np_img.shape[0] - 1, bottom + pad)
        right = min(np_img.shape[1] - 1, right + pad)
        image = image.crop((left, top, right + 1, bottom + 1))

    # 대비 강화 (상하위 0.5~1% 컷)
    image = ImageOps.autocontrast(image, cutoff=1)

    # Otsu 이진화 + 굵기 강화 2회 + 샤픈
    try:
        arr = np.array(image, dtype=np.uint8)
        # Otsu 임계값 계산
        hist = np.bincount(arr.ravel(), minlength=256)
        total = arr.size
        sum_total = np.dot(np.arange(256), hist)
        sumB = 0.0
        wB = 0.0
        varMax = -1.0
        threshold = 200
        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            varBetween = wB * wF * (mB - mF) ** 2
            if varBetween > varMax:
                varMax = varBetween
                threshold = t
        # 너무 낮게 잡히면 흐려지므로 하한선 적용
        thresh = max(200, int(threshold))
        arr = np.where(arr < thresh, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode='L')
        # 굵기 강화: MinFilter size=5 두 번
        image = image.filter(ImageFilter.MinFilter(size=5))
        image = image.filter(ImageFilter.MinFilter(size=5))
        # 샤픈
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=0))
    except Exception:
        pass

    # 바운딩박스 외곽에 추가 여백 부여(조금 더 넉넉하게)
    try:
        image = ImageOps.expand(image, border=18, fill=255)
    except Exception:
        pass

    return image

 

def process_image_enhanced(image: Image.Image, target_height: int = 32) -> np.ndarray:
    """전처리 + 리사이즈 + 정규화(0~1)
    - PIL 전처리(preprocess_for_ocr)만 사용
    - 높이 32로 리사이즈 후 0~1 정규화
    """
    try:
        print("🔧 Using PIL preprocessing")
    except Exception:
        pass
    pre = preprocess_for_ocr(image)
    return process_image(pre, target_height=target_height)

def encode_text(text, char_to_idx):
    """텍스트를 인덱스 시퀀스로 변환합니다."""
    return [char_to_idx[char] for char in text]

def decode_prediction(pred, idx_to_char):
    """CTC 디코딩을 수행합니다."""
    # pred: [seq_length, batch_size, num_classes]
    # 학습 중에도 사용되므로 detach() 후 CPU/NumPy로 변환
    pred = pred.detach().permute(1, 0, 2).cpu().numpy()  # [batch_size, seq_length, num_classes]
    
    outputs = []
    for p in pred:
        p = p.argmax(axis=1)  # 각 타임스텝에서 가장 높은 확률의 문자 선택
        
        # Merge repeated characters and remove blank label
        previous = -1
        out = []
        for c in p:
            if c != previous and c != 0:  # 0은 blank label
                out.append(idx_to_char[c])
            previous = c
        outputs.append(''.join(out))
    
    return outputs

class HandwritingDataset(Dataset):
    """손글씨 데이터셋 클래스"""
    def __init__(self, image_paths, labels, char_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드 및 전처리
        image = load_image(image_path)
        image = process_image(image)
        
        if self.transform:
            image = self.transform(image)
        
        # 텍스트를 인덱스로 변환
        label_encoded = encode_text(label, self.char_to_idx)
        
        return {
            'image': torch.FloatTensor(image).unsqueeze(0),  # [1, H, W]
            'label': torch.LongTensor(label_encoded),
            'label_length': len(label_encoded),
            'text': label
        }


def ctc_collate_fn(batch):
    """CTC 학습용 배치 결합 함수
    - 이미지: 폭을 최대 폭에 맞춰 좌측 정렬 제로패딩 [B, 1, H, W_max]
    - 라벨: 1D로 이어붙임
    - 라벨 길이: 각 항목 길이 텐서
    """
    # 이미지 크기 수집
    heights = [sample['image'].shape[-2] for sample in batch]
    widths = [sample['image'].shape[-1] for sample in batch]
    max_height = max(heights)
    max_width = max(widths)

    # 패딩된 이미지 텐서 준비
    images = torch.zeros((len(batch), 1, max_height, max_width), dtype=torch.float32)
    for i, sample in enumerate(batch):
        img = sample['image']  # [1, H, W]
        _, h, w = img.shape
        images[i, :, :h, :w] = img

    # 라벨 이어붙이기
    labels_list = [sample['label'] for sample in batch]
    labels = torch.cat(labels_list, dim=0)
    label_lengths = torch.tensor([sample['label_length'] for sample in batch], dtype=torch.long)

    texts = [sample['text'] for sample in batch]

    return {
        'image': images,
        'label': labels,
        'label_length': label_lengths,
        'text': texts,
    }

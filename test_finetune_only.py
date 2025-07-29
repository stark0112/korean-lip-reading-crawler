#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
본학습만 진행하는 테스트 스크립트
기존 사전학습된 가중치를 사용하여 본학습 수행
"""

import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train import (
    Config, LipROIExtractor, KoreanGraphemeProcessor,
    LipReadingModel, FinetuneTrainer, ModelEvaluator, collate_fn
)

def test_finetune_only():
    """기존 사전학습 가중치를 사용한 본학습 테스트"""
    print("🚀 본학습 테스트 시작!")
    
    # 설정
    config = Config()
    config.FINETUNE_EPOCHS = 5  # 본학습 에포크 수
    config.BATCH_SIZE = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 디바이스: {device}")
    
    # 1단계: 본학습 데이터 준비
    print("\n📊 1단계: 본학습 데이터 준비 중...")
    
    # TL48에서 라벨링된 데이터 사용
    labeled_data_dir = Path("009.립리딩(입모양) 음성인식 데이터/01.데이터/1.Training/자소_라벨링데이터")
    tl48_label_dir = labeled_data_dir / "TL48" / "소음환경1" / "C(일반인)" / "M(남성)" / "M(남성)_24"
    
    # 첫 번째 JSON 파일 사용
    json_files = list(tl48_label_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"라벨링 데이터를 찾을 수 없습니다: {tl48_label_dir}")
    
    json_path = json_files[0]
    # JSON 파일명에서 _jaso를 제거하고 .mp4 확장자 추가
    video_filename = json_path.stem.replace('_jaso', '') + ".mp4"
    # 원천데이터 경로에서 해당 비디오 파일 찾기
    source_data_dir = Path("009.립리딩(입모양) 음성인식 데이터/01.데이터/1.Training/원천데이터")
    ts48_source_dir = source_data_dir / "TS48" / "소음환경1" / "C(일반인)" / "M(남성)" / "M(남성)_24"
    video_path = ts48_source_dir / video_filename
    
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일이 없습니다: {video_path}")
    
    print(f"📊 본학습 데이터:")
    print(f"   - 비디오: {video_path.name}")
    print(f"   - 라벨: {json_path.name}")
    
    # 본학습 데이터셋 생성
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            annotations = data[0]['Sentence_info']
        else:
            annotations = data.get('Sentence_info', [])
        
        print(f"📊 JSON 구조: {type(data)}, 어노테이션 수: {len(annotations)}")
        
        finetune_dataset = CustomLabeledDataset(str(video_path), annotations)
        print(f"✅ 본학습 데이터셋 생성 완료: {len(finetune_dataset)}개 샘플")
    except Exception as e:
        print(f"⚠️ 본학습 데이터셋 생성 실패: {e}")
        raise
    
    # 데이터 분할 (8:2)
    total_samples = len(finetune_dataset)
    train_size = int(total_samples * 0.8)
    val_size = total_samples - train_size
    
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(finetune_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(finetune_dataset, val_indices)
    
    print(f"📈 훈련 데이터: {len(train_dataset)}개 샘플")
    print(f"📊 검증 데이터: {len(val_dataset)}개 샘플")
    
    # 본학습 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # 2단계: 본학습 모델 생성 및 사전학습 가중치 로드
    print("\n🎯 2단계: 본학습 모델 준비...")
    
    # 본학습 모델 생성
    finetune_model = LipReadingModel(vocab_size=config.VOCAB_SIZE).to(device)
    
    # 기존 사전학습된 가중치 로드
    pretrain_weight_path = Path("models/pretrain/pretrained_encoder.pt")
    
    if pretrain_weight_path.exists():
        try:
            # 사전학습 모델의 인코더 부분만 로드
            pretrained_state = torch.load(pretrain_weight_path, map_location='cpu')
            
            # Visual Frontend 가중치 로드
            if 'visual_frontend' in pretrained_state:
                finetune_model.visual_frontend.load_state_dict(pretrained_state['visual_frontend'])
                print("✅ Visual frontend 가중치 로드 완료")
            
            # Temporal Encoder 가중치 로드
            if 'temporal_encoder' in pretrained_state:
                finetune_model.temporal_encoder.load_state_dict(pretrained_state['temporal_encoder'])
                print("✅ Temporal encoder 가중치 로드 완료")
                
        except Exception as e:
            print(f"⚠️ 사전학습 가중치 로드 실패: {e}")
            print("처음부터 훈련을 시작합니다.")
    else:
        print("⚠️ 사전학습 가중치 파일을 찾을 수 없습니다.")
        print("처음부터 훈련을 시작합니다.")
    
    # 3단계: 본학습 실행
    print("\n🎯 3단계: 본학습 시작...")
    
    # 본학습 훈련기 생성
    finetune_trainer = FinetuneTrainer(
        model=finetune_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # 본학습 실행
    finetune_trainer.train(epochs=config.FINETUNE_EPOCHS)
    
    # 4단계: 모델 평가
    print("\n📊 4단계: 모델 평가...")
    
    # 평가기 생성
    evaluator = ModelEvaluator(finetune_model, device)
    
    # 검증 데이터로 평가
    metrics = evaluator.evaluate(val_loader)
    
    print(f"📊 평가 결과:")
    print(f"   - Grapheme Error Rate: {metrics['ger']:.4f}")
    print(f"   - Character Error Rate: {metrics['cer']:.4f}")
    
    # 5단계: 모델 저장
    print("\n💾 5단계: 모델 저장...")
    
    save_path = Path("models/finetune") / "finetuned_model.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(finetune_model.state_dict(), save_path)
    print(f"💾 본학습 모델 저장: {save_path}")
    
    return metrics

class CustomLabeledDataset(Dataset):
    """라벨링된 데이터셋"""
    
    def __init__(self, video_path, annotations):
        self.video_path = video_path
        self.annotations = annotations
        self.lip_extractor = LipROIExtractor()
        self.grapheme_processor = KoreanGraphemeProcessor()
        
        print(f"📊 비디오: {video_path}")
        print(f"📊 어노테이션 수: {len(annotations)}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 시간 구간 정보
        start_time = annotation['start_time']
        end_time = annotation['end_time']
        
        # 입술 시퀀스 추출
        lip_sequence = self.lip_extractor.extract_video_lip_sequence(
            self.video_path, start_time, end_time
        )
        
        if lip_sequence is None or len(lip_sequence) == 0:
            raise ValueError(f"입술 시퀀스를 추출할 수 없습니다: {self.video_path}")
        
        # 프레임 길이 맞추기 (패딩 또는 자르기)
        target_length = 64
        if len(lip_sequence) < target_length:
            # 패딩: 마지막 프레임으로 채우기
            padding_frames = [lip_sequence[-1]] * (target_length - len(lip_sequence))
            lip_sequence = np.concatenate([lip_sequence, padding_frames], axis=0)
        elif len(lip_sequence) > target_length:
            # 자르기: 앞부분 사용
            lip_sequence = lip_sequence[:target_length]
        
        # 텐서로 변환 (VisualFrontend가 기대하는 형태: [T, H, W, C])
        frames = torch.from_numpy(lip_sequence).float()
        
        # 자소 라벨 처리
        sentence_text = annotation['sentence_text']
        # 자소 배열을 문자열로 결합
        text = ''.join(sentence_text)
        label = self.grapheme_processor.text_to_graphemes(text)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'frames': frames,
            'graphemes': label,
            'frame_length': torch.tensor(len(lip_sequence), dtype=torch.long),
            'grapheme_length': torch.tensor(len(label), dtype=torch.long),
            'text': text
        }

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='본학습 테스트')
    parser.add_argument('--epochs', type=int, default=5, help='본학습 에포크 수')
    args = parser.parse_args()
    
    # 설정 업데이트
    config = Config()
    config.FINETUNE_EPOCHS = args.epochs
    
    # 테스트 실행
    metrics = test_finetune_only()
    
    print("\n🎉 본학습 테스트 완료!")
    print(f"📊 최종 성능: GER={metrics['ger']:.4f}, CER={metrics['cer']:.4f}")

if __name__ == "__main__":
    main() 
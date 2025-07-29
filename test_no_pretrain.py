# 사전학습 없이 본 훈련만 테스트
# TL48 하나 파일만 사용해서 빠르게 실험

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
import time
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import shutil
from datetime import datetime
import pandas as pd
import seaborn as sns

# train.py에서 필요한 클래스들 import
from train import (
    Config, LipROIExtractor, KoreanGraphemeProcessor,
    UnlabeledVideoDataset, LabeledVideoDataset, CrawlerVideoDataset,
    VisualFrontend, PositionalEncoding, TemporalEncoder, 
    MaskedVideoModel, LipReadingModel, PretrainTrainer, 
    FinetuneTrainer, ModelEvaluator, collate_fn,
    setup_directories
)

def test_no_pretrain():
    """사전학습 없이 본 훈련만 테스트"""
    print("🚀 사전학습 없이 본 훈련 테스트 시작!")
    
    # 설정
    config = Config()
    # 빠른 실험을 위해 에포크 수 조정
    config.PRETRAIN_EPOCHS = 10  # 100 -> 10
    config.FINETUNE_EPOCHS = 5   # 50 -> 5
    config.BATCH_SIZE = 2         # 1 -> 2 (BatchNorm 문제 해결)
    
    device = config.DEVICE
    print(f"⚙️ 설정:")
    print(f"   - 에포크: {config.FINETUNE_EPOCHS}")
    print(f"   - 배치 크기: {config.BATCH_SIZE}")
    print(f"   - 디바이스: {device}")
    
    # 메모리 정리
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 데이터 준비
    print("\n📊 데이터 준비 중...")
    
    # TL48에서 하나의 MP4 파일과 JSON 파일만 사용
    labeled_data_dir = Path("009.립리딩(입모양) 음성인식 데이터/01.데이터/1.Training/자소_라벨링데이터")
    source_data_dir = Path("009.립리딩(입모양) 음성인식 데이터/01.데이터/1.Training/원천데이터")
    
    tl48_label_dir = labeled_data_dir / "TL48" / "소음환경1" / "C(일반인)" / "M(남성)" / "M(남성)_24"
    ts48_source_dir = source_data_dir / "TS48" / "소음환경1" / "C(일반인)" / "M(남성)" / "M(남성)_24"
    
    # 첫 번째 JSON 파일 찾기
    json_files = list(tl48_label_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"TL48에서 JSON 파일을 찾을 수 없습니다: {tl48_label_dir}")
    
    # 첫 번째 JSON 파일과 대응하는 MP4 파일 (원천데이터에서)
    json_path = json_files[0]
    video_filename = json_path.stem.replace('_jaso', '') + ".mp4"  # _jaso 제거
    video_path = ts48_source_dir / video_filename
    
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일이 없습니다: {video_path}")
    
    print(f"📊 사용할 데이터:")
    print(f"   - 비디오: {video_path.name}")
    print(f"   - 라벨: {json_path.name}")
    
    # 데이터셋 생성
    try:
        # JSON 파일 구조 확인
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # 리스트 형태인 경우 첫 번째 항목의 Sentence_info 사용
            annotations = data[0]['Sentence_info']
        else:
            # 딕셔너리 형태인 경우 Sentence_info 키 사용
            annotations = data.get('Sentence_info', [])
        
        print(f"📊 JSON 구조: {type(data)}, 어노테이션 수: {len(annotations)}")
        
        # 커스텀 데이터셋 생성
        dataset = CustomLabeledDataset(str(video_path), annotations)
        print(f"✅ 데이터셋 생성 완료: {len(dataset)}개 샘플")
    except Exception as e:
        print(f"⚠️ 데이터셋 생성 실패: {e}")
        raise
    
    # 데이터 분할 (8:2)
    total_samples = len(dataset)
    train_size = int(total_samples * 0.8)
    val_size = total_samples - train_size
    
    # 랜덤 분할
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 서브셋 생성
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"📈 훈련 데이터: {len(train_dataset)}개 샘플")
    print(f"📊 검증 데이터: {len(val_dataset)}개 샘플")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 멀티프로세싱 오류 해결
        collate_fn=collate_fn,
        drop_last=True # 마지막 배치가 1개일 때 버리도록 설정
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # 멀티프로세싱 오류 해결
        collate_fn=collate_fn,
        drop_last=True # 마지막 배치가 1개일 때 버리도록 설정
    )
    
    # 모델 생성 (사전학습 없음)
    print("\n🤖 모델 생성 중...")
    model = LipReadingModel(vocab_size=config.VOCAB_SIZE).to(device)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ 모델 파라미터: {total_params:,}개")
    
    # 훈련
    print("\n🎯 본 훈련 시작...")
    trainer = FinetuneTrainer(model, train_loader, val_loader, device)
    trainer.train(epochs=config.FINETUNE_EPOCHS)
    
    # 평가
    print("\n🔍 모델 평가 중...")
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(val_loader)
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 사전학습 없이 본 훈련 결과")
    print("="*50)
    
    print(f"📈 훈련 완료!")
    print(f"📊 평가 결과:")
    print(f"   - GER (자소 오류율): {metrics['ger']:.4f} ({metrics['ger']*100:.2f}%)")
    print(f"   - CER (음절 오류율): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    
    # 모델 저장
    save_path = Path("test_results") / "no_pretrain_model.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 모델 저장: {save_path}")
    
    return metrics

class CustomLabeledDataset(Dataset):
    """커스텀 라벨링된 비디오 데이터셋 (자소 라벨링 데이터용)"""
    
    def __init__(self, video_path, annotations):
        self.video_path = Path(video_path)
        self.annotations = annotations
        self.lip_extractor = LipROIExtractor()
        self.grapheme_processor = KoreanGraphemeProcessor()
        
        print(f"📊 라벨 데이터: {len(self.annotations)}개 문장")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 시간 구간 정보
        start_time = annotation['start_time']
        end_time = annotation['end_time']
        
        # 자소 라벨 (이미 분해된 상태)
        sentence_graphemes = annotation['sentence_text']  # 자소 리스트
        grapheme_indices = []
        
        for grapheme in sentence_graphemes:
            if grapheme in self.grapheme_processor.grapheme_to_idx:
                grapheme_indices.append(self.grapheme_processor.grapheme_to_idx[grapheme])
        
        # 입술 시퀀스 추출
        lip_sequence = self.lip_extractor.extract_video_lip_sequence(
            self.video_path, start_time, end_time
        )
        
        return {
            'frames': torch.FloatTensor(lip_sequence),
            'graphemes': torch.LongTensor(grapheme_indices),
            'frame_length': len(lip_sequence),
            'grapheme_length': len(grapheme_indices),
            'text': ''.join(sentence_graphemes)
        }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='사전학습 없이 본 훈련 테스트')
    parser.add_argument('--run', action='store_true', help='테스트 실행')
    
    args = parser.parse_args()
    
    if args.run:
        # 테스트 실행
        metrics = test_no_pretrain()
        print(f"\n✅ 테스트 완료!")
    else:
        print("🚀 사전학습 없이 본 훈련 테스트")
        print("\n사용법:")
        print("python test_no_pretrain.py --run")
        print("\n이 테스트는 다음을 수행합니다:")
        print("1. TL48 하나 파일만 사용")
        print("2. 사전학습 없이 모델을 처음부터 훈련")
        print("3. 5 에포크 훈련 후 평가")
        print("4. 성능 지표 출력")

if __name__ == "__main__":
    main() 
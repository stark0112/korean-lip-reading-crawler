# 사전학습 + 본학습 순차 테스트
# 빠른 실험을 위해 작은 데이터셋 사용

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

def collate_fn_pretrain(batch):
    """사전학습용 collate 함수"""
    frames = torch.stack([item['frames'] for item in batch])
    return frames

def test_pretrain_finetune():
    """사전학습 + 본학습 순차 테스트"""
    print("🚀 사전학습 + 본학습 순차 테스트 시작!")
    
    # 설정
    config = Config()
    # 빠른 실험을 위해 에포크 수 조정
    config.PRETRAIN_EPOCHS = 5   # 100 -> 5
    config.FINETUNE_EPOCHS = 3   # 50 -> 3
    config.BATCH_SIZE = 2         # 메모리 절약
    
    device = config.DEVICE
    print(f"⚙️ 설정:")
    print(f"   - 사전학습 에포크: {config.PRETRAIN_EPOCHS}")
    print(f"   - 본학습 에포크: {config.FINETUNE_EPOCHS}")
    print(f"   - 배치 크기: {config.BATCH_SIZE}")
    print(f"   - 디바이스: {device}")
    
    # 메모리 정리
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 1단계: 사전학습 데이터 준비
    print("\n📊 1단계: 사전학습 데이터 준비 중...")
    
    # data/lip_videos에서 라벨링되지 않은 비디오 파일들 사용 (사전학습용)
    lip_videos_dir = Path("data/lip_videos")
    
    # 사전학습용 비디오 파일들 찾기
    video_files = list(lip_videos_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"사전학습용 비디오 파일을 찾을 수 없습니다: {lip_videos_dir}")
    
    # 모든 비디오 파일 사용
    pretrain_videos = video_files
    print(f"📊 사전학습용 비디오: {len(pretrain_videos)}개")
    for video in pretrain_videos:
        print(f"   - {video.name}")
    
    # 사전학습 데이터셋 생성
    try:
        pretrain_dataset = PretrainUnlabeledDataset(pretrain_videos)
        print(f"✅ 사전학습 데이터셋 생성 완료: {len(pretrain_dataset)}개 샘플")
    except Exception as e:
        print(f"⚠️ 사전학습 데이터셋 생성 실패: {e}")
        raise
    
    # 사전학습 데이터 로더
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn_pretrain
    )
    
    # 2단계: 사전학습
    print("\n🎯 2단계: 사전학습 시작...")
    
    # 사전학습 모델 생성
    pretrain_model = MaskedVideoModel().to(device)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in pretrain_model.parameters())
    print(f"🏗️ 사전학습 모델 파라미터: {total_params:,}개")
    
    # 사전학습 실행
    pretrain_trainer = PretrainTrainer(pretrain_model, pretrain_loader, device)
    pretrain_trainer.train(epochs=config.PRETRAIN_EPOCHS)
    
    # 사전학습된 가중치 저장
    pretrain_save_path = Path("test_results") / "pretrained_weights.pt"
    pretrain_save_path.parent.mkdir(exist_ok=True)
    
    # 인코더 부분만 저장
    encoder_weights = {
        'visual_frontend': pretrain_model.visual_frontend.state_dict(),
        'temporal_encoder': pretrain_model.temporal_encoder.state_dict()
    }
    torch.save(encoder_weights, pretrain_save_path)
    print(f"💾 사전학습 인코더 가중치 저장: {pretrain_save_path}")
    
    # 3단계: 본학습 데이터 준비
    print("\n📊 3단계: 본학습 데이터 준비 중...")
    
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
    
    # 4단계: 본학습
    print("\n🎯 4단계: 본학습 시작...")
    
    # 본학습 모델 생성
    finetune_model = LipReadingModel(vocab_size=config.VOCAB_SIZE).to(device)
    
    # 사전학습된 가중치 로드
    try:
        # 사전학습 모델의 인코더 부분만 로드
        pretrained_state = torch.load(pretrain_save_path, map_location='cpu')
        
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
    
    # 본학습 실행
    finetune_trainer = FinetuneTrainer(finetune_model, train_loader, val_loader, device)
    finetune_trainer.train(epochs=config.FINETUNE_EPOCHS)
    
    # 5단계: 평가
    print("\n🔍 5단계: 모델 평가 중...")
    evaluator = ModelEvaluator(finetune_model, device)
    metrics = evaluator.evaluate(val_loader)
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 사전학습 + 본학습 결과")
    print("="*50)
    
    print(f"📈 훈련 완료!")
    print(f"📊 평가 결과:")
    print(f"   - GER (자소 오류율): {metrics['ger']:.4f} ({metrics['ger']*100:.2f}%)")
    print(f"   - CER (음절 오류율): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    
    # 최종 모델 저장
    final_save_path = Path("test_results") / "pretrain_finetune_model.pt"
    torch.save(finetune_model.state_dict(), final_save_path)
    print(f"💾 최종 모델 저장: {final_save_path}")
    
    return metrics

class PretrainUnlabeledDataset(Dataset):
    """사전학습용 라벨링되지 않은 비디오 데이터셋 - 전체 영상 활용"""
    
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.lip_extractor = LipROIExtractor()
        self.target_length = 64  # 고정된 시퀀스 길이
        self.clip_duration = 2.0  # 2초 클립
        
        # 각 영상의 총 길이와 클립 수 계산
        self.video_info = []
        total_clips = 0
        
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            
            # 전체 영상을 2초 클립으로 나누기 (중복 없이)
            num_clips = int(duration / self.clip_duration)
            if num_clips < 1:
                num_clips = 1  # 최소 1개 클립 보장
            
            self.video_info.append({
                'path': video_path,
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'num_clips': num_clips,
                'start_clip_idx': total_clips
            })
            
            total_clips += num_clips
        
        self.total_clips = total_clips
        
        print(f"📊 사전학습 비디오: {len(self.video_paths)}개")
        for info in self.video_info:
            print(f"   - {info['path'].name}: {info['duration']:.1f}초 → {info['num_clips']}개 클립")
        print(f"📊 총 클립 수: {total_clips}개")
    
    def __len__(self):
        return self.total_clips
    
    def __getitem__(self, idx):
        # 어떤 영상의 몇 번째 클립인지 계산
        video_idx = 0
        local_clip_idx = idx
        
        for i, info in enumerate(self.video_info):
            if local_clip_idx < info['num_clips']:
                video_idx = i
                break
            local_clip_idx -= info['num_clips']
        
        video_info = self.video_info[video_idx]
        video_path = video_info['path']
        duration = video_info['duration']
        
        # 연속적인 클립 생성 (전체 영상을 순차적으로 사용)
        start_time = local_clip_idx * self.clip_duration
        end_time = min(start_time + self.clip_duration, duration)
        
        # 입술 시퀀스 추출
        lip_sequence = self.lip_extractor.extract_video_lip_sequence(
            video_path, start_time, end_time
        )
        
        # 프레임 길이 맞추기 (패딩 또는 자르기)
        if len(lip_sequence) < self.target_length:
            # 패딩: 마지막 프레임으로 채우기
            padding_frames = [lip_sequence[-1]] * (self.target_length - len(lip_sequence))
            lip_sequence = np.concatenate([lip_sequence, padding_frames], axis=0)
        elif len(lip_sequence) > self.target_length:
            # 자르기: 앞부분 사용
            lip_sequence = lip_sequence[:self.target_length]
        
        return {
            'frames': torch.FloatTensor(lip_sequence),
            'frame_length': self.target_length
        }

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
    parser = argparse.ArgumentParser(description='사전학습 + 본학습 순차 테스트')
    parser.add_argument('--run', action='store_true', help='테스트 실행')
    
    args = parser.parse_args()
    
    if args.run:
        # 테스트 실행
        metrics = test_pretrain_finetune()
        print(f"\n✅ 테스트 완료!")
    else:
        print("🚀 사전학습 + 본학습 순차 테스트")
        print("\n사용법:")
        print("python test_pretrain_finetune.py --run")
        print("\n이 테스트는 다음을 수행합니다:")
        print("1. 사전학습: 라벨링되지 않은 비디오로 마스킹 비디오 모델링")
        print("2. 본학습: 라벨링된 데이터로 자소 예측 훈련")
        print("3. 사전학습된 가중치를 본학습에 전이")
        print("4. 성능 지표 출력")

if __name__ == "__main__":
    main() 
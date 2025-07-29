# 완전한 한국어 립리딩 파이프라인
# MP4 파일만 있으면 사전학습부터 평가까지 모든 과정 자동화

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

# ===========================================
# 설정 및 하이퍼파라미터
# ===========================================

class Config:
    # 입술 ROI 설정
    LIP_ROI_SIZE = (112, 112)  # 입술 영역 크기 (정사각형)
    LIP_MARGIN = 10           # 입술 주변 여백 (픽셀) - 작은 영역
    
    # 시퀀스 설정
    PRETRAIN_SEQUENCE_LENGTH = 16  # 사전학습 시퀀스 길이 (0.64초 @ 25fps)
    TARGET_FPS = 25               # 표준 프레임레이트
    
    # 훈련 설정
    PRETRAIN_EPOCHS = 100
    FINETUNE_EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    
    # 모델 설정
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    VOCAB_SIZE = 54  # 53개 자소 + blank
    
    # 데이터 경로
    UNLABELED_VIDEOS_DIR = "data/unlabeled_videos"
    LABELED_VIDEO_PATH = "data/labeled_video.mp4"
    LABELS_JSON_PATH = "data/labels.json"
    PROCESSED_DIR = "data/processed"
    MODELS_DIR = "models"
    
    # GPU 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===========================================
# 1. 입술 ROI 추출기 (MediaPipe 기반)
# ===========================================

class LipROIExtractor:
    """입술 영역 정밀 추출기 (MediaPipe 사용)"""
    
    def __init__(self):
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,  # 더 안정적인 설정
            min_tracking_confidence=0.5     # 더 안정적인 설정
        )
        
        # 입술 랜드마크 인덱스 (MediaPipe Face Mesh 기준)
        # 외곽 입술 (468개 랜드마크 중)
        self.outer_lip_indices = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]
        
        # 내곽 입술
        self.inner_lip_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318
        ]
        
        self.all_lip_indices = self.outer_lip_indices + self.inner_lip_indices
        
        # 이전 프레임 저장 (백업용)
        self.last_valid_lip_region = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10  # 연속 실패 제한
    
    def extract_lip_region(self, frame):
        """
        프레임에서 입술 영역 추출 (MediaPipe 사용)
        
        Args:
            frame: BGR 이미지 [H, W, 3]
            
        Returns:
            lip_region: 입술 영역 [112, 112, 3] 또는 None
            landmarks: 입술 랜드마크 좌표
        """
        try:
            # RGB로 변환 (MediaPipe는 RGB 사용)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 얼굴 랜드마크 검출
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None, None
            
            # 첫 번째 얼굴 선택
            face_landmarks = results.multi_face_landmarks[0]
            
            # 입술 랜드마크 좌표 추출
            lip_points = []
            for idx in self.all_lip_indices:
                landmark = face_landmarks.landmark[idx]
                # 좌표를 픽셀 좌표로 변환
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                lip_points.append([x, y])
            
            lip_points = np.array(lip_points)
            
            # 입술 바운딩 박스 계산
            x_min, y_min = lip_points.min(axis=0)
            x_max, y_max = lip_points.max(axis=0)
            
            # 여백 추가 (입술 주변 정보도 포함)
            margin = Config.LIP_MARGIN
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(frame.shape[1], x_max + margin)
            y_max = min(frame.shape[0], y_max + margin)
            
            # 정사각형 ROI 생성 (긴 쪽에 맞춤)
            width = x_max - x_min
            height = y_max - y_min
            size = max(width, height)
            
            # 중심점 계산
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # 정사각형 좌표 계산
            half_size = size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(frame.shape[1], center_x + half_size)
            y2 = min(frame.shape[0], center_y + half_size)
            
            # ROI 추출
            lip_roi = frame[y1:y2, x1:x2]
            
            # 112x112로 리사이즈
            lip_region = cv2.resize(lip_roi, Config.LIP_ROI_SIZE)
            
            return lip_region, lip_points
            
        except Exception as e:
            print(f"입술 추출 실패: {e}")
            return None, None
            return None, None
    
    def extract_video_lip_sequence(self, video_path, start_time=None, end_time=None):
        """
        비디오에서 입술 시퀀스 추출
        
        Args:
            video_path: 비디오 파일 경로
            start_time: 시작 시간 (초, None이면 처음부터)
            end_time: 끝 시간 (초, None이면 끝까지)
            
        Returns:
            lip_sequence: 입술 시퀀스 [T, 112, 112, 3]
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 시작/끝 프레임 계산
        if start_time is not None:
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            start_frame = 0
            
        if end_time is not None:
            end_frame = int(end_time * fps)
        else:
            end_frame = total_frames
        
        lip_frames = []
        frame_count = start_frame
        
        video_path_obj = Path(video_path)
        print(f"📹 비디오 처리 중: {video_path_obj.name}")
        print(f"   FPS: {fps}, 총 프레임: {total_frames}")
        
        pbar = tqdm(total=end_frame-start_frame, desc="프레임 처리")
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 입술 영역 추출
            lip_region, landmarks = self.extract_lip_region(frame)
            
            if lip_region is not None:
                # 정규화 [0, 255] → [0, 1]
                lip_region = lip_region.astype(np.float32) / 255.0
                lip_frames.append(lip_region)
            else:
                # 입술 검출 실패 시 이전 프레임 복사 (있다면)
                if len(lip_frames) > 0:
                    lip_frames.append(lip_frames[-1].copy())
            
            frame_count += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        if len(lip_frames) == 0:
            raise ValueError(f"유효한 입술 프레임을 찾을 수 없습니다: {video_path}")
        
        print(f"✅ 추출 완료: {len(lip_frames)}프레임")
        return np.array(lip_frames)

# ===========================================
# 2. 한국어 자소 처리기
# ===========================================

class KoreanGraphemeProcessor:
    """한국어 자소 분해/조합 처리기"""
    
    def __init__(self):
        # 한국어 자소 매핑 (논문 기준 54개 클래스)
        self.grapheme_to_idx = {
            '<blank>': 0,  # CTC blank 토큰
            ' ': 1,        # 공백
            
            # 초성 (19개)
            'ㄱ': 2, 'ㄲ': 3, 'ㄴ': 4, 'ㄷ': 5, 'ㄸ': 6, 'ㄹ': 7, 'ㅁ': 8, 'ㅂ': 9,
            'ㅃ': 10, 'ㅅ': 11, 'ㅆ': 12, 'ㅇ': 13, 'ㅈ': 14, 'ㅉ': 15, 'ㅊ': 16,
            'ㅋ': 17, 'ㅌ': 18, 'ㅍ': 19, 'ㅎ': 20,
            
            # 중성 (21개)
            'ㅏ': 21, 'ㅐ': 22, 'ㅑ': 23, 'ㅒ': 24, 'ㅓ': 25, 'ㅔ': 26, 'ㅕ': 27,
            'ㅖ': 28, 'ㅗ': 29, 'ㅘ': 30, 'ㅙ': 31, 'ㅚ': 32, 'ㅛ': 33, 'ㅜ': 34,
            'ㅝ': 35, 'ㅞ': 36, 'ㅟ': 37, 'ㅠ': 38, 'ㅡ': 39, 'ㅢ': 40, 'ㅣ': 41,
            
            # 종성 (간소화된 12개)
            'ㄱ_': 42, 'ㄴ_': 43, 'ㄷ_': 44, 'ㄹ_': 45, 'ㅁ_': 46, 'ㅂ_': 47,
            'ㅅ_': 48, 'ㅇ_': 49, 'ㅈ_': 50, 'ㅊ_': 51, 'ㅋ_': 52, 'ㅌ_': 53
        }
        
        self.idx_to_grapheme = {v: k for k, v in self.grapheme_to_idx.items()}
        
    def text_to_graphemes(self, text):
        """텍스트를 자소 인덱스 리스트로 변환"""
        graphemes = []
        
        for char in text:
            if char == ' ':
                graphemes.append(self.grapheme_to_idx[' '])
            elif '가' <= char <= '힣':
                decomposed = self.decompose_hangul(char)
                for g in decomposed:
                    if g in self.grapheme_to_idx:
                        graphemes.append(self.grapheme_to_idx[g])
            # 기타 문자는 무시
            
        return graphemes
    
    def decompose_hangul(self, char):
        """한글 음절을 자소로 분해"""
        if not ('가' <= char <= '힣'):
            return []
        
        base = ord(char) - ord('가')
        초성_idx = base // (21 * 28)
        중성_idx = (base % (21 * 28)) // 28
        종성_idx = base % 28
        
        초성_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        중성_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        종성_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        
        result = [초성_list[초성_idx], 중성_list[중성_idx]]
        
        if 종성_idx > 0:
            # 종성을 간소화 (복잡한 종성은 기본 자음으로)
            종성 = 종성_list[종성_idx]
            if 종성 in ['ㄱ', 'ㄲ', 'ㄳ']:
                result.append('ㄱ_')
            elif 종성 in ['ㄴ', 'ㄵ', 'ㄶ']:
                result.append('ㄴ_')
            elif 종성 == 'ㄷ':
                result.append('ㄷ_')
            elif 종성 in ['ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ']:
                result.append('ㄹ_')
            elif 종성 == 'ㅁ':
                result.append('ㅁ_')
            elif 종성 in ['ㅂ', 'ㅄ']:
                result.append('ㅂ_')
            elif 종성 in ['ㅅ', 'ㅆ']:
                result.append('ㅅ_')
            elif 종성 == 'ㅇ':
                result.append('ㅇ_')
            elif 종성 == 'ㅈ':
                result.append('ㅈ_')
            elif 종성 == 'ㅊ':
                result.append('ㅊ_')
            elif 종성 == 'ㅋ':
                result.append('ㅋ_')
            elif 종성 == 'ㅌ':
                result.append('ㅌ_')
            
        return result
    
    def indices_to_text(self, indices):
        """인덱스 리스트를 텍스트로 변환 (CTC 디코딩용)"""
        graphemes = []
        prev_idx = None
        
        for idx in indices:
            # blank 토큰 제거
            if idx == 0:
                prev_idx = None
                continue
                
            # 연속된 같은 토큰 제거 (CTC 규칙)
            if idx != prev_idx:
                graphemes.append(self.idx_to_grapheme[idx])
                
            prev_idx = idx
        
        # 자소를 음절로 재조합 (간단한 버전)
        return ''.join(graphemes)

# ===========================================
# 3. 데이터셋 클래스들
# ===========================================

class UnlabeledVideoDataset(Dataset):
    """사전학습용 무라벨 비디오 데이터셋"""
    
    def __init__(self, video_dir, num_clips_per_video=50):
        self.video_files = list(Path(video_dir).glob("*.mp4"))
        self.num_clips_per_video = num_clips_per_video
        self.lip_extractor = LipROIExtractor()
        
        if len(self.video_files) == 0:
            print("⚠️ 비디오 파일이 없어 더미 데이터를 사용합니다.")
            self.use_dummy = True
            self.dummy_size = 1000
        else:
            self.use_dummy = False
            print(f"📁 {len(self.video_files)}개 비디오 파일 발견")
    
    def __len__(self):
        if self.use_dummy:
            return self.dummy_size
        return len(self.video_files) * self.num_clips_per_video
    
    def __getitem__(self, idx):
        if self.use_dummy:
            # 더미 데이터
            sequence = np.random.rand(Config.PRETRAIN_SEQUENCE_LENGTH, 112, 112, 3).astype(np.float32)
            return torch.FloatTensor(sequence)
        
        # 실제 데이터
        video_idx = idx // self.num_clips_per_video
        video_file = self.video_files[video_idx]
        
        try:
            # 비디오 전체 길이 구하기
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            # 랜덤 시작 시간 선택
            clip_duration = Config.PRETRAIN_SEQUENCE_LENGTH / Config.TARGET_FPS
            if duration > clip_duration:
                start_time = random.uniform(0, duration - clip_duration)
                end_time = start_time + clip_duration
            else:
                start_time = 0
                end_time = duration
            
            # 입술 시퀀스 추출
            lip_sequence = self.lip_extractor.extract_video_lip_sequence(
                video_file, start_time, end_time
            )
            
            # 정확히 16프레임으로 맞추기
            if len(lip_sequence) >= Config.PRETRAIN_SEQUENCE_LENGTH:
                lip_sequence = lip_sequence[:Config.PRETRAIN_SEQUENCE_LENGTH]
            else:
                # 부족하면 마지막 프레임 반복
                padding = np.tile(
                    lip_sequence[-1:], 
                    (Config.PRETRAIN_SEQUENCE_LENGTH - len(lip_sequence), 1, 1, 1)
                )
                lip_sequence = np.concatenate([lip_sequence, padding], axis=0)
            
            return torch.FloatTensor(lip_sequence)
            
        except Exception as e:
            print(f"데이터 로딩 실패: {e}")
            # 실패 시 더미 데이터 반환
            sequence = np.random.rand(Config.PRETRAIN_SEQUENCE_LENGTH, 112, 112, 3).astype(np.float32)
            return torch.FloatTensor(sequence)

class LabeledVideoDataset(Dataset):
    """본학습용 라벨 비디오 데이터셋"""
    
    def __init__(self, video_path, json_path):
        self.video_path = Path(video_path)
        self.lip_extractor = LipROIExtractor()
        self.grapheme_processor = KoreanGraphemeProcessor()
        
        # 라벨 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.annotations = data['Sentence_info']
        
        print(f"📊 라벨 데이터: {len(self.annotations)}개 문장")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 시간 구간 정보
        start_time = annotation['start_time']
        end_time = annotation['end_time']
        
        # 자소 라벨 (이미 분해된 상태)
        sentence_graphemes = annotation['sentence_text']
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

def collate_fn(batch):
    """가변 길이 배치 처리"""
    frames_list = [item['frames'] for item in batch]
    graphemes_list = [item['graphemes'] for item in batch]
    frame_lengths = [item['frame_length'] for item in batch]
    grapheme_lengths = [item['grapheme_length'] for item in batch]
    texts = [item['text'] for item in batch]
    
    # 패딩
    max_frame_len = max(frame_lengths)
    max_grapheme_len = max(grapheme_lengths)
    
    padded_frames = []
    padded_graphemes = []
    
    for i, frames in enumerate(frames_list):
        # 프레임 패딩
        if len(frames) < max_frame_len:
            padding = torch.zeros(max_frame_len - len(frames), 112, 112, 3)
            frames = torch.cat([frames, padding], dim=0)
        padded_frames.append(frames)
        
        # 자소 패딩
        graphemes = graphemes_list[i]
        if len(graphemes) < max_grapheme_len:
            padding = torch.zeros(max_grapheme_len - len(graphemes), dtype=torch.long)
            graphemes = torch.cat([graphemes, padding], dim=0)
        padded_graphemes.append(graphemes)
    
    return {
        'frames': torch.stack(padded_frames),
        'graphemes': torch.stack(padded_graphemes),
        'frame_lengths': torch.LongTensor(frame_lengths),
        'grapheme_lengths': torch.LongTensor(grapheme_lengths),
        'texts': texts
    }

class CrawlerVideoDataset(Dataset):
    """크롤러로 생성된 비라벨링 비디오 데이터셋"""
    
    def __init__(self, pickle_path):
        self.pickle_path = Path(pickle_path)
        
        # Pickle 파일 로드
        with open(self.pickle_path, 'rb') as f:
            self.sequences = pickle.load(f)
        
        print(f"📊 크롤러 데이터: {len(self.sequences)}개 시퀀스")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        
        # 시퀀스 데이터 추출
        lip_sequence = sequence_data['sequence']  # (16, 112, 112, 3)
        
        # 정규화 (0-1 범위)
        if lip_sequence.dtype == np.uint8:
            lip_sequence = lip_sequence.astype(np.float32) / 255.0
        
        # 정확히 16프레임으로 맞추기
        if len(lip_sequence) >= Config.PRETRAIN_SEQUENCE_LENGTH:
            lip_sequence = lip_sequence[:Config.PRETRAIN_SEQUENCE_LENGTH]
        else:
            # 부족하면 마지막 프레임 반복
            padding = np.tile(
                lip_sequence[-1:], 
                (Config.PRETRAIN_SEQUENCE_LENGTH - len(lip_sequence), 1, 1, 1)
            )
            lip_sequence = np.concatenate([lip_sequence, padding], axis=0)
        
        return torch.FloatTensor(lip_sequence)

# ===========================================
# 4. 모델 아키텍처
# ===========================================

class VisualFrontend(nn.Module):
    """3D CNN + ResNet18 기반 시각적 특징 추출기"""
    
    def __init__(self):
        super().__init__()
        
        # 3D CNN for temporal features
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # ResNet18 backbone
        resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet_layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        """
        x: [B, T, H, W, C] = [batch, time, height, width, channels]
        return: [B, T, 512]
        """
        B, T, H, W, C = x.shape
        
        # 3D Conv 적용
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        x = self.conv3d(x)  # [B, 64, T, H', W']
        
        # 시간별로 ResNet 적용
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, 64, H', W']
        
        frame_features = []
        for t in range(T):
            frame = x[:, t]  # [B, 64, H', W']
            feat = self.resnet_layers(frame)  # [B, 512, 7, 7]
            feat = self.global_pool(feat)  # [B, 512, 1, 1]
            feat = feat.squeeze(-1).squeeze(-1)  # [B, 512]
            frame_features.append(feat)
        
        output = torch.stack(frame_features, dim=1)  # [B, T, 512]
        return output

class PositionalEncoding(nn.Module):
    """위치 인코딩"""
    
    def __init__(self, d_model, max_len=500):  # 200 -> 500으로 증가
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TemporalEncoder(nn.Module):
    """Transformer 기반 시간적 인코더"""
    
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, attention_mask=None):
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        return x

class MaskedVideoModel(nn.Module):
    """사전학습용 Masked Video Modeling"""
    
    def __init__(self):
        super().__init__()
        
        self.visual_frontend = VisualFrontend()
        self.temporal_encoder = TemporalEncoder()
        
        # 디코더 (마스킹된 프레임 복원)
        self.decoder = nn.Sequential(
            nn.Linear(Config.D_MODEL, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 112 * 112 * 3)
        )
        
        self.mask_ratio = 0.15
    
    def create_mask(self, batch_size, seq_length, device):
        """랜덤 마스킹 패턴 생성"""
        num_masked = int(seq_length * self.mask_ratio)
        
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(seq_length, dtype=torch.bool)
            masked_indices = torch.randperm(seq_length)[:num_masked]
            mask[masked_indices] = True
            masks.append(mask)
        
        return torch.stack(masks).to(device)
    
    def forward(self, lip_frames):
        B, T, H, W, C = lip_frames.shape
        device = lip_frames.device
        
        # 마스킹 패턴 생성
        mask = self.create_mask(B, T, device)
        
        # 마스킹된 프레임 생성
        masked_frames = lip_frames.clone()
        for b in range(B):
            masked_frames[b, mask[b]] = 0
        
        # 인코더 통과
        visual_features = self.visual_frontend(masked_frames)
        temporal_features = self.temporal_encoder(visual_features)
        
        # 마스킹된 부분만 디코딩하여 손실 계산
        total_loss = 0
        num_masked_total = 0
        
        for b in range(B):
            if mask[b].sum() > 0:
                masked_features = temporal_features[b, mask[b]]
                reconstructed = self.decoder(masked_features)
                reconstructed = reconstructed.view(-1, H, W, C)
                
                original = lip_frames[b, mask[b]]
                loss = F.mse_loss(reconstructed, original)
                
                total_loss += loss
                num_masked_total += mask[b].sum().item()
        
        avg_loss = total_loss / B if B > 0 else 0
        return avg_loss

class LipReadingModel(nn.Module):
    """본학습용 립리딩 모델"""
    
    def __init__(self, vocab_size=54):
        super().__init__()
        
        self.visual_frontend = VisualFrontend()
        self.temporal_encoder = TemporalEncoder()
        
        # CTC Head
        self.ctc_head = nn.Sequential(
            nn.Linear(Config.D_MODEL, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, vocab_size)
        )
    
    def forward(self, lip_frames, frame_lengths=None):
        B, T = lip_frames.shape[:2]
        
        # 시각적 특징 추출
        visual_features = self.visual_frontend(lip_frames)
        
        # 어텐션 마스크 생성 (패딩 부분 무시)
        attention_mask = None
        if frame_lengths is not None:
            attention_mask = torch.zeros(B, T, dtype=torch.bool, device=lip_frames.device)
            for i, length in enumerate(frame_lengths):
                if length < T:
                    attention_mask[i, length:] = True
        
        # 시간적 인코딩
        temporal_features = self.temporal_encoder(visual_features, attention_mask)
        
        # CTC 출력
        logits = self.ctc_head(temporal_features)
        
        return logits
    
    def load_pretrained_weights(self, pretrained_path):
        """사전학습된 가중치 로드"""
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 인코더 부분만 로드
        if 'visual_frontend' in checkpoint:
            self.visual_frontend.load_state_dict(checkpoint['visual_frontend'])
            print("✅ Visual frontend 가중치 로드 완료")
        
        if 'temporal_encoder' in checkpoint:
            self.temporal_encoder.load_state_dict(checkpoint['temporal_encoder'])
            print("✅ Temporal encoder 가중치 로드 완료")

# ===========================================
# 5. 훈련 클래스들
# ===========================================

class PretrainTrainer:
    """사전학습 트레이너"""
    
    def __init__(self, model, train_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000
        )
        
        self.train_losses = []
        
        # 저장 디렉토리
        self.save_dir = Path(Config.MODELS_DIR) / 'pretrain'
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="사전학습")
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch.to(self.device)
            
            # 순전파
            loss = self.model(frames)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def train(self, epochs):
        print(f"🚀 사전학습 시작: {epochs} 에포크")
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            
            # 모델 저장 (10 에포크마다)
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, train_loss)
        
        # 최종 인코더 가중치 저장
        self.save_encoder_weights()
        print("✅ 사전학습 완료!")
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
        print(f"✅ 체크포인트 저장: epoch {epoch}")
    
    def save_encoder_weights(self):
        """사전학습된 인코더 가중치만 저장"""
        encoder_weights = {
            'visual_frontend': self.model.visual_frontend.state_dict(),
            'temporal_encoder': self.model.temporal_encoder.state_dict()
        }
        torch.save(encoder_weights, self.save_dir / 'pretrained_encoder.pt')
        print("✅ 인코더 가중치 저장 완료")

class FinetuneTrainer:
    """본학습 트레이너"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE / 10,  # 사전학습보다 낮은 학습률
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=15, gamma=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # 저장 디렉토리
        self.save_dir = Path(Config.MODELS_DIR) / 'finetune'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 평가 메트릭
        self.grapheme_processor = KoreanGraphemeProcessor()
    
    def compute_ctc_loss(self, logits, targets, input_lengths, target_lengths):
        """CTC 손실 계산"""
        log_probs = F.log_softmax(logits.transpose(0, 1), dim=-1)
        
        loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=0,
            reduction='mean',
            zero_infinity=True
        )
        
        return loss
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="본학습")
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            graphemes = batch['graphemes'].to(self.device)
            frame_lengths = batch['frame_lengths'].to(self.device)
            grapheme_lengths = batch['grapheme_lengths'].to(self.device)
            
            # 순전파
            logits = self.model(frames, frame_lengths)
            
            # CTC 손실
            loss = self.compute_ctc_loss(logits, graphemes, frame_lengths, grapheme_lengths)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="검증"):
                frames = batch['frames'].to(self.device)
                graphemes = batch['graphemes'].to(self.device)
                frame_lengths = batch['frame_lengths'].to(self.device)
                grapheme_lengths = batch['grapheme_lengths'].to(self.device)
                
                logits = self.model(frames, frame_lengths)
                loss = self.compute_ctc_loss(logits, graphemes, frame_lengths, grapheme_lengths)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs):
        print(f"🎯 본학습 시작: {epochs} 에포크")
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # 학습률 스케줄링
            self.scheduler.step()
            
            # 최고 성능 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_best_model(epoch + 1, val_loss)
            
            # 정기 체크포인트
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, train_loss, val_loss)
        
        print("✅ 본학습 완료!")
    
    def save_checkpoint(self, epoch, train_loss, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def save_best_model(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, self.save_dir / 'best_model.pt')
        print(f"🏆 최고 성능 모델 저장: epoch {epoch}, val_loss {val_loss:.4f}")

# ===========================================
# 6. 평가 클래스
# ===========================================

class ModelEvaluator:
    """모델 평가기"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.grapheme_processor = KoreanGraphemeProcessor()
    
    def ctc_decode(self, logits, frame_lengths):
        """CTC 디코딩"""
        predictions = torch.argmax(logits, dim=-1)  # [B, T]
        
        decoded_texts = []
        
        for batch_idx in range(predictions.shape[0]):
            pred_sequence = predictions[batch_idx, :frame_lengths[batch_idx]].cpu().numpy()
            
            # CTC 디코딩
            decoded_indices = []
            prev_token = None
            
            for token in pred_sequence:
                if token == 0:  # blank
                    prev_token = None
                    continue
                
                if token != prev_token:
                    decoded_indices.append(token)
                
                prev_token = token
            
            # 인덱스를 텍스트로 변환
            decoded_text = self.grapheme_processor.indices_to_text(decoded_indices)
            decoded_texts.append(decoded_text)
        
        return decoded_texts
    
    def compute_metrics(self, predictions, ground_truths):
        """평가 지표 계산"""
        total_ger = 0  # Grapheme Error Rate
        total_cer = 0  # Character Error Rate
        
        for pred, gt in zip(predictions, ground_truths):
            # 자소 단위 오류율
            pred_graphemes = list(pred.replace(' ', ''))
            gt_graphemes = list(gt.replace(' ', ''))
            
            ger = self.edit_distance(pred_graphemes, gt_graphemes) / max(len(gt_graphemes), 1)
            total_ger += ger
            
            # 음절 단위 오류율
            cer = self.edit_distance(list(pred), list(gt)) / max(len(gt), 1)
            total_cer += cer
        
        avg_ger = total_ger / len(predictions)
        avg_cer = total_cer / len(predictions)
        
        return avg_ger, avg_cer
    
    def edit_distance(self, seq1, seq2):
        """편집 거리 계산"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def evaluate(self, dataloader):
        """모델 평가"""
        self.model.eval()
        
        all_predictions = []
        all_ground_truths = []
        
        print("🔍 모델 평가 중...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="평가"):
                frames = batch['frames'].to(self.device)
                frame_lengths = batch['frame_lengths'].to(self.device)
                texts = batch['texts']
                
                # 예측
                logits = self.model(frames, frame_lengths)
                predictions = self.ctc_decode(logits, frame_lengths)
                
                all_predictions.extend(predictions)
                all_ground_truths.extend(texts)
        
        # 메트릭 계산
        ger, cer = self.compute_metrics(all_predictions, all_ground_truths)
        
        print(f"\n📊 평가 결과:")
        print(f"GER (자소 오류율): {ger:.4f} ({ger*100:.2f}%)")
        print(f"CER (음절 오류율): {cer:.4f} ({cer*100:.2f}%)")
        
        # 샘플 결과 출력
        print(f"\n📝 예측 샘플:")
        for i in range(min(5, len(all_predictions))):
            print(f"정답: {all_ground_truths[i]}")
            print(f"예측: {all_predictions[i]}")
            print()
        
        return {
            'ger': ger,
            'cer': cer,
            'predictions': all_predictions,
            'ground_truths': all_ground_truths
        }

# ===========================================
# 7. 메인 파이프라인 실행기
# ===========================================

def setup_directories():
    """필요한 디렉토리 생성"""
    dirs = [
        Config.UNLABELED_VIDEOS_DIR,
        Config.PROCESSED_DIR,
        Config.MODELS_DIR,
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def pretrain_pipeline():
    """사전학습 파이프라인"""
    print("=" * 50)
    print("🎯 사전학습 시작")
    print("=" * 50)
    
    # 데이터셋 생성
    dataset = UnlabeledVideoDataset(Config.UNLABELED_VIDEOS_DIR)

    # === 크롭된 입술 이미지 3개 저장 ===
    import cv2
    import os
    os.makedirs("cropped_samples", exist_ok=True)
    for i in range(3):
        lip_img = dataset[i]
        # torch.Tensor라면 numpy로 변환
        if hasattr(lip_img, "numpy"):
            lip_img = lip_img.numpy()
        # (채널, H, W) → (H, W, 채널) 변환
        if len(lip_img.shape) == 3 and lip_img.shape[0] in [1, 3]:
            lip_img = lip_img.transpose(1, 2, 0)
        cv2.imwrite(f"cropped_samples/lip_sample_{i+1}.png", lip_img)
    # ===============================
    
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows에서 pickle 오류 방지
        pin_memory=True
    )
    
    print(f"📊 사전학습 데이터: {len(dataset)}개 클립")
    
    # 모델 및 트레이너 생성
    model = MaskedVideoModel()
    trainer = PretrainTrainer(model, dataloader, Config.DEVICE)
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ 모델 파라미터: {total_params:,}개")
    
    # 훈련 실행
    trainer.train(Config.PRETRAIN_EPOCHS)
    
    return trainer.save_dir / 'pretrained_encoder.pt'

def pretrain_with_crawler_data(pickle_path):
    """크롤러 데이터로 사전학습"""
    print("🚀 크롤러 데이터로 사전학습 시작")
    print("="*50)
    
    # 데이터셋 생성
    dataset = CrawlerVideoDataset(pickle_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    # 모델 초기화
    model = MaskedVideoModel().to(Config.DEVICE)
    
    # 훈련기 초기화
    trainer = PretrainTrainer(model, dataloader, Config.DEVICE)
    
    # 훈련 실행
    trainer.train(Config.PRETRAIN_EPOCHS)
    
    print("✅ 크롤러 데이터 사전학습 완료!")
    return model

def finetune_pipeline(pretrained_weights_path):
    """본학습 파이프라인"""
    print("=" * 50)
    print("🎯 본학습 시작")
    print("=" * 50)
    
    # 데이터셋 생성
    dataset = LabeledVideoDataset(Config.LABELED_VIDEO_PATH, Config.LABELS_JSON_PATH)
    
    # 훈련/검증 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE // 2,  # 본학습은 배치 크기 줄임
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows에서 pickle 오류 방지
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE // 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Windows에서 pickle 오류 방지
    )
    
    print(f"📊 본학습 데이터: 훈련 {len(train_dataset)}개, 검증 {len(val_dataset)}개")
    
    # 모델 생성 및 사전학습 가중치 로드
    model = LipReadingModel(Config.VOCAB_SIZE)
    model.load_pretrained_weights(pretrained_weights_path)
    
    # 트레이너 생성 및 훈련
    trainer = FinetuneTrainer(model, train_loader, val_loader, Config.DEVICE)
    trainer.train(Config.FINETUNE_EPOCHS)
    
    return trainer.save_dir / 'best_model.pt', val_loader

def evaluation_pipeline(model_path, val_loader):
    """평가 파이프라인"""
    print("=" * 50)
    print("🔍 모델 평가")
    print("=" * 50)
    
    # 모델 로드
    model = LipReadingModel(Config.VOCAB_SIZE)
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 평가 실행
    evaluator = ModelEvaluator(model, Config.DEVICE)
    results = evaluator.evaluate(val_loader)
    
    # 결과 저장
    with open('logs/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'ger': results['ger'],
            'cer': results['cer'],
            'sample_predictions': results['predictions'][:10],
            'sample_ground_truths': results['ground_truths'][:10]
        }, f, ensure_ascii=False, indent=2)
    
    return results

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='립리딩 모델 훈련')
    parser.add_argument('--mode', choices=['pretrain', 'finetune', 'evaluate', 'crawler_pretrain'], 
                       default='pretrain', help='훈련 모드')
    parser.add_argument('--crawler-data', type=str, help='크롤러 데이터 pickle 파일 경로')
    parser.add_argument('--pretrained-weights', type=str, help='사전학습된 가중치 경로')
    parser.add_argument('--model-path', type=str, help='평가할 모델 경로')
    
    args = parser.parse_args()
    
    # 디렉토리 설정
    setup_directories()
    
    if args.mode == 'crawler_pretrain':
        if not args.crawler_data:
            print("❌ 크롤러 데이터 파일 경로를 지정해주세요: --crawler-data")
            return
        pretrain_with_crawler_data(args.crawler_data)
    
    elif args.mode == 'pretrain':
        pretrain_pipeline()
    
    elif args.mode == 'finetune':
        if not args.pretrained_weights:
            print("❌ 사전학습된 가중치 경로를 지정해주세요: --pretrained-weights")
            return
        finetune_pipeline(args.pretrained_weights)
    
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("❌ 평가할 모델 경로를 지정해주세요: --model-path")
            return
        # 평가 파이프라인 구현 필요
        print("평가 기능은 아직 구현되지 않았습니다.")
    
    else:
        print("❌ 알 수 없는 모드입니다.")

if __name__ == "__main__":
    main()
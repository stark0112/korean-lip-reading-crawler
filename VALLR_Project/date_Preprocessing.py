#!/usr/bin/env python3
"""
VALLR 한국어 특화 전처리 파이프라인
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# 한국어 특화 모듈들
from src.korean_grapheme_processor import KoreanGraphemeProcessor
from src.korean_data_loader import KoreanLipReadingDataset, create_korean_dataloader

class KoreanVALLRPreprocessor:
    """한국어 VALLR 전처리기"""
    
    def __init__(self, 
                 data_root: str,
                 output_dir: str = "processed_data",
                 max_video_length: int = 768,  # VALLR 논문: ViT 최대 입력 길이
                 max_text_length: int = 300,
                 lip_size: Tuple[int, int] = (224, 224),  # VALLR 논문 기준
                 use_audio: bool = False,
                 use_visual: bool = True):
        """
        Args:
            data_root: 원본 데이터 루트 경로
            output_dir: 전처리된 데이터 저장 경로
            max_video_length: 최대 비디오 길이
            max_text_length: 최대 텍스트 길이
            lip_size: 립 이미지 크기
            use_audio: 오디오 사용 여부
            use_visual: 비주얼 사용 여부
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.max_video_length = max_video_length
        self.max_text_length = max_text_length
        self.lip_size = lip_size
        self.use_audio = use_audio
        self.use_visual = use_visual
        
        # 한국어 자소 처리기
        self.grapheme_processor = KoreanGraphemeProcessor()
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔤 한국어 VALLR 전처리기 초기화 완료")
        print(f"📁 데이터 루트: {self.data_root}")
        print(f"📁 출력 디렉토리: {self.output_dir}")
        print(f"📊 어휘 크기: {self.grapheme_processor.vocab_size}")
    
    def preprocess_dataset(self, split: str = 'train', batch_size: int = 1, max_retries: int = 100) -> Dict:
        """데이터셋 전처리 (한 개씩 바로 저장) - 자동 재시작 기능 포함"""
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"\n🔄 {split} 데이터셋 전처리 시작... (시도 {retry_count + 1}/{max_retries})")
                
                # 데이터셋 생성
                dataset = KoreanLipReadingDataset(
                    data_root=str(self.data_root),
                    split=split,
                    max_video_length=self.max_video_length,
                    max_text_length=self.max_text_length,
                    lip_size=self.lip_size,
                    use_audio=self.use_audio,
                    use_visual=self.use_visual
                )
                
                total_samples = len(dataset)
                print(f"📊 전체 샘플 수: {total_samples}")
                print(f"📦 처리 방식: 한 개씩 바로 저장")
                
                # 이미 처리된 샘플 확인
                processed_samples = set()
                samples_dir = self.output_dir / f"{split}_samples"
                if samples_dir.exists():
                    for sample_dir in samples_dir.iterdir():
                        if sample_dir.is_dir() and sample_dir.name.startswith("sample_"):
                            try:
                                sample_idx = int(sample_dir.name.split("_")[1])
                                # 완전히 처리된 샘플인지 확인 (video_frames.npy와 text_indices.npy가 모두 존재)
                                video_file = sample_dir / "video_frames.npy"
                                text_file = sample_dir / "text_indices.npy"
                                if video_file.exists() and text_file.exists():
                                    processed_samples.add(sample_idx)
                            except ValueError:
                                continue
                
                if processed_samples:
                    print(f"🔄 이미 처리된 샘플 {len(processed_samples)}개 발견")
                    print(f"🔄 {len(processed_samples)}번째 샘플부터 재시작합니다")
                    start_idx = max(processed_samples) + 1
                else:
                    start_idx = 0
                    print(f"🆕 처음부터 시작합니다")
                
                # 전체 통계 정보
                all_stats = {
                    'total_samples': total_samples,
                    'vocab_size': dataset.grapheme_processor.vocab_size,
                    'max_video_length': self.max_video_length,
                    'max_text_length': self.max_text_length,
                    'lip_size': self.lip_size,
                    'use_audio': self.use_audio,
                    'use_visual': self.use_visual,
                    'person_types': {},
                    'genders': {},
                    'environments': {},
                    'text_lengths': [],
                    'video_lengths': [],
                    'processed_samples': len(processed_samples),
                    'failed_samples': 0,
                    'resumed_from': start_idx,
                    'retry_count': retry_count
                }
                
                # 한 개씩 처리하고 바로 저장
                for i in range(start_idx, total_samples):
                    try:
                        print(f"\n📹 샘플 {i+1}/{total_samples} 처리 중...")
                        
                        # 이미 처리된 샘플인지 확인
                        if i in processed_samples:
                            print(f"  ⏭️ 샘플 {i+1} 이미 처리됨 - 건너뛰기")
                            continue
                        
                        # 샘플 로드
                        sample = dataset[i]
                        
                        # 통계 정보 수집
                        person_type = sample['person_type']
                        gender = sample['gender']
                        environment = sample['environment']
                        
                        all_stats['person_types'][person_type] = all_stats['person_types'].get(person_type, 0) + 1
                        all_stats['genders'][gender] = all_stats['genders'].get(gender, 0) + 1
                        all_stats['environments'][environment] = all_stats['environments'].get(environment, 0) + 1
                        
                        all_stats['text_lengths'].append(sample['text_length'].item())
                        all_stats['video_lengths'].append(sample['video_length'].item())
                        
                        # 샘플 정보 바로 저장
                        self._save_single_sample(sample, i, split)
                        
                        all_stats['processed_samples'] += 1
                        
                        print(f"  ✅ 샘플 {i+1} 저장 완료: 비디오 {sample['video_frames'].shape}, 텍스트 길이 {sample['text_length'].item()}")
                        
                        # 10개마다 중간 통계 출력
                        if (i + 1) % 10 == 0:
                            print(f"  📊 진행률: {i+1}/{total_samples} ({((i+1)/total_samples*100):.1f}%)")
                            print(f"  📈 성공: {all_stats['processed_samples']}, 실패: {all_stats['failed_samples']}")
                        
                    except Exception as e:
                        print(f"  ❌ 샘플 {i+1} 처리 실패: {e}")
                        all_stats['failed_samples'] += 1
                        continue
                
                # 전체 통계 계산
                if all_stats['text_lengths']:
                    all_stats['avg_text_length'] = np.mean(all_stats['text_lengths'])
                    all_stats['max_text_length_actual'] = max(all_stats['text_lengths'])
                if all_stats['video_lengths']:
                    all_stats['avg_video_length'] = np.mean(all_stats['video_lengths'])
                    all_stats['max_video_length_actual'] = max(all_stats['video_lengths'])
                
                # 전체 결과 저장
                self._save_processed_data(dataset, split, all_stats)
                
                print(f"\n✅ {split} 데이터셋 전처리 완료")
                print(f"📊 성공: {all_stats['processed_samples']}, 실패: {all_stats['failed_samples']}")
                if start_idx > 0:
                    print(f"🔄 재시작 지점: {start_idx}번째 샘플")
                
                return all_stats
                
            except KeyboardInterrupt:
                print(f"\n⚠️ 사용자에 의해 중단됨 (시도 {retry_count + 1}/{max_retries})")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"🔄 5초 후 자동 재시작...")
                    import time
                    time.sleep(5)
                    continue
                else:
                    print(f"❌ 최대 재시작 횟수({max_retries})에 도달했습니다.")
                    raise
                    
            except Exception as e:
                print(f"\n❌ 오류 발생 (시도 {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"🔄 10초 후 자동 재시작...")
                    import time
                    time.sleep(10)
                    continue
                else:
                    print(f"❌ 최대 재시작 횟수({max_retries})에 도달했습니다.")
                    raise
        
        print(f"❌ 모든 재시작 시도 실패")
        return {}
    
    def _save_single_sample(self, sample: Dict, sample_idx: int, split: str):
        """단일 샘플 정보 저장 (VALLR 모델용 numpy 배열)"""
        # 샘플별 디렉토리 생성
        sample_dir = self.output_dir / f"{split}_samples" / f"sample_{sample_idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 샘플 정보 저장
        sample_info = {
            'sample_idx': sample_idx,
            'video_frames_shape': list(sample['video_frames'].shape),
            'text_indices_shape': list(sample['text_indices'].shape),
            'text_length': sample['text_length'].item(),
            'video_length': sample['video_length'].item(),
            'original_text': sample['original_text'],
            'sentence_id': sample['sentence_id'],
            'start_time': sample['start_time'],
            'end_time': sample['end_time'],
            'person_type': sample['person_type'],
            'gender': sample['gender'],
            'environment': sample['environment']
        }
        
        # 샘플 정보 파일 저장
        info_file = sample_dir / "sample_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(sample_info, f, ensure_ascii=False, indent=2)
        
        # VALLR 모델용 데이터 저장
        import numpy as np
        
        # 1. 비디오 프레임 저장 (VALLR 입력용) - 논문과 동일한 순서
        video_frames = sample['video_frames'].numpy()  # [T, H, W, C]
        
        # VALLR 논문 형식: (T, C, H, W) 순서로 변환
        video_frames_tchw = np.transpose(video_frames, (0, 3, 1, 2))  # [T, C, H, W]
        
        video_file = sample_dir / "video_frames.npy"
        np.save(video_file, video_frames_tchw)
        
        # 2. 텍스트 인덱스 저장 (VALLR 라벨용)
        text_file = sample_dir / "text_indices.npy"
        np.save(text_file, sample['text_indices'].numpy())
        
        # 3. 길이 정보 저장
        length_file = sample_dir / "lengths.json"
        with open(length_file, 'w', encoding='utf-8') as f:
            json.dump({
                'text_length': sample['text_length'].item(),
                'video_length': sample['video_length'].item(),
                'num_frames': video_frames.shape[0],
                'tensor_shape': list(video_frames_tchw.shape),  # VALLR 형식
                'original_shape': list(video_frames.shape)      # 원본 형식
            }, f, ensure_ascii=False, indent=2)
        
        print(f"    💾 비디오 프레임 저장: {video_frames_tchw.shape} (VALLR 형식: TCHW)")
        print(f"    💾 텍스트 라벨 저장: {sample['text_indices'].shape}")
    
    def _process_batch(self, dataset: KoreanLipReadingDataset, start_idx: int, end_idx: int, batch_idx: int) -> Dict:
        """배치 단위 처리"""
        stats = {
            'batch_idx': batch_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'sample_count': end_idx - start_idx,
            'person_types': {},
            'genders': {},
            'environments': {},
            'text_lengths': [],
            'video_lengths': []
        }
        
        for i in range(start_idx, end_idx):
            try:
                sample = dataset[i]
                
                # 인구통계학적 정보
                person_type = sample['person_type']
                gender = sample['gender']
                environment = sample['environment']
                
                stats['person_types'][person_type] = stats['person_types'].get(person_type, 0) + 1
                stats['genders'][gender] = stats['genders'].get(gender, 0) + 1
                stats['environments'][environment] = stats['environments'].get(environment, 0) + 1
                
                # 길이 정보
                stats['text_lengths'].append(sample['text_length'].item())
                stats['video_lengths'].append(sample['video_length'].item())
                
                if (i - start_idx + 1) % 100 == 0:
                    print(f"    📊 {i - start_idx + 1}/{end_idx - start_idx} 샘플 처리 완료")
                
            except Exception as e:
                print(f"    ⚠️ 샘플 {i} 처리 중 오류: {e}")
                continue
        
        return stats
    
    def _save_batch_data(self, dataset: KoreanLipReadingDataset, start_idx: int, end_idx: int, batch_idx: int, split: str):
        """배치 데이터 저장"""
        batch_dir = self.output_dir / f"{split}_batch_{batch_idx:03d}"
        batch_dir.mkdir(exist_ok=True)
        
        # 배치 정보 저장
        batch_info = {
            'batch_idx': batch_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'sample_count': end_idx - start_idx,
            'samples': []
        }
        
        # 샘플 정보 수집 (처음 10개만)
        for i in range(start_idx, min(start_idx + 10, end_idx)):
            try:
                sample = dataset[i]
                sample_info = {
                    'index': i,
                    'video_frames_shape': list(sample['video_frames'].shape),
                    'text_indices_shape': list(sample['text_indices'].shape),
                    'text_length': sample['text_length'].item(),
                    'video_length': sample['video_length'].item(),
                    'original_text': sample['original_text'],
                    'sentence_id': sample['sentence_id'],
                    'start_time': sample['start_time'],
                    'end_time': sample['end_time'],
                    'person_type': sample['person_type'],
                    'gender': sample['gender'],
                    'environment': sample['environment']
                }
                batch_info['samples'].append(sample_info)
            except Exception as e:
                print(f"    ⚠️ 샘플 {i} 정보 저장 중 오류: {e}")
                continue
        
        # 배치 정보 저장
        batch_file = batch_dir / "batch_info.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, ensure_ascii=False, indent=2)
        
        print(f"    💾 배치 {batch_idx} 정보 저장 완료: {batch_file}")
    
    def _collect_statistics(self, dataset: KoreanLipReadingDataset) -> Dict:
        """데이터셋 통계 정보 수집"""
        print("📊 통계 정보 수집 중...")
        
        stats = {
            'total_samples': len(dataset),
            'vocab_size': dataset.grapheme_processor.vocab_size,
            'max_video_length': self.max_video_length,
            'max_text_length': self.max_text_length,
            'lip_size': self.lip_size,
            'use_audio': self.use_audio,
            'use_visual': self.use_visual,
            'person_types': {},
            'genders': {},
            'environments': {},
            'text_lengths': [],
            'video_lengths': []
        }
        
        # 샘플별 통계 수집
        for i in range(min(100, len(dataset))):  # 처음 100개 샘플만 분석
            try:
                sample = dataset[i]
                
                # 인구통계학적 정보
                person_type = sample['person_type']
                gender = sample['gender']
                environment = sample['environment']
                
                stats['person_types'][person_type] = stats['person_types'].get(person_type, 0) + 1
                stats['genders'][gender] = stats['genders'].get(gender, 0) + 1
                stats['environments'][environment] = stats['environments'].get(environment, 0) + 1
                
                # 길이 정보
                stats['text_lengths'].append(sample['text_length'].item())
                stats['video_lengths'].append(sample['video_length'].item())
                
            except Exception as e:
                print(f"⚠️ 샘플 {i} 처리 중 오류: {e}")
                continue
        
        # 평균 길이 계산
        if stats['text_lengths']:
            stats['avg_text_length'] = np.mean(stats['text_lengths'])
            stats['max_text_length_actual'] = max(stats['text_lengths'])
        if stats['video_lengths']:
            stats['avg_video_length'] = np.mean(stats['video_lengths'])
            stats['max_video_length_actual'] = max(stats['video_lengths'])
        
        return stats
    
    def _save_processed_data(self, dataset: KoreanLipReadingDataset, split: str, stats: Dict):
        """전처리된 데이터 저장"""
        print("💾 전처리된 데이터 저장 중...")
        
        # 통계 정보 저장
        stats_file = self.output_dir / f"{split}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"📊 통계 정보 저장 완료: {stats_file}")
        
        # 어휘 정보 저장
        vocab_info = dataset.grapheme_processor.get_vocab_info()
        vocab_file = self.output_dir / f"{split}_vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)
        print(f"🔤 어휘 정보 저장 완료: {vocab_file}")
        
        # 데이터 매핑 정보 저장
        mapping_file = self.output_dir / f"{split}_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(dataset.data_mapping, f, ensure_ascii=False, indent=2)
        print(f"🗂️ 데이터 매핑 정보 저장 완료: {mapping_file}")
        
        # 실제 전처리된 데이터 샘플 저장 (처음 10개)
        print("🎬 실제 데이터 샘플 저장 중...")
        sample_data = []
        for i in range(min(10, len(dataset))):
            try:
                sample = dataset[i]
                sample_info = {
                    'index': i,
                    'video_frames_shape': list(sample['video_frames'].shape),
                    'text_indices_shape': list(sample['text_indices'].shape),
                    'text_length': sample['text_length'].item(),
                    'video_length': sample['video_length'].item(),
                    'original_text': sample['original_text'],
                    'sentence_id': sample['sentence_id'],
                    'start_time': sample['start_time'],
                    'end_time': sample['end_time'],
                    'person_type': sample['person_type'],
                    'gender': sample['gender'],
                    'environment': sample['environment']
                }
                sample_data.append(sample_info)
                print(f"  📹 샘플 {i}: 비디오 {sample['video_frames'].shape}, 텍스트 길이 {sample['text_length'].item()}")
            except Exception as e:
                print(f"  ⚠️ 샘플 {i} 저장 중 오류: {e}")
                continue
        
        # 샘플 정보 저장
        sample_file = self.output_dir / f"{split}_samples.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"📋 샘플 정보 저장 완료: {sample_file}")
        
        print(f"✅ 전처리 데이터 저장 완료: {self.output_dir}")
        print(f"📁 생성된 파일들:")
        for file in self.output_dir.glob(f"{split}_*"):
            print(f"  - {file.name}")
    
    def create_dataloader(self, split: str = 'train', batch_size: int = 8, num_workers: int = 4) -> DataLoader:
        """데이터로더 생성"""
        return create_korean_dataloader(
            data_root=str(self.data_root),
            batch_size=batch_size,
            split=split,
            num_workers=num_workers,
            max_video_length=self.max_video_length,
            max_text_length=self.max_text_length,
            lip_size=self.lip_size,
            use_audio=self.use_audio,
            use_visual=self.use_visual
        )
    
    def test_preprocessing(self):
        """전처리 테스트"""
        print("\n🧪 전처리 테스트 시작...")
        
        try:
            # 데이터셋 직접 생성 (배치 처리 없이)
            dataset = KoreanLipReadingDataset(
                data_root=str(self.data_root),
                split='train',
                max_video_length=self.max_video_length,
                max_text_length=self.max_text_length,
                lip_size=self.lip_size,
                use_audio=self.use_audio,
                use_visual=self.use_visual
            )
            
            print(f"📊 데이터셋 크기: {len(dataset)}")
            
            # 첫 번째 샘플 로드
            sample = dataset[0]
            print("✅ 샘플 로드 성공!")
            print(f"📹 비디오 프레임: {sample['video_frames'].shape}")
            print(f"📝 텍스트 인덱스: {sample['text_indices'].shape}")
            print(f"📏 텍스트 길이: {sample['text_length']}")
            print(f"🎬 비디오 길이: {sample['video_length']}")
            print(f"👤 원본 텍스트: {sample['original_text']}")
            print(f"🆔 문장 ID: {sample['sentence_id']}")
            print(f"⏰ 시작 시간: {sample['start_time']}")
            print(f"⏰ 종료 시간: {sample['end_time']}")
            
            if self.use_audio and 'mel_spectrogram' in sample:
                print(f"🎵 멜스펙트로그램: {sample['mel_spectrogram'].shape}")
            
            # 두 번째 샘플도 확인 (다른 길이)
            sample2 = dataset[1]
            print(f"\n📹 두 번째 샘플 비디오 프레임: {sample2['video_frames'].shape}")
            print(f"📝 두 번째 샘플 텍스트 인덱스: {sample2['text_indices'].shape}")
            print(f"👤 두 번째 샘플 원본 텍스트: {sample2['original_text']}")
                
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            return False
        
        return True

def main():
    """메인 함수"""
    import sys
    
    # 명령행 인수 처리
    if len(sys.argv) > 1:
        # 명령행 인수가 있으면 argparse 사용
        parser = argparse.ArgumentParser(description="한국어 VALLR 전처리")
        parser.add_argument("--data_root", type=str, default="009.립리딩(입모양) 음성인식 데이터",
                           help="데이터 루트 경로")
        parser.add_argument("--output_dir", type=str, default="processed_data",
                           help="출력 디렉토리")
        parser.add_argument("--max_video_length", type=int, default=768,
                           help="최대 비디오 길이 (VALLR 논문: ViT 최대 입력 길이)")
        parser.add_argument("--max_text_length", type=int, default=300,
                           help="최대 텍스트 길이")
        parser.add_argument("--lip_size", type=int, nargs=2, default=[224, 224],
                           help="립 이미지 크기 (VALLR 논문: 224x224)")
        parser.add_argument("--use_audio", action="store_true", default=False,
                           help="오디오 사용 (VALLR은 비디오만 사용)")
        parser.add_argument("--use_visual", action="store_true", default=True,
                           help="비주얼 사용")
        parser.add_argument("--test_only", action="store_true",
                           help="테스트만 실행")
        parser.add_argument("--batch_size", type=int, default=1,
                           help="배치 크기 (메모리 사용량 조절용, 1=한개씩 처리)")
        
        args = parser.parse_args()
    else:
        # 기본값으로 실행
        class Args:
            data_root = "009.립리딩(입모양) 음성인식 데이터"
            output_dir = "processed_data"
            max_video_length = 768
            max_text_length = 300
            lip_size = [224, 224]
            use_audio = False
            use_visual = True
            test_only = False
            batch_size = 1
        
        args = Args()
    
    try:
        print("🚀 한국어 VALLR 전처리 시작...")
        print(f"📁 데이터 루트: {args.data_root}")
        print(f"📁 출력 디렉토리: {args.output_dir}")
        
        # 전처리기 초기화
        preprocessor = KoreanVALLRPreprocessor(
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_video_length=args.max_video_length,
            max_text_length=args.max_text_length,
            lip_size=tuple(args.lip_size),
            use_audio=args.use_audio,
            use_visual=args.use_visual
        )
        
        if args.test_only:
            # 테스트만 실행
            print("\n🧪 전처리 테스트 실행...")
            success = preprocessor.test_preprocessing()
            if success:
                print("🎉 전처리 테스트 성공!")
            else:
                print("❌ 전처리 테스트 실패!")
                sys.exit(1)
        else:
            # 전체 전처리 실행
            print("\n🔄 훈련 데이터 전처리 시작...")
            train_stats = preprocessor.preprocess_dataset('train', args.batch_size)
            print(f"✅ 훈련 데이터 전처리 완료: {train_stats['total_samples']}개 샘플")
            
            # 검증 데이터 전처리 (선택적)
            try:
                print("\n🔄 검증 데이터 전처리 시작...")
                val_stats = preprocessor.preprocess_dataset('val', args.batch_size)
                print(f"✅ 검증 데이터 전처리 완료: {val_stats['total_samples']}개 샘플")
            except Exception as e:
                print(f"⚠️ 검증 데이터 전처리 건너뜀: {e}")
            
            # 테스트 실행
            print("\n🧪 최종 테스트 실행...")
            preprocessor.test_preprocessing()
            
            print("\n🎉 전처리 완료!")
            print(f"📁 결과 파일들:")
            for file in preprocessor.output_dir.glob("*"):
                print(f"  - {file.name}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
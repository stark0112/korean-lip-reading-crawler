#!/usr/bin/env python3
"""
한 개씩 처리하는 간단한 전처리 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from date_Preprocessing import KoreanVALLRPreprocessor

def main():
    print("🚀 한 개씩 처리하는 전처리 시작...")
    
    # 전처리기 초기화
    preprocessor = KoreanVALLRPreprocessor(
        data_root="009.립리딩(입모양) 음성인식 데이터",
        output_dir="processed_data",
        max_video_length=768,
        max_text_length=100,
        lip_size=(224, 224),
        use_audio=False,
        use_visual=True
    )
    
    # 한 개씩 처리
    print("📦 배치 크기: 1 (한 개씩 처리)")
    train_stats = preprocessor.preprocess_dataset('train', batch_size=1)
    
    print(f"✅ 전처리 완료! 총 {train_stats['total_samples']}개 샘플 처리됨")
    
    # 결과 확인
    print("\n📁 생성된 파일들:")
    for file in preprocessor.output_dir.glob("*"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main() 
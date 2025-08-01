#!/bin/bash

# 한국어 VALLR 전처리 실행 스크립트

echo "🚀 한국어 VALLR 전처리 시작..."
echo "📅 시작 시간: $(date)"

# Python 환경 확인
if ! command -v python &> /dev/null; then
    echo "❌ Python이 설치되지 않았습니다."
    exit 1
fi

# 필요한 디렉토리 확인
if [ ! -d "009.립리딩(입모양) 음성인식 데이터" ]; then
    echo "❌ 데이터 디렉토리를 찾을 수 없습니다."
    exit 1
fi

# processed_data 디렉토리 생성
mkdir -p processed_data

echo "🔧 전처리 실행 중..."
python date_Preprocessing.py

if [ $? -eq 0 ]; then
    echo "✅ 전처리 완료!"
    echo "📁 생성된 파일들:"
    ls -la processed_data/
    echo "📅 완료 시간: $(date)"
else
    echo "❌ 전처리 실패!"
    exit 1
fi 
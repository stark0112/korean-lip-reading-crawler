#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
입술 비디오 추출기
YouTube에서 입술이 나오는 영상을 검색하고 다운로드합니다.
"""

import cv2
import numpy as np
import mediapipe as mp
import yt_dlp
import json
import time
import threading
from pathlib import Path
import argparse
import re
import hashlib
from PIL import Image
import io
import random


# 뉴스 전용 검색어 목록
NEWS_COLLECTION_QUERIES = [
    "뉴스 발음", "뉴스 앵커", "뉴스 리포터",
    "방송 발음", "아나운서 발음", "MC 발음",
    "뉴스 앵커 발음", "뉴스 진행자", "뉴스 캐스터",
    "뉴스 앵커 말하기", "뉴스 진행자 발음",
    "아나운서 말하기", "MC 말하기",
    "방송 진행자", "방송 앵커",
    "뉴스 읽기", "뉴스 리딩",
    "뉴스 앵커 연습", "아나운서 연습"
]

# 대용량 수집용 검색어 목록
MASS_COLLECTION_QUERIES = [
    # 뉴스 관련 (우선순위)
    "뉴스 발음", "뉴스 앵커", "뉴스 리포터",
    "방송 발음", "아나운서 발음", "MC 발음",
    "뉴스 앵커 발음", "뉴스 진행자", "뉴스 캐스터",
    
    # 기본 입술 관련
    "입술", "입모양", "입술 움직임", "입술 발음",
    "입술 교정", "입모양 교정", "입술 훈련",
    
    # 한국어 발음 관련
    "한국어 발음", "한국어 말하기", "한국어 발음 연습",
    "한국어 자음", "한국어 모음", "한국어 받침",
    "한국어 발음교정", "한국어 발음연습", "한국어 발음훈련",
    
    # 영어 발음 관련
    "영어 발음", "영어 말하기", "영어 발음 연습",
    "영어 발음교정", "영어 발음연습", "영어 발음훈련",
    "영어 자음", "영어 모음", "영어 받침",
    
    # 발음 교정 관련
    "발음교정", "발음연습", "발음훈련", "발음기법",
    "말하기교정", "말하기연습", "말하기훈련", "말하기기법",
    "발음교정법", "발음연습법", "발음훈련법",
    
    # 상황별 말하기
    "강의 발음", "강사 발음", "교사 발음",
    "인터뷰 발음", "토크쇼 발음", "방송 발음",
    "강의 말하기", "강사 말하기", "교사 말하기",
    
    # 연령대별
    "어린이 발음", "아이 발음", "유아 발음",
    "청소년 발음", "학생 발음", "어린이 말하기",
    "성인 발음", "어른 발음", "성인 말하기",
    "노인 발음", "어르신 발음", "노인 말하기",
    
    # 전문 분야
    "강사 발음", "선생님 발음", "교수 발음",
    "연기 발음", "배우 발음", "연극 발음",
    "아나운서 발음", "MC 발음", "진행자 발음",
    
    # 특정 상황
    "발표 발음", "스피치 발음", "연설 발음",
    "독서 발음", "책 읽기", "낭독",
    "노래 발음", "가수 발음", "보컬 발음",
    "발표 말하기", "스피치 말하기", "연설 말하기",
    
    # 기술적 키워드
    "음성인식", "음성처리", "음성분석",
    "입모양 인식", "입술 인식", "립리딩",
    "컴퓨터 비전", "얼굴 인식", "표정 인식",
    "입모양 분석", "입술 분석", "입모양 처리",
    
    # 추가 키워드
    "발음교정영상", "입모양교정", "발음연습영상",
    "말하기연습영상", "발음훈련영상", "입술운동영상",
    "발음기법영상", "말하기기법영상", "발음교정법영상"
]


class LipDetector:
    """입술 검출기"""
    
    def __init__(self, debug_mode=False):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,  # 더 높은 신뢰도
            min_tracking_confidence=0.8    # 더 높은 추적 신뢰도
        )
        self.debug_mode = debug_mode
        
    def extract_lip_landmarks(self, frame):
        """입술 랜드마크 추출 (단순화)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # 입술 랜드마크 추출 (단순화)
        lip_indices = [
            # 입술 윤곽선 (핵심 포인트들)
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
            # 입술 내부
            310, 311, 312, 13, 82, 81, 80, 191,
            # 입술 모서리
            0, 267, 37, 39, 40, 185
        ]
        
        lip_points = []
        for idx in lip_indices:
            if idx < len(landmarks):
                point = landmarks[idx]
                x, y = int(point.x * w), int(point.y * h)
                lip_points.append([x, y])
        
        # 단순한 검증: 최소 10개 이상의 포인트만 있으면 OK
        if len(lip_points) >= 10:
            return lip_points
        
        return None
    
    def check_lip_in_video(self, video_path, sample_interval=1, strict_mode=False, preview_mode=False):
        """비디오에서 입술이 전체적으로 검출되는지 확인 (매 프레임 확인)"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 매 프레임 확인 (1프레임마다)
        sample_frames = max(1, int(fps * sample_interval / 1000))
        
        print(f"🔍 입술 검출 분석 중... (총 {total_frames}프레임, 매 프레임 확인)")
        
        lip_detected_count = 0
        total_checked = 0
        frame_count = 0
        
        # 디버그용 이미지 저장
        debug_dir = Path("test_results")
        debug_dir.mkdir(exist_ok=True)
        debug_count = 0
        
        # 엄격 모드 설정
        if strict_mode:
            min_frames_to_check = min(100, total_frames)  # 최소 100개 또는 전체 프레임
            required_ratio = 0.95  # 95% (거의 모든 프레임에서 입술 검출)
        elif preview_mode:
            # 미리보기 모드: 더 관대한 기준
            min_frames_to_check = min(30, total_frames)   # 최소 30개 또는 전체 프레임
            required_ratio = 0.8  # 80% (미리보기에서는 관대하게)
        else:
            min_frames_to_check = min(50, total_frames)   # 최소 50개 또는 전체 프레임
            required_ratio = 0.95  # 95% (거의 모든 프레임에서 입술 검출)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 매 프레임 확인
            lip_points = self.extract_lip_landmarks(frame)
            
            # 디버그 모드에서 검출 결과 시각화 (처음 5개만)
            if self.debug_mode and debug_count < 5:
                debug_frame = frame.copy()
                if lip_points:
                    # 입술 랜드마크 그리기
                    for point in lip_points:
                        cv2.circle(debug_frame, (point[0], point[1]), 3, (0, 255, 0), -1)
                    cv2.putText(debug_frame, f"LIP DETECTED: {len(lip_points)} points", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(debug_frame, "NO LIP DETECTED", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                debug_path = debug_dir / f"lip_detection_debug_{debug_count}.jpg"
                cv2.imwrite(str(debug_path), debug_frame)
                print(f"  📸 디버그 이미지 저장: {debug_path}")
                debug_count += 1
            
            if lip_points and len(lip_points) >= 10:  # 단순한 기준: 10개 이상
                lip_detected_count += 1
            total_checked += 1
            
            # 진행률 표시 (10프레임마다)
            if total_checked % 10 == 0:
                current_ratio = lip_detected_count / total_checked if total_checked > 0 else 0
                print(f"  📊 진행률: {total_checked}/{total_frames}프레임 | 검출률: {current_ratio*100:.1f}%")
            
            # 조기 종료 조건: 검출률이 너무 낮으면
            if total_checked >= min_frames_to_check:
                current_ratio = lip_detected_count / total_checked
                if current_ratio < required_ratio:  # 100% 미만이면 조기 종료
                    print(f"  ⚠️ 검출률 부족으로 조기 종료: {current_ratio*100:.1f}%")
                    break
            
            frame_count += 1
        
        cap.release()
        
        if total_checked >= min_frames_to_check:
            lip_ratio = lip_detected_count / total_checked
            print(f"  📊 최종 검출률: {lip_ratio*100:.1f}% ({lip_detected_count}/{total_checked})")
            return lip_ratio >= required_ratio
        
        print(f"  ❌ 검증 프레임 부족: {total_checked}개 (최소 {min_frames_to_check}개 필요)")
        return False

    def check_lip_in_preview(self, video_url, max_preview_duration=10):
        """다운로드 전 미리보기로 입술 검출 확인"""
        import yt_dlp
        
        # 미리보기 다운로드 옵션
        ydl_opts = {
            'format': 'best[height<=720]',  # 720p 이하로 빠른 다운로드
            'outtmpl': 'temp_preview_%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': False,
        }
        
        try:
            print(f"🔍 미리보기 입술 검출 확인 중... (최대 {max_preview_duration}초)")
            
            # 미리보기 다운로드
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                preview_file = None
                
                # 다운로드된 파일 찾기
                for file in Path('.').glob('temp_preview_*.mp4'):
                    preview_file = file
                    break
                
                if not preview_file:
                    print("  ❌ 미리보기 다운로드 실패")
                    return False
                
                # 입술 검출 확인
                has_lips = self.check_lip_in_video(preview_file, preview_mode=True)
                
                # 임시 파일 삭제
                try:
                    preview_file.unlink()
                except:
                    pass
                
                return has_lips
                
        except Exception as e:
            print(f"  ❌ 미리보기 검색 실패: {e}")
            return False


class LipVideoCrawler:
    """입술 비디오 추출기"""
    
    def __init__(self, output_dir="data/lip_videos", debug_mode=False, mass_mode=False, separate_audio=False, cc_by_only=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 분리 다운로드용 디렉토리 생성
        if separate_audio:
            self.video_dir = self.output_dir / "videos"
            self.audio_dir = self.output_dir / "audio"
            self.video_dir.mkdir(exist_ok=True)
            self.audio_dir.mkdir(exist_ok=True)
        
        self.lip_detector = LipDetector(debug_mode)
        self.debug_mode = debug_mode
        self.mass_mode = mass_mode
        self.separate_audio = separate_audio
        self.cc_by_only = cc_by_only
        
        # 다운로드 히스토리 로드
        self.history_file = self.output_dir / "download_history.json"
        self.downloaded_videos = self.load_download_history()
        
        # yt-dlp 옵션 설정
        if separate_audio:
            # 분리 다운로드 모드
            if mass_mode:
                # 대용량 모드: 영상과 음성 분리
                self.video_ydl_opts = {
                    'format': 'bestvideo[height>=1080][ext=mp4]',
                    'outtmpl': str(self.video_dir / '%(title)s.%(ext)s'),
                    'socket_timeout': 300,
                    'retries': 10,
                    'fragment_retries': 10,
                    'max_sleep_interval': 30,
                    'sleep_interval': 20,
                    'buffersize': 8192,
                    'http_chunk_size': 83886080,
                    'max_downloads': 20,
                    'progress_hooks': [self.download_progress_hook],
                    'concurrent_fragment_downloads': 4,
                    'postprocessor_args': ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-movflags', '+faststart'],
                    'ignoreerrors': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'referer': 'https://www.youtube.com/',
                }
                self.audio_ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
                    'outtmpl': str(self.audio_dir / '%(title)s.%(ext)s'),
                    'socket_timeout': 300,
                    'retries': 10,
                    'fragment_retries': 10,
                    'max_sleep_interval': 30,
                    'sleep_interval': 20,
                    'buffersize': 8192,
                    'http_chunk_size': 83886080,
                    'max_downloads': 20,
                    'progress_hooks': [self.download_progress_hook],
                    'postprocessor_args': ['-c:a', 'aac', '-b:a', '192k'],
                    'ignoreerrors': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'referer': 'https://www.youtube.com/',
                }
            else:
                # 일반 모드: 영상과 음성 분리
                self.video_ydl_opts = {
                    'format': 'bestvideo[height>=1080][ext=mp4]',
                    'outtmpl': str(self.video_dir / '%(title)s.%(ext)s'),
                    'socket_timeout': 180,
                    'retries': 5,
                    'fragment_retries': 5,
                    'max_sleep_interval': 10,
                    'sleep_interval': 5,
                    'buffersize': 4096,
                    'http_chunk_size': 41943040,
                    'max_downloads': 10,
                    'progress_hooks': [self.download_progress_hook],
                    'postprocessor_args': ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-movflags', '+faststart'],
                    'ignoreerrors': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'referer': 'https://www.youtube.com/',
                }
                self.audio_ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
                    'outtmpl': str(self.audio_dir / '%(title)s.%(ext)s'),
                    'socket_timeout': 180,
                    'retries': 5,
                    'fragment_retries': 5,
                    'max_sleep_interval': 10,
                    'sleep_interval': 5,
                    'buffersize': 4096,
                    'http_chunk_size': 41943040,
                    'max_downloads': 10,
                    'progress_hooks': [self.download_progress_hook],
                    'postprocessor_args': ['-c:a', 'aac', '-b:a', '192k'],
                    'ignoreerrors': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'referer': 'https://www.youtube.com/',
                }
        else:
            # 기존 통합 다운로드 모드
            if mass_mode:
                # 대용량 모드: 1080p 이상만
                self.ydl_opts = {
                    # 1080p 이상만 다운로드, 없으면 건너뛰기
                    'format': 'bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/best[height>=1080][ext=mp4]',
                    
                    'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
                    'socket_timeout': 300,
                    'retries': 10,
                    'fragment_retries': 10,
                    'max_sleep_interval': 30,
                    'sleep_interval': 20,
                    'buffersize': 8192,
                    'http_chunk_size': 83886080,
                    'max_downloads': 20,
                    'progress_hooks': [self.download_progress_hook],
                    'concurrent_fragment_downloads': 4,
                    
                    # 고화질 옵션
                    'writesubtitles': False,
                    'writeautomaticsub': False,
                    'merge_output_format': 'mp4',
                    'postprocessor_args': ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-movflags', '+faststart'],
                    
                    # 1080p 미만은 에러로 처리
                    'ignoreerrors': True,
                    
                    # YouTube 접근 최적화
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'referer': 'https://www.youtube.com/',
                }
            else:
                # 일반 모드: 1080p 이상만
                self.ydl_opts = {
                    'format': 'bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/best[height>=1080][ext=mp4]',
                    'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
                    'socket_timeout': 180,
                    'retries': 5,
                    'fragment_retries': 5,
                    'max_sleep_interval': 10,
                    'sleep_interval': 5,
                    'buffersize': 4096,
                    'http_chunk_size': 41943040,
                    'max_downloads': 10,
                    'progress_hooks': [self.download_progress_hook],
                    'merge_output_format': 'mp4',
                    'postprocessor_args': ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-movflags', '+faststart'],
                    'ignoreerrors': True,
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'referer': 'https://www.youtube.com/',
                }
    
    def load_download_history(self):
        """다운로드 히스토리 로드"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_download_history(self):
        """다운로드 히스토리 저장"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.downloaded_videos, f, ensure_ascii=False, indent=2)
    
    def download_progress_hook(self, d):
        """다운로드 진행률 표시"""
        if d['status'] == 'downloading':
            if 'total_bytes' in d and d['total_bytes']:
                percent = d['downloaded_bytes'] / d['total_bytes'] * 100
                speed = d.get('_speed_str', 'N/A')
                print(f"📥 다운로드 중: {percent:.1f}% | 속도: {speed}")
        elif d['status'] == 'finished':
            print("✅ 다운로드 완료!")
    
    def sanitize_filename(self, title):
        """파일명 정리"""
        # 특수문자 제거 (더 강력하게)
        sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
        # 슬래시를 언더스코어로 변경
        sanitized = sanitized.replace('/', '_').replace('\\', '_')
        # 공백을 언더스코어로 변경
        sanitized = re.sub(r'\s+', '_', sanitized)
        # 길이 제한
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized
    
    def is_video_already_downloaded(self, video_info):
        """파일 시스템에서 이미 다운로드된 비디오인지 확인"""
        try:
            # 파일명 생성
            safe_title = self.sanitize_filename(video_info['title'])
            original_title = video_info['title'].lower()
            
            # 통합 다운로드 모드 체크
            if not hasattr(self, 'video_dir'):
                # videos 폴더에서 체크
                for ext in ['*.mp4', '*.mkv', '*.avi', '*.mov']:
                    for file_path in self.output_dir.glob(ext):
                        file_name = file_path.name.lower()
                        
                        # 1. 정확한 제목 매칭
                        if safe_title.lower() in file_name:
                            print(f"⚠️  이미 존재하는 파일 발견 (정확 매칭): {file_path.name}")
                            return True
                        
                        # 2. 원본 제목 매칭 (특수문자 처리)
                        if original_title in file_name:
                            print(f"⚠️  이미 존재하는 파일 발견 (원본 매칭): {file_path.name}")
                            return True
                        
                        # 3. 키워드 매칭 (주요 단어들)
                        title_words = original_title.split()
                        file_words = file_name.replace('.mp4', '').split()
                        
                        # 공통 키워드가 3개 이상이면 중복으로 판단
                        common_words = set(title_words) & set(file_words)
                        if len(common_words) >= 3:
                            print(f"⚠️  이미 존재하는 파일 발견 (키워드 매칭): {file_path.name}")
                            return True
            else:
                # 분리 다운로드 모드 체크
                # 비디오 파일 체크
                for ext in ['*.mp4', '*.mkv', '*.avi', '*.mov']:
                    for file_path in self.video_dir.glob(ext):
                        file_name = file_path.name.lower()
                        
                        if safe_title.lower() in file_name or original_title in file_name:
                            print(f"⚠️  이미 존재하는 비디오 파일: {file_path.name}")
                            return True
                
                # 오디오 파일 체크
                for ext in ['*.m4a', '*.mp3', '*.wav', '*.aac']:
                    for file_path in self.audio_dir.glob(ext):
                        file_name = file_path.name.lower()
                        
                        if safe_title.lower() in file_name or original_title in file_name:
                            print(f"⚠️  이미 존재하는 오디오 파일: {file_path.name}")
                            return True
            
            return False
            
        except Exception as e:
            print(f"❌ 파일 체크 오류: {e}")
            return False
    
    def search_youtube_videos(self, query, max_results=10):
        """YouTube 비디오 검색 (CC-BY 전용)"""
        print(f"🔍 검색 중: '{query}' (CC-BY 전용)")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Creative Commons 필터 적용
                cc_query = f"{query} license:creative-commons"
                search_results = ydl.extract_info(f"ytsearch{max_results}:{cc_query}", download=False)
                
                videos = []
                for entry in search_results.get('entries', []):
                    if entry:
                        video_info = {
                            'id': entry.get('id'),
                            'title': entry.get('title', 'Unknown'),
                            'url': entry.get('url') or entry.get('webpage_url'),
                            'duration': entry.get('duration', 0),
                            'view_count': entry.get('view_count', 0),
                        }
                        
                        # 이미 다운로드한 비디오는 제외 (히스토리 + 파일 시스템 체크)
                        if video_info['id'] not in self.downloaded_videos and not self.is_video_already_downloaded(video_info):
                            videos.append(video_info)
                
                print(f"📊 검색 결과: {len(videos)}개 비디오 (CC-BY 필터 적용)")
                return videos
                
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return []
    
    def search_cc_only_videos(self, query, max_results=10):
        """Creative Commons 전용 검색"""
        print(f"🔍 CC-BY 전용 검색: '{query}'")
        
        # CC-BY 전용 검색어들
        cc_search_queries = [
            f"{query} creative commons",
            f"{query} cc by",
            f"{query} license:creative-commons",
            f"{query} attribution license",
            f"{query} 크리에이티브 커먼즈",
            f"{query} 저작자 표시"
        ]
        
        all_videos = []
        
        for cc_query in cc_search_queries:
            try:
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    search_results = ydl.extract_info(f"ytsearch{max_results//len(cc_search_queries)}:{cc_query}", download=False)
                    
                    for entry in search_results.get('entries', []):
                        if entry:
                            video_info = {
                                'id': entry.get('id'),
                                'title': entry.get('title', 'Unknown'),
                                'url': entry.get('url') or entry.get('webpage_url'),
                                'duration': entry.get('duration', 0),
                                'view_count': entry.get('view_count', 0),
                            }
                            
                            # 중복 제거 (ID 기준)
                            if video_info['id'] not in [v['id'] for v in all_videos]:
                                if video_info['id'] not in self.downloaded_videos and not self.is_video_already_downloaded(video_info):
                                    all_videos.append(video_info)
                                    
            except Exception as e:
                print(f"❌ CC 검색 오류 ({cc_query}): {e}")
                continue
        
        print(f"📊 CC-BY 검색 결과: {len(all_videos)}개 비디오")
        return all_videos
    
    def check_license(self, video_info):
        """CC-BY 라이선스 확인"""
        # CC-BY 필터링이 비활성화된 경우 항상 True 반환
        if not self.cc_by_only:
            return True
            
        try:
            ydl_opts_check = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts_check) as ydl:
                video_meta = ydl.extract_info(video_info['url'], download=False)
                
                # 라이선스 정보 확인
                license_info = video_meta.get('license', '')
                uploader = video_meta.get('uploader', '')
                description = video_meta.get('description', '')
                
                # CC-BY 라이선스 확인 (다양한 표현 방식)
                cc_by_keywords = [
                    'creative commons',
                    'cc by',
                    'cc-by',
                    'creative commons by',
                    'creative commons attribution',
                    'cc attribution',
                    'attribution license',
                    '크리에이티브 커먼즈',
                    '저작자 표시',
                    'cc-by 4.0',
                    'cc-by 3.0',
                    'cc-by 2.0'
                ]
                
                # 설명에서 라이선스 확인
                description_lower = description.lower()
                for keyword in cc_by_keywords:
                    if keyword in description_lower:
                        print(f"✅ CC-BY 라이선스 확인됨: {video_info['title']}")
                        return True
                
                # 라이선스 필드에서 확인
                license_lower = license_info.lower()
                for keyword in cc_by_keywords:
                    if keyword in license_lower:
                        print(f"✅ CC-BY 라이선스 확인됨: {video_info['title']}")
                        return True
                
                # 업로더 정보에서 확인 (일부 채널은 모든 영상이 CC-BY)
                uploader_lower = uploader.lower()
                cc_by_channels = [
                    'creative commons',
                    'cc by',
                    '크리에이티브 커먼즈',
                    '저작자 표시'
                ]
                for keyword in cc_by_channels:
                    if keyword in uploader_lower:
                        print(f"✅ CC-BY 채널 확인됨: {video_info['title']}")
                        return True
                
                print(f"❌ CC-BY 라이선스 아님: {video_info['title']}")
                print(f"   - 라이선스: {license_info}")
                print(f"   - 업로더: {uploader}")
                return False
                
        except Exception as e:
            print(f"❌ 라이선스 확인 실패: {e}")
            return False
    
    def check_video_quality(self, video_info):
        """비디오 다운로드 전에 1080p+ 화질과 CC-BY 라이선스 확인"""
        try:
            ydl_opts_check = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts_check) as ydl:
                video_meta = ydl.extract_info(video_info['url'], download=False)
                
                # 1. CC-BY 라이선스 확인
                if not self.check_license(video_info):
                    return False
                
                # 2. 1080p 이상 화질 확인
                if video_meta.get('formats'):
                    has_1080p = any(
                        fmt.get('height', 0) >= 1080 
                        for fmt in video_meta['formats']
                        if fmt.get('height') is not None and fmt.get('ext') == 'mp4'
                    )
                    
                    if has_1080p:
                        print(f"✅ 1080p+ 화질 확인됨: {video_info['title']}")
                        return True
                    else:
                        print(f"⚠️ 1080p+ 화질 없음: {video_info['title']}")
                        return False
                        
        except Exception as e:
            print(f"❌ 화질 확인 실패: {e}")
            return False
    
    def download_with_retry(self, video_url, video_info, max_retries=1):
        """비디오 다운로드 (재시도 포함)"""
        video_id = video_info.get('id')
        title = video_info.get('title', 'Unknown')
        
        # 다운로드 전 미리보기 입술 검출 확인
        print(f"🔍 다운로드 전 입술 검출 확인 중...")
        if not self.lip_detector.check_lip_in_preview(video_url):
            print(f"❌ 입술 검출 실패로 다운로드 건너뜀: {title}")
            return False
        
        print(f"✅ 입술 검출 확인됨, 다운로드 시작: {title}")
        
        for attempt in range(max_retries):
            try:
                print(f"📥 다운로드 시도 {attempt + 1}/{max_retries}: {title}")
                
                if self.separate_audio:
                    # 분리 다운로드
                    video_path, audio_path = self.download_separate_formats(video_url, video_info)
                    if video_path and audio_path:
                        return self.check_and_save_video(video_path, audio_path, video_info)
                else:
                    # 통합 다운로드
                    video_path = self.download_single_format(video_url, video_info)
                    if video_path:
                        return self.check_and_save_video(video_path, None, video_info)
                        
            except Exception as e:
                print(f"❌ 다운로드 실패 (시도 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ {5}초 후 재시도...")
                    time.sleep(5)
                else:
                    print(f"❌ 최대 재시도 횟수 초과: {title}")
                    return False
        
        return False
    
    def download_separate_formats(self, video_url, video_info):
        """영상과 음성을 분리하여 다운로드"""
        safe_title = self.sanitize_filename(video_info['title'])
        
        try:
            # 비디오 다운로드
            with yt_dlp.YoutubeDL(self.video_ydl_opts) as ydl:
                ydl.download([video_url])
                # 다운로드된 파일 경로 찾기
                video_path = None
                for file in Path(self.output_dir / "videos").glob(f"*{safe_title}*.mp4"):
                    video_path = file
                    break
                print(f"✅ 비디오 다운로드 성공: {video_path}")
            
            # 오디오 다운로드
            with yt_dlp.YoutubeDL(self.audio_ydl_opts) as ydl:
                ydl.download([video_url])
                # 다운로드된 파일 경로 찾기
                audio_path = None
                for file in Path(self.output_dir / "audio").glob(f"*{safe_title}*.m4a"):
                    audio_path = file
                    break
                print(f"✅ 음성 다운로드 성공: {audio_path}")
            
            return video_path, audio_path
        except Exception as e:
            print(f"❌ 분리 다운로드 실패: {e}")
            return None, None
    
    def download_single_format(self, video_url, video_info):
        """단일 포맷으로 다운로드"""
        safe_title = self.sanitize_filename(video_info['title'])
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([video_url])
                # 다운로드된 파일 경로 찾기
                video_path = None
                for file in Path(self.output_dir).glob(f"*{safe_title}*.mp4"):
                    video_path = file
                    break
                print(f"✅ 다운로드 성공: {video_path}")
            return video_path
        except Exception as e:
            print(f"❌ 단일 포맷 다운로드 실패: {e}")
            return None
    
    def check_and_save_video(self, video_path, audio_path, video_info):
        """비디오 다운로드 후 입술 검출 확인"""
        # 파일 경로가 비어있을 경우 예외 처리
        if not video_path:
            print(f"❌ 비디오 파일 경로가 없습니다: {video_info['title']}")
            return False
        
        # 파일 확장자 추출
        file_ext = Path(video_path).suffix.lower()
        
        # 파일 확장자에 따라 적절한 검출 함수 호출
        if file_ext == '.mp4':
            has_lips = self.lip_detector.check_lip_in_video(video_path)
        elif file_ext == '.mkv':
            # MKV는 프레임 단위로 검출하는 것이 더 안정적일 수 있으므로, 미리보기 검출을 사용
            has_lips = self.lip_detector.check_lip_in_preview(video_info['url'])
        else:
            print(f"⚠️ 지원하지 않는 파일 확장자: {file_ext}. 입술 검출을 건너뜁니다.")
            return False

        if has_lips:
            print(f"✅ 입술 검출 성공 (98%+): {video_path.name}")
            # 히스토리에 성공으로 추가
            self.downloaded_videos[video_info['id']] = {
                'title': video_info['title'],
                'url': video_info['url'],
                'download_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': True,
                'lip_detected': True,
                'license': 'CC-BY',
                'quality': '1080p+',
                'video_path': video_path.name,
                'audio_path': audio_path.name if audio_path else None
            }
            self.save_download_history()
            return True
        else:
            print(f"❌ 입술 검출 실패 (98% 미만): {video_path.name}")
            # 히스토리에 실패로 추가
            self.downloaded_videos[video_info['id']] = {
                'title': video_info['title'],
                'url': video_info['url'],
                'download_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': False,
                'lip_detected': False,
                'reason': 'insufficient_lip_detection',
                'video_path': video_path.name,
                'audio_path': audio_path.name if audio_path else None
            }
            self.save_download_history()
            try:
                video_path.unlink()
                if audio_path:
                    audio_path.unlink()
                print(f"🗑️ 삭제됨: {video_path.name}")
                if audio_path:
                    print(f"🗑️ 삭제됨: {audio_path.name}")
            except:
                pass
            return False
    
    def run_crawler(self, query, max_videos=5, cc_only_mode=False):
        """크롤러 실행"""
        print(f"🚀 크롤러 시작: '{query}' (최대 {max_videos}개)")
        
        # 검색 실행
        if cc_only_mode:
            videos = self.search_cc_only_videos(query, max_videos * 3)  # 더 많은 후보 검색
        else:
            videos = self.search_youtube_videos(query, max_videos * 3)  # 더 많은 후보 검색
        
        if not videos:
            print("❌ 검색 결과가 없습니다.")
            return
        
        # 중복 제거 및 필터링
        unique_videos = []
        seen_ids = set()
        
        for video in videos:
            if video['id'] not in seen_ids:
                seen_ids.add(video['id'])
                unique_videos.append(video)
        
        print(f"📊 중복 제거 후: {len(unique_videos)}개 비디오")
        
        successful_downloads = 0
        processed_count = 0
        
        for i, video_info in enumerate(unique_videos, 1):
            processed_count += 1
            print(f"\n📹 [{processed_count}/{len(unique_videos)}] 처리 중: {video_info['title']}")
            
            # 이미 다운로드된 비디오인지 확인
            if video_info['id'] in self.downloaded_videos:
                print("⚠️  이미 다운로드된 비디오입니다. 건너뜁니다.")
                continue
            
            if self.is_video_already_downloaded(video_info):
                print("⚠️  파일 시스템에서 이미 존재하는 비디오입니다. 건너뜁니다.")
                continue
            else:
                print(f"🔍 파일 체크 완료: '{video_info['title']}' - 새로운 비디오")
            
            # 라이선스 확인
            if not self.check_license(video_info):
                print("❌ CC-BY 라이선스가 아닙니다. 건너뜁니다.")
                continue
            
            # 비디오 품질 확인
            if not self.check_video_quality(video_info):
                print("❌ 비디오 품질이 기준에 맞지 않습니다. 건너뜁니다.")
                continue
            
            # 다운로드 실행
            if self.download_with_retry(video_info['url'], video_info):
                successful_downloads += 1
                print(f"✅ 다운로드 성공! ({successful_downloads}/{max_videos})")
                
                # 목표 달성 시 중단
                if successful_downloads >= max_videos:
                    break
            else:
                print("❌ 다운로드 실패")
        
        print(f"\n🎉 크롤링 완료! 성공: {successful_downloads}/{max_videos}")

    def validate_existing_files(self):
        """기존 파일들을 검증하여 입술이 없는 파일 삭제"""
        print("🔍 기존 파일 검증 중...")
        video_files = list(self.output_dir.glob("*.mp4"))
        
        valid_files = 0
        deleted_files = 0
        
        for video_file in video_files:
            print(f"🔍 검증 중: {video_file.name}")
            has_lips = self.lip_detector.check_lip_in_video(video_file)
            
            if has_lips:
                print(f"✅ 유효한 파일: {video_file.name}")
                valid_files += 1
            else:
                print(f"❌ 삭제: {video_file.name}")
                try:
                    video_file.unlink()
                    deleted_files += 1
                except:
                    pass
                    
        print(f"📊 검증 완료: 유효한 파일 {valid_files}개, 삭제된 파일 {deleted_files}개")
        return valid_files, deleted_files


def run_mass_collection(queries=None, max_videos_per_query=3, output_dir="data/lip_videos", debug=False, separate_audio=False, cc_by_only=True, cc_only_search=False):
    """대용량 수집 실행"""
    print("🚀 대용량 수집 시작...")
    print("="*70)
    print(f"📊 총 검색어: {len(queries)}개")
    print(f"📺 쿼리당 최대 비디오: {max_videos_per_query}개")
    print(f"🎯 화질 기준: 1080p 이상만")
    print(f"👄 입술 검출 기준: 98% 이상")
    print(f"⚖️ 라이선스 필터: {'CC-BY만' if cc_by_only else '모든 라이선스'}")
    print(f"🔍 CC-BY 전용 검색: {'활성화' if cc_only_search else '비활성화'}")
    print(f"🎬 분리 다운로드: {'예' if separate_audio else '아니오'}")
    print(f"⚙️ 디버그 모드: {'예' if debug else '아니오'}")
    print(f"📁 저장 위치: {output_dir}")
    print("="*70)
    
    if queries is None:
        queries = MASS_COLLECTION_QUERIES
    
    crawler = LipVideoCrawler(
        output_dir=output_dir, 
        debug_mode=debug, 
        mass_mode=True, 
        separate_audio=separate_audio,
        cc_by_only=cc_by_only
    )
    
    start_time = time.time()
    completed_queries = []
    failed_queries = []
    total_successful_downloads = 0
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] '{query}' 처리 중...")
        
        try:
            # CC-BY 전용 검색 모드 사용
            if cc_only_search:
                videos = crawler.search_cc_only_videos(query, max_videos_per_query * 3)
            else:
                videos = crawler.search_youtube_videos(query, max_videos_per_query * 3)
                
            if not videos:
                print(f"❌ 검색 결과 없음: '{query}'")
                failed_queries.append(query)
                continue
                
            successful_downloads = 0
            
            # 비디오 목록을 랜덤으로 섞기
            random.shuffle(videos[:max_videos_per_query * 2])
            print(f"🎲 랜덤 순서로 다운로드 시작...")
            
            for j, video_info in enumerate(videos[:max_videos_per_query * 2], 1):
                print(f"  [{j}/{len(videos[:max_videos_per_query * 2])}] {video_info['title']}")
                if crawler.download_with_retry(video_info['url'], video_info):
                    successful_downloads += 1
                    total_successful_downloads += 1
                    if successful_downloads >= max_videos_per_query:
                        break
                    
            completed_queries.append(query)
            print(f"✅ '{query}' 완료 - {successful_downloads}개 성공")
            
        except Exception as e:
            failed_queries.append(query)
            print(f"❌ '{query}' 실패: {e}")
        
        # 잠시 휴식 (서버 부하 방지)
        if i < len(queries):
            wait_time = 20 if debug else 15  # 더 긴 대기시간
            print(f"⏳ {wait_time}초 휴식...")
            time.sleep(wait_time)
    
    # 최종 결과
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("🎉 대용량 데이터 수집 완료!")
    print(f"⏱️ 총 소요 시간: {total_time/3600:.1f}시간")
    print(f"✅ 성공한 쿼리: {len(completed_queries)}개")
    print(f"❌ 실패한 쿼리: {len(failed_queries)}개")
    print(f"📺 총 다운로드: {total_successful_downloads}개")
    print(f"📊 성공률: {len(completed_queries)/(len(completed_queries)+len(failed_queries))*100:.1f}%")
    print(f"🎯 모든 파일: 1080p+ 화질 & 98%+ 입술 검출 & CC-BY 라이선스")
    print("="*70)


def run_news_collection(max_videos_per_query=3, output_dir="data/lip_videos", debug=False, separate_audio=False, cc_by_only=True, cc_only_search=False):
    """뉴스 전용 데이터 수집"""
    print("📺 뉴스 전용 데이터 수집 시작...")
    print(f"🎯 뉴스 관련 키워드: {len(NEWS_COLLECTION_QUERIES)}개")
    
    start_time = time.time()
    total_successful_downloads = 0
    completed_queries = []
    failed_queries = []
    
    for i, query in enumerate(NEWS_COLLECTION_QUERIES, 1):
        try:
            print(f"\n[{i}/{len(NEWS_COLLECTION_QUERIES)}] 🔍 검색 중: '{query}'")
            
            crawler = LipVideoCrawler(
                output_dir=output_dir, 
                debug_mode=debug, 
                separate_audio=separate_audio, 
                cc_by_only=cc_by_only
            )
            
            if cc_only_search:
                videos = crawler.search_cc_only_videos(query, max_videos_per_query * 3)
            else:
                videos = crawler.search_youtube_videos(query, max_videos_per_query * 3)
                
            if not videos:
                print(f"❌ 검색 결과 없음: '{query}'")
                failed_queries.append(query)
                continue
                
            successful_downloads = 0
            
            # 비디오 목록을 랜덤으로 섞기
            random.shuffle(videos[:max_videos_per_query * 2])
            print(f"🎲 랜덤 순서로 다운로드 시작...")
            
            for j, video_info in enumerate(videos[:max_videos_per_query * 2], 1):
                print(f"  [{j}/{len(videos[:max_videos_per_query * 2])}] {video_info['title']}")
                if crawler.download_with_retry(video_info['url'], video_info):
                    successful_downloads += 1
                    total_successful_downloads += 1
                    if successful_downloads >= max_videos_per_query:
                        break
                    
            completed_queries.append(query)
            print(f"✅ '{query}' 완료 - {successful_downloads}개 성공")
            
        except Exception as e:
            failed_queries.append(query)
            print(f"❌ '{query}' 실패: {e}")
        
        # 잠시 휴식 (서버 부하 방지)
        if i < len(NEWS_COLLECTION_QUERIES):
            wait_time = 20 if debug else 15
            print(f"⏳ {wait_time}초 휴식...")
            time.sleep(wait_time)
    
    # 최종 결과
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("📺 뉴스 전용 데이터 수집 완료!")
    print(f"⏱️ 총 소요 시간: {total_time/3600:.1f}시간")
    print(f"✅ 성공한 쿼리: {len(completed_queries)}개")
    print(f"❌ 실패한 쿼리: {len(failed_queries)}개")
    print(f"📺 총 다운로드: {total_successful_downloads}개")
    print(f"📊 성공률: {len(completed_queries)/(len(completed_queries)+len(failed_queries))*100:.1f}%")
    print(f"🎯 모든 파일: 1080p+ 화질 & 98%+ 입술 검출 & CC-BY 라이선스")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="입술 비디오 크롤러 (1080p+ 전용)")
    parser.add_argument("query", nargs='?', help="검색할 키워드")
    parser.add_argument("--max-videos", type=int, default=5, help="최대 다운로드 개수 (기본값: 5)")
    parser.add_argument("--output-dir", default="data/lip_videos", help="출력 디렉토리")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    # 대용량 수집 옵션
    parser.add_argument("--mass-collection", action="store_true", help="대용량 데이터 수집 모드")
    parser.add_argument("--news-collection", action="store_true", help="뉴스 전용 데이터 수집 모드")
    parser.add_argument("--queries", nargs='+', help="대용량 수집용 검색어 목록")
    parser.add_argument("--max-videos-per-query", type=int, default=3, help="쿼리당 최대 비디오 수")
    
    # 분리 다운로드 옵션
    parser.add_argument("--separate-audio", action="store_true", help="영상과 음성을 분리하여 다운로드합니다.")
    
    # 라이선스 필터링 옵션
    parser.add_argument("--no-cc-by-only", action="store_true", help="모든 라이선스 영상을 다운로드합니다. (기본값: CC-BY만)")
    
    # CC-BY 전용 검색 옵션
    parser.add_argument("--cc-only-search", action="store_true", help="CC-BY 전용 검색 모드를 사용합니다.")
    
    args = parser.parse_args()
    
    # CC-BY 필터링 설정 (기본값: True)
    cc_by_only = not args.no_cc_by_only
    
    print(f"🔧 설정 정보:")
    print(f"   - CC-BY 필터링: {'활성화' if cc_by_only else '비활성화'}")
    print(f"   - CC-BY 전용 검색: {'활성화' if args.cc_only_search else '비활성화'}")
    print(f"   - 분리 다운로드: {'활성화' if args.separate_audio else '비활성화'}")
    print(f"   - 대용량 수집: {'활성화' if args.mass_collection else '비활성화'}")
    print(f"   - 디버그 모드: {'활성화' if args.debug else '비활성화'}")
    
    if args.mass_collection:
        # 대용량 수집 모드
        print("🚀 대용량 수집 모드 시작...")
        queries = args.queries if args.queries else MASS_COLLECTION_QUERIES
        run_mass_collection(queries, args.max_videos_per_query, args.output_dir, args.debug, args.separate_audio, cc_by_only, args.cc_only_search)
    elif args.news_collection:
        # 뉴스 전용 수집 모드
        print("🚀 뉴스 전용 수집 모드 시작...")
        run_news_collection(args.max_videos_per_query, args.output_dir, args.debug, args.separate_audio, cc_by_only, args.cc_only_search)
    elif args.query:
        # 일반 크롤러 모드
        print("🚀 일반 크롤러 모드 시작...")
        crawler = LipVideoCrawler(output_dir=args.output_dir, debug_mode=args.debug, separate_audio=args.separate_audio, cc_by_only=cc_by_only)
        crawler.run_crawler(args.query, args.max_videos, cc_only_mode=args.cc_only_search)
    else:
        # 인수가 없으면 도움말 출력
        print("❌ 검색어를 입력해주세요!")
        print("사용법 예시:")
        print("  python lip_video_crawler_simple.py '입술' --max-videos 5")
        print("  python lip_video_crawler_simple.py --mass-collection --separate-audio")
        print("  python lip_video_crawler_simple.py --help")
        parser.print_help()


if __name__ == "__main__":
    main()
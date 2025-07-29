#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자동 재시작 크롤러 래퍼
크롤러가 중간에 멈춰도 자동으로 재시작합니다.
"""

import subprocess
import sys
import time
import signal
import os
from datetime import datetime
import argparse

class AutoRestartCrawler:
    def __init__(self, max_restarts=10, restart_delay=30):
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.restart_count = 0
        self.start_time = datetime.now()
        
    def signal_handler(self, signum, frame):
        """시그널 핸들러 (Ctrl+C 등)"""
        print(f"\n🛑 사용자에 의해 중단됨 (시그널: {signum})")
        print(f"⏱️ 총 실행 시간: {datetime.now() - self.start_time}")
        print(f"🔄 총 재시작 횟수: {self.restart_count}")
        sys.exit(0)
        
    def run_with_restart(self, script_args):
        """크롤러를 재시작 가능하게 실행"""
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 자동 재시작 크롤러 시작...")
        print(f"🔄 최대 재시작 횟수: {self.max_restarts}")
        print(f"⏳ 재시작 대기 시간: {self.restart_delay}초")
        print(f"📅 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        while self.restart_count < self.max_restarts:
            try:
                print(f"\n🔄 실행 시도 {self.restart_count + 1}/{self.max_restarts}")
                print(f"⏰ 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"📋 실행 명령: python lip_video_crawler_simple.py {' '.join(script_args)}")
                print("-" * 50)
                
                # 크롤러 실행
                result = subprocess.run(
                    [sys.executable, "lip_video_crawler_simple.py"] + script_args,
                    capture_output=False,  # 실시간 출력 표시
                    text=True
                )
                
                # 정상 종료인 경우
                if result.returncode == 0:
                    print(f"\n✅ 크롤러가 정상적으로 완료되었습니다!")
                    print(f"⏱️ 총 실행 시간: {datetime.now() - self.start_time}")
                    print(f"🔄 총 재시작 횟수: {self.restart_count}")
                    return True
                    
                # 비정상 종료인 경우
                else:
                    print(f"\n❌ 크롤러가 비정상 종료됨 (종료 코드: {result.returncode})")
                    
            except KeyboardInterrupt:
                print(f"\n🛑 사용자에 의해 중단됨")
                break
                
            except Exception as e:
                print(f"\n💥 예외 발생: {e}")
                
            # 재시작 준비
            self.restart_count += 1
            
            if self.restart_count < self.max_restarts:
                print(f"\n⏳ {self.restart_delay}초 후 재시작...")
                print(f"🔄 남은 재시작 횟수: {self.max_restarts - self.restart_count}")
                
                # 카운트다운
                for i in range(self.restart_delay, 0, -1):
                    print(f"⏰ {i}초 후 재시작...", end='\r')
                    time.sleep(1)
                print()
                
            else:
                print(f"\n❌ 최대 재시작 횟수({self.max_restarts})에 도달했습니다.")
                print(f"⏱️ 총 실행 시간: {datetime.now() - self.start_time}")
                print(f"🔄 총 재시작 횟수: {self.restart_count}")
                return False
        
        return False

def main():
    parser = argparse.ArgumentParser(description="자동 재시작 크롤러")
    parser.add_argument("--max-restarts", type=int, default=10, 
                       help="최대 재시작 횟수 (기본값: 10)")
    parser.add_argument("--restart-delay", type=int, default=30, 
                       help="재시작 대기 시간(초) (기본값: 30)")
    parser.add_argument("--mass-collection", action="store_true", 
                       help="대용량 수집 모드")
    parser.add_argument("--separate-audio", action="store_true", 
                       help="분리 다운로드 (비디오/오디오)")
    parser.add_argument("--cc-only-search", action="store_true", 
                       help="CC-BY 전용 검색")
    parser.add_argument("--debug", action="store_true", 
                       help="디버그 모드")
    parser.add_argument("--max-videos-per-query", type=int, default=3, 
                       help="쿼리당 최대 비디오 수 (기본값: 3)")
    
    args = parser.parse_args()
    
    # 크롤러 인자 구성
    script_args = []
    
    if args.mass_collection:
        script_args.append("--mass-collection")
    if args.separate_audio:
        script_args.append("--separate-audio")
    if args.cc_only_search:
        script_args.append("--cc-only-search")
    if args.debug:
        script_args.append("--debug")
    if args.max_videos_per_query:
        script_args.extend(["--max-videos-per-query", str(args.max_videos_per_query)])
    
    # 자동 재시작 크롤러 실행
    crawler = AutoRestartCrawler(
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay
    )
    
    success = crawler.run_with_restart(script_args)
    
    if success:
        print("🎉 모든 작업이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("❌ 최대 재시작 횟수에 도달하여 중단되었습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
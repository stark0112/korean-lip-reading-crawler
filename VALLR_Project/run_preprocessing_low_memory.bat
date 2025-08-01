@echo off
chcp 65001 > nul
echo ========================================
echo    VALLR 데이터 전처리 (저메모리 모드)
echo ========================================
echo.

set MAX_RETRIES=100
set RETRY_COUNT=0

:restart_loop
set /a RETRY_COUNT+=1
echo.
echo 🔄 전처리 시작 (시도 %RETRY_COUNT%/%MAX_RETRIES%)
echo 📅 시작 시간: %date% %time%
echo 💾 저메모리 모드: 한 번에 10개씩 처리
echo.

python date_Preprocessing.py --batch_size 10

set EXIT_CODE=%errorlevel%

if %EXIT_CODE% equ 0 (
    echo.
    echo ✅ 전처리가 성공적으로 완료되었습니다!
    goto :end
) else (
    echo.
    echo ⚠️ 전처리가 중단되었습니다 (종료 코드: %EXIT_CODE%)
    
    if %EXIT_CODE% equ -1073741819 (
        echo 💾 메모리 부족 오류 감지됨
        echo 🔄 메모리 정리 후 재시작... (60초 대기)
        timeout /t 60 /nobreak > nul
    ) else (
        echo 🔄 15초 후 자동 재시작...
        timeout /t 15 /nobreak > nul
    )
    
    if %RETRY_COUNT% lss %MAX_RETRIES% (
        echo 🔄 재시작 중... (시도 %RETRY_COUNT%/%MAX_RETRIES%)
        goto :restart_loop
    ) else (
        echo ❌ 최대 재시작 횟수(%MAX_RETRIES%)에 도달했습니다.
        echo 💡 수동으로 다시 시작하려면: python date_Preprocessing.py --batch_size 10
    )
)

:end
echo.
echo 🎉 작업 완료!
pause 
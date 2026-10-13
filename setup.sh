#!/bin/bash
echo "=============================="
echo "  코인 선물 신호 분석기 설치"
echo "=============================="

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3이 설치되어 있지 않습니다."
    echo "   https://www.python.org/downloads/ 에서 설치 후 다시 실행하세요."
    exit 1
fi

echo "✅ Python3 확인: $(python3 --version)"

# pip 업그레이드
python3 -m pip install --upgrade pip -q

# 패키지 설치
echo ""
echo "📦 패키지 설치 중... (1~3분 소요)"
python3 -m pip install -r requirements.txt -q

echo ""
echo "✅ 설치 완료!"
echo ""
echo "▶ 실행하려면: run.sh 파일을 더블클릭하거나 터미널에서 ./run.sh"

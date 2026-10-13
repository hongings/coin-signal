"""
Notifier를 별도 모듈로 분리 — 모듈은 sys.modules에 캐시되므로
Streamlit 재실행 시에도 한 번만 임포트됨 → 스레드 중복 방지
"""
import threading

_started = False
_lock = threading.Lock()

def _run():
    try:
        from notifier import main as notifier_main
        notifier_main()
    except Exception as e:
        print(f"NOTIFIER ERROR: {e}", flush=True)

def start():
    global _started
    with _lock:
        if not _started:
            _started = True
            t = threading.Thread(target=_run, daemon=True)
            t.start()

start()  # 최초 임포트 시 1회만 실행

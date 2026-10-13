"""
Telegram signal notifier — runs alongside Streamlit app.
Checks BTC/ETH scalp signals every 60s and sends Telegram alert on LONG/SHORT.
"""
import os, time, logging
import requests, ccxt
import pandas as pd
import numpy as np
import ta
from scipy.signal import find_peaks
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
KST = timezone(timedelta(hours=9))

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
INTERVAL = 60  # seconds between checks

# ── 텔레그램 전송
def send(msg):
    if not TOKEN or not CHAT_ID:
        logging.warning("Telegram 환경변수 미설정 (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        logging.info("Telegram 전송 완료")
    except Exception as e:
        logging.error(f"Telegram 전송 실패: {e}")

# ── OKX 데이터 (캐시 없음, 직접 호출)
def fetch_ohlcv(symbol, tf, limit=300):
    try:
        ex = ccxt.okx({"options": {"defaultType": "swap"}})
        raw = ex.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"OHLCV 오류 {symbol} {tf}: {e}")
        return None

def fetch_funding(symbol):
    try:
        ex = ccxt.okx({"options": {"defaultType": "swap"}})
        return float(ex.fetch_funding_rate(symbol).get("fundingRate", 0))
    except:
        return 0.0

# ── 지표 계산 (Streamlit 없는 순수 Python)
def calc_ema(df, periods=(9, 21, 50, 200)):
    c = df["close"]
    price = c.iloc[-1]
    series = {p: ta.trend.ema_indicator(c, window=p) for p in periods}
    above = sum(price > s.iloc[-1] for s in series.values() if not np.isnan(s.iloc[-1]))
    n = len(periods)
    return (above - n/2) / (n/2), series

def calc_market_structure(df, lookback=80):
    h = df["high"].values[-lookback:]
    l = df["low"].values[-lookback:]
    peaks,_   = find_peaks(h, distance=5, prominence=h.std()*0.3)
    troughs,_ = find_peaks(-l, distance=5, prominence=l.std()*0.3)
    if len(peaks) < 2 or len(troughs) < 2:
        return 0.0
    hh = h[peaks[-1]] > h[peaks[-2]]
    hl = l[troughs[-1]] > l[troughs[-2]]
    lh = h[peaks[-1]] < h[peaks[-2]]
    ll = l[troughs[-1]] < l[troughs[-2]]
    if hh and hl:   return  0.7
    elif lh and ll: return -0.7
    else:           return  0.0

def calc_ichimoku_score(df):
    try:
        h, l, c = df["high"], df["low"], df["close"]
        ichi = ta.trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
        sa, sb = ichi.ichimoku_a(), ichi.ichimoku_b()
        tenkan = ichi.ichimoku_conversion_line()
        kijun  = ichi.ichimoku_base_line()
        price = c.iloc[-1]
        if len(df) < 36: return 0.0
        cloud_top = max(sa.iloc[-26], sb.iloc[-26])
        cloud_bot = min(sa.iloc[-26], sb.iloc[-26])
        sc = 0.0
        if price > cloud_top: sc += 0.35
        elif price < cloud_bot: sc -= 0.35
        tv, kv = tenkan.iloc[-1], kijun.iloc[-1]
        if not (np.isnan(tv) or np.isnan(kv)):
            sc += 0.30 if tv > kv else -0.30
        sc += 0.20 if sa.iloc[-1] > sb.iloc[-1] else -0.20
        if not np.isnan(kv):
            sc += 0.15 if price > kv else -0.15
        return float(np.clip(sc, -1, 1))
    except:
        return 0.0

def calc_atr(df, period=14):
    return ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=period
    ).average_true_range()

def calc_rsi_score(df):
    rsi = ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1]
    if np.isnan(rsi): return 0.0
    if rsi < 35: return  0.7
    elif rsi > 65: return -0.7
    else: return (50 - rsi) / 50 * 0.5

def calc_volume_ok(df):
    vol = df["volume"]
    avg = vol.rolling(20).mean().iloc[-1]
    return (vol.iloc[-1] / avg) >= 1.3 if (avg and avg > 0) else False

# ── S/R 구간
def find_sr_zones(df, price, lookback=120, tolerance=0.005):
    h = df["high"].values[-lookback:]
    l = df["low"].values[-lookback:]
    peaks,_   = find_peaks(h, distance=5, prominence=h.std()*0.25)
    troughs,_ = find_peaks(-l, distance=5, prominence=l.std()*0.25)
    res_raw = h[peaks].tolist()
    sup_raw = l[troughs].tolist()
    mag = 10 ** max(3, int(np.log10(price)) - 1)
    for m in range(-3, 4):
        lv = round(price / mag) * mag + m * mag
        (res_raw if lv > price else sup_raw).append(float(lv))
    tol2 = tolerance * 2
    def cluster(lvs):
        if not lvs: return []
        lvs = sorted(lvs)
        groups = [[lvs[0]]]
        for lv in lvs[1:]:
            if (lv - groups[-1][-1]) / max(groups[-1][-1], 1) < tol2:
                groups[-1].append(lv)
            else:
                groups.append([lv])
        return [(float(np.mean(g)), len(g)) for g in groups]
    r = [(lv, "저항", cnt) for lv, cnt in cluster(res_raw) if lv > price * 0.997]
    s = [(lv, "지지",  cnt) for lv, cnt in cluster(sup_raw) if lv < price * 1.003]
    all_z = r + s
    all_z.sort(key=lambda x: abs(x[0] - price))
    return all_z[:8]

def find_sr_levels(df, price, lookback=120, tol=0.008):
    h, l = df["high"].values[-lookback:], df["low"].values[-lookback:]
    peaks,_   = find_peaks(h, distance=5, prominence=h.std()*0.25)
    troughs,_ = find_peaks(-l, distance=5, prominence=l.std()*0.25)
    res, sup  = h[peaks].tolist(), l[troughs].tolist()
    mag = 10 ** max(3, int(np.log10(price)) - 1)
    for m in range(-6, 7):
        lv = round(price / mag) * mag + m * mag
        (res if lv > price else sup).append(float(lv))
    def cluster(lvs):
        if not lvs: return []
        lvs = sorted(set(lvs)); groups = [[lvs[0]]]
        for lv in lvs[1:]:
            if (lv - groups[-1][-1]) / max(groups[-1][-1], 1) < tol: groups[-1].append(lv)
            else: groups.append([lv])
        return [float(np.mean(g)) for g in groups]
    r_lvs = [(lv, "저항") for lv in cluster(res) if lv > price * 0.997]
    s_lvs = [(lv, "지지")  for lv in cluster(sup) if lv < price * 1.003]
    all_l = r_lvs + s_lvs
    all_l.sort(key=lambda x: abs(x[0] - price))
    return all_l[:10]

def get_atr_stop(df, entry, is_long, sr_levels, mult=0.8):
    atr_val  = calc_atr(df).iloc[-1]
    atr_stop = entry - atr_val * mult if is_long else entry + atr_val * mult
    sups = sorted([(lv, t) for lv, t in sr_levels if lv < entry], reverse=True)
    ress = sorted([(lv, t) for lv, t in sr_levels if lv > entry])
    if is_long and sups:
        sr_stop = sups[0][0] * 0.9965
        sl = min(atr_stop, sr_stop) if sr_stop > entry * 0.90 else atr_stop
    elif not is_long and ress:
        sr_stop = ress[0][0] * 1.0035
        sl = max(atr_stop, sr_stop) if sr_stop < entry * 1.10 else atr_stop
    else:
        sl = atr_stop
    return sl

# ── 5m 반응 확인
def check_reaction(df_5m, zone_lv, is_support, tolerance=0.004):
    last8 = df_5m.tail(8)
    z_lo, z_hi = zone_lv * (1 - tolerance), zone_lv * (1 + tolerance)
    if is_support:
        touched = last8[last8["low"] <= z_hi]
    else:
        touched = last8[last8["high"] >= z_lo]
    if touched.empty: return False
    cur = df_5m["close"].iloc[-1]
    if is_support and cur < z_lo: return False
    if not is_support and cur > z_hi: return False
    signals = 0
    for _, row in touched.iterrows():
        body = abs(row["close"] - row["open"])
        if body == 0: continue
        if is_support:
            lw = min(row["open"], row["close"]) - row["low"]
            if row["close"] >= row["open"] and lw > body * 0.5: signals += 1
        else:
            uw = row["high"] - max(row["open"], row["close"])
            if row["close"] < row["open"] and uw > body * 0.5: signals += 1
    if calc_volume_ok(df_5m): signals += 1
    return signals >= 1

# ── 핵심 신호 계산
def compute_signal(symbol, display, funding_rate, btc_dir=None):
    df_4h  = fetch_ohlcv(symbol, "4h",  300)
    df_1h  = fetch_ohlcv(symbol, "1h",  200)
    df_15m = fetch_ohlcv(symbol, "15m", 300)
    df_5m  = fetch_ohlcv(symbol, "5m",  200)
    if any(d is None for d in [df_4h, df_1h, df_15m, df_5m]):
        return None

    price = df_15m["close"].iloc[-1]

    # ATR 변동성
    atr_s = calc_atr(df_15m)
    atr_now = atr_s.iloc[-1]
    atr_avg = atr_s.rolling(20).mean().iloc[-1]
    if atr_avg and atr_now / atr_avg > 2.0:
        return None  # 고변동성 → 무시

    # Stage 1: 4H 방향
    ema_s = calc_ema(df_4h, (9, 21, 50, 200))[0]
    ms_s  = calc_market_structure(df_4h)
    ich_s = calc_ichimoku_score(df_4h)
    dscore = ema_s * 0.4 + ms_s * 0.35 + ich_s * 0.25
    direction = "bull" if dscore > 0.20 else ("bear" if dscore < -0.20 else "neutral")

    # Stage 2: 1H 구조
    struct = calc_market_structure(df_1h) * 0.5 + calc_ema(df_1h, (9, 21, 50, 100))[0] * 0.3 + calc_rsi_score(df_1h) * 0.2
    aligned = (direction == "bull" and struct > 0) or (direction == "bear" and struct < 0)

    if direction == "neutral" or (not aligned):
        return None  # 방향 불명확 또는 불일치

    # BTC 연동 (ETH 전용)
    if btc_dir and direction == "bull" and btc_dir == "bear": return None
    if btc_dir and direction == "bear" and btc_dir == "bull": return None

    # Stage 3: S/R 구간
    sr_zones = find_sr_zones(df_15m, price)
    zone_price, zone_type = None, None
    for lv, zt, _ in sr_zones:
        dist = abs(lv - price) / price
        if dist <= 0.008:
            if direction == "bull" and zt == "지지":
                zone_price, zone_type = lv, zt; break
            elif direction == "bear" and zt == "저항":
                zone_price, zone_type = lv, zt; break

    if zone_price is None: return None

    # Stage 4: 5m 반응
    if not check_reaction(df_5m, zone_price, zone_type == "지지"):
        return None

    # 손절 / 목표가
    trade_dir = "LONG" if zone_type == "지지" else "SHORT"
    is_long   = trade_dir == "LONG"
    sr_levels = find_sr_levels(df_15m, price)
    sl        = get_atr_stop(df_15m, price, is_long, sr_levels, mult=0.8)
    sups = sorted([(lv, t) for lv, t in sr_levels if lv < price], reverse=True)
    ress = sorted([(lv, t) for lv, t in sr_levels if lv > price])

    risk = price - sl if is_long else sl - price
    tp1 = (ress[0][0] if ress else price + risk * 2.0) if is_long else (sups[0][0] if sups else price - risk * 2.0)
    tp2 = (ress[1][0] if len(ress) > 1 else price + risk * 3.5) if is_long else (sups[1][0] if len(sups) > 1 else price - risk * 3.5)

    fee_rate = 0.0016
    net_gain = abs(tp1 - price) - price * fee_rate
    net_loss = abs(price - sl) + price * fee_rate
    rr1 = net_gain / net_loss if net_loss > 0 else 0

    # 최소 조건 필터
    if rr1 < 1.5: return None
    if abs(tp1 - price) / price < 0.003: return None

    # 패턴 라벨
    if direction == "bull": pattern = "눌림목"
    elif direction == "neutral": pattern = "지지반응" if is_long else "저항반응"
    else: pattern = "역추세반등" if is_long else "저항반응"

    return {
        "decision": trade_dir, "pattern": pattern,
        "price": price, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr1": rr1, "zone_price": zone_price, "zone_type": zone_type,
        "symbol": display, "funding": funding_rate,
    }

def get_btc_direction():
    df = fetch_ohlcv("BTC/USDT:USDT", "4h", 150)
    if df is None: return None
    score = (calc_ema(df, (9,21,50,200))[0] * 0.4
             + calc_market_structure(df) * 0.35
             + calc_ichimoku_score(df) * 0.25)
    if score > 0.20: return "bull"
    elif score < -0.20: return "bear"
    return "neutral"

# ── 메인 루프
def main():
    logging.info("텔레그램 신호 봇 시작")

    last = {"BTC": None, "ETH": None}

    while True:
        try:
            btc_dir = get_btc_direction()
            targets = [
                ("BTC/USDT:USDT", "BTC", None),
                ("ETH/USDT:USDT", "ETH", btc_dir),
            ]
            for symbol, display, btc_arg in targets:
                fr = fetch_funding(symbol)
                sig = compute_signal(symbol, display, fr, btc_arg)

                if sig and sig["decision"] in ("LONG", "SHORT"):
                    key = f"{display}_{sig['decision']}_{sig['zone_price']:.0f}"
                    if last[display] != key:
                        last[display] = key
                        em = "🟢" if sig["decision"] == "LONG" else "🔴"
                        sl_pct  = (sig["sl"]  - sig["price"]) / sig["price"] * 100
                        tp1_pct = (sig["tp1"] - sig["price"]) / sig["price"] * 100
                        now_str = datetime.now(KST).strftime("%H:%M")
                        msg = (
                            f"{em} *{display} {sig['decision']}* — {sig['pattern']}\n"
                            f"진입 `${sig['price']:,.2f}`\n"
                            f"손절 `${sig['sl']:,.2f}` ({sl_pct:+.2f}%)\n"
                            f"TP1  `${sig['tp1']:,.2f}` ({tp1_pct:+.2f}%) · R:R {sig['rr1']:.1f}\n"
                            f"🚫 `${sig['sl']:,.2f}` 이탈 시 즉시 손절\n"
                            f"_{now_str} KST_"
                        )
                        send(msg)
                        logging.info(f"신호 전송: {display} {sig['decision']}")
                else:
                    last[display] = None

        except Exception as e:
            logging.error(f"체크 오류: {e}")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()

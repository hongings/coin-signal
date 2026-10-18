import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import ta
import requests
import xml.etree.ElementTree as ET
from scipy.signal import find_peaks
from datetime import datetime, timezone, timedelta
KST = timezone(timedelta(hours=9))
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ── 텔레그램 노티파이어 (모듈 캐시로 1회만 실행)
import notifier_thread  # noqa: F401

# ══════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════
st.set_page_config(page_title="선물 단타 분석기", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container { padding-top: 0.8rem; }
  .signal-card {
    border-radius: 10px; padding: 11px 9px;
    text-align: center; font-size: 12px;
    line-height: 1.5; min-height: 110px;
  }
  .signal-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 16px;
  }
  .section-title {
    font-size: 10px; font-weight: 700;
    letter-spacing: 2px; color: #555;
    text-transform: uppercase; margin: 14px 0 8px;
  }
  @media (max-width: 640px) {
    .block-container { padding: 0.3rem 0.3rem !important; }
    .signal-grid { grid-template-columns: repeat(3, 1fr) !important; gap: 5px !important; }
    .signal-card { padding: 8px 5px !important; min-height: 90px !important; font-size: 11px !important; }
    .signal-card div[style*="font-size:18px"] { font-size: 14px !important; }
    .signal-card div[style*="font-size:9px"] { font-size: 8px !important; }
    h2 { font-size: 1.2rem !important; }
  }
</style>
""", unsafe_allow_html=True)

st_autorefresh(interval=60_000, key="ar")


# ══════════════════════════════════════════════
# 데이터 수집
# ══════════════════════════════════════════════
def _exchange():
    return ccxt.okx({"options": {"defaultType": "swap"}})

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol, timeframe, limit=300):
    try:
        raw = _exchange().fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"OHLCV 로딩 실패: {e}"); return None

@st.cache_data(ttl=60)
def fetch_funding(symbol):
    try:
        info = _exchange().fetch_funding_rate(symbol)
        return float(info.get("fundingRate", 0))
    except: return 0.0

@st.cache_data(ttl=1800)
def fetch_fear_greed():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except: return None, None

@st.cache_data(ttl=300)
def fetch_ls_ratio(usdt_symbol="BTCUSDT"):
    # Bybit 먼저 시도 (사용자 거래소)
    try:
        r = requests.get("https://api.bybit.com/v5/market/account-ratio",
                         params={"category": "linear", "symbol": usdt_symbol,
                                 "period": "1h", "limit": 1}, timeout=5)
        d = r.json()["result"]["list"][0]
        return float(d["buyRatio"])*100, float(d["sellRatio"])*100
    except: pass
    # Binance 폴백
    try:
        r = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                         params={"symbol": usdt_symbol, "period": "1h", "limit": 1}, timeout=5)
        d = r.json()[0]
        return float(d["longAccount"])*100, float(d["shortAccount"])*100
    except: return None, None

@st.cache_data(ttl=1800)
def fetch_dominance():
    try:
        d = requests.get("https://api.coingecko.com/api/v3/global", timeout=8).json()["data"]
        mp = d["market_cap_percentage"]
        return round(mp.get("btc", 0), 1), round(mp.get("eth", 0), 1)
    except: return None, None

@st.cache_data(ttl=3600)
def fetch_macro_events():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=8)
        root = ET.fromstring(r.content)
        events = []
        for ev in root.findall("event"):
            def gt(tag): el = ev.find(tag); return el.text if el is not None else ""
            if gt("country") != "USD" or gt("impact") not in ["High","Medium"]: continue
            events.append({"title": gt("title"), "date": gt("date"),
                           "time": gt("time"), "impact": gt("impact")})
        return events
    except: return []

@st.cache_data(ttl=60)
def fetch_open_interest(usdt_symbol="BTCUSDT"):
    try:
        r = requests.get("https://api.bybit.com/v5/market/open-interest",
                         params={"category": "linear", "symbol": usdt_symbol,
                                 "intervalTime": "1h", "limit": 2}, timeout=5)
        data = r.json()["result"]["list"]
        if len(data) >= 2:
            curr = float(data[0]["openInterest"])
            prev = float(data[1]["openInterest"])
            chg = (curr - prev) / prev * 100 if prev > 0 else 0
            return curr, chg
        return None, None
    except:
        return None, None


# ══════════════════════════════════════════════
# 기술 지표 계산
# ══════════════════════════════════════════════

def calc_ema(df, periods=(9, 21, 50, 200)):
    c = df["close"]; price = c.iloc[-1]
    series = {p: ta.trend.ema_indicator(c, window=p) for p in periods}
    above = sum(price > s.iloc[-1] for s in series.values()
                if not np.isnan(s.iloc[-1]))
    n = len(periods)
    score = (above - n/2) / (n/2)
    top2 = [f"EMA{k}:{'위' if price>s.iloc[-1] else '아래'}"
            for k,s in series.items() if not np.isnan(s.iloc[-1])][:2]
    return score, "  ".join(top2)+f"\n({above}/{n} 위)", series

def calc_bb(df):
    c = df["close"]
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    u,l,m = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    rng = u.iloc[-1]-l.iloc[-1]; price = c.iloc[-1]
    if rng==0 or np.isnan(rng): return 0,"계산 불가",None,None,None
    pos = (price-l.iloc[-1])/rng; bw = (rng/m.iloc[-1])*100
    if pos<=0.10: sc,lb = 1.0,f"하단터치({pos:.0%}) 과매도"
    elif pos<=0.25: sc,lb = 0.6,f"하단근접({pos:.0%})"
    elif pos>=0.90: sc,lb = -1.0,f"상단터치({pos:.0%}) 과매수"
    elif pos>=0.75: sc,lb = -0.6,f"상단근접({pos:.0%})"
    else: sc,lb = (0.5-pos)*0.4,f"중간구간({pos:.0%})"
    return sc, lb+f"\n밴드폭 {bw:.1f}%", u, l, m

def _divergence(price_arr, rsi_arr, lookback=60):
    p = np.array(price_arr[-lookback:], dtype=float)
    r = np.array(rsi_arr[-lookback:], dtype=float)
    if np.any(np.isnan(r)): return 0,""
    prom = p.std()*0.3
    peaks,_ = find_peaks(p, distance=5, prominence=prom)
    troughs,_ = find_peaks(-p, distance=5, prominence=prom)
    if len(peaks)>=2:
        i1,i2 = peaks[-2],peaks[-1]; diff = r[i1]-r[i2]
        if p[i2]>p[i1] and diff>2: return -min(0.9,0.4+diff/20),"🔻 베어리시 다이버전스"
    if len(troughs)>=2:
        i1,i2 = troughs[-2],troughs[-1]; diff = r[i2]-r[i1]
        if p[i2]<p[i1] and diff>2: return min(0.9,0.4+diff/20),"🔺 불리시 다이버전스"
    return 0,""

def calc_rsi(df):
    c = df["close"]
    rsi_s = ta.momentum.RSIIndicator(c, window=14).rsi()
    rsi = rsi_s.iloc[-1]
    if np.isnan(rsi): return 0,"계산 불가",50,rsi_s
    if rsi<25: base,lb = 1.0,f"RSI {rsi:.0f} 극단 과매도"
    elif rsi<35: base,lb = 0.7,f"RSI {rsi:.0f} 과매도"
    elif rsi<45: base,lb = 0.3,f"RSI {rsi:.0f} 약한 롱"
    elif rsi>75: base,lb = -1.0,f"RSI {rsi:.0f} 극단 과매수"
    elif rsi>65: base,lb = -0.7,f"RSI {rsi:.0f} 과매수"
    elif rsi>55: base,lb = -0.3,f"RSI {rsi:.0f} 약한 숏"
    else: base,lb = 0.0,f"RSI {rsi:.0f} 중립"
    ds,dl = _divergence(c.values, rsi_s.values)
    if ds!=0:
        lb += "\n"+dl
        total = np.clip(base*0.6+ds*0.4 if ds*base>=0 else ds*0.7+base*0.3,-1,1)
    else: total = base
    return total,lb,rsi,rsi_s

def calc_macd(df):
    c = df["close"]
    m = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    ml,sl,hl = m.macd(), m.macd_signal(), m.macd_diff()
    mv,sv,hv,hp = ml.iloc[-1],sl.iloc[-1],hl.iloc[-1],hl.iloc[-2]
    if any(np.isnan(x) for x in [mv,sv,hv]): return 0,"계산 불가",ml,sl,hl
    if mv>sv and hv>0:
        sc,lb = (0.8 if hv>hp else 0.4), ("골든크로스↑ 히스토 증가" if hv>hp else "골든크로스↑ 히스토 감소")
    elif mv<sv and hv<0:
        sc,lb = (-0.8 if hv<hp else -0.4), ("데드크로스↓ 히스토 증가" if hv<hp else "데드크로스↓ 히스토 감소")
    else: sc,lb = 0,"크로스 없음"
    if mv>0 and ml.iloc[-5]<0: sc=min(1.0,sc+0.2); lb+=" 제로선↑"
    elif mv<0 and ml.iloc[-5]>0: sc=max(-1.0,sc-0.2); lb+=" 제로선↓"
    return sc,lb,ml,sl,hl

def calc_stoch_rsi(df):
    c = df["close"]
    sr = ta.momentum.StochRSIIndicator(c, window=14, smooth1=3, smooth2=3)
    k,d = sr.stochrsi_k(), sr.stochrsi_d()
    kv,dv,kp = k.iloc[-1]*100, d.iloc[-1]*100, k.iloc[-2]*100
    if np.isnan(kv): return 0,"계산 불가",k,d
    if kv<20: sc,lb = 0.9,f"K:{kv:.0f} 극단 과매도"
    elif kv<30: sc,lb = 0.6,f"K:{kv:.0f} 과매도"
    elif kv>80: sc,lb = -0.9,f"K:{kv:.0f} 극단 과매수"
    elif kv>70: sc,lb = -0.6,f"K:{kv:.0f} 과매수"
    else: sc,lb = 0,f"K:{kv:.0f} 중립"
    if kv>dv and kp<=dv:
        sc=min(1.0,sc+0.3); lb+="\nK↑D 골든크로스"
    elif kv<dv and kp>=dv:
        sc=max(-1.0,sc-0.3); lb+="\nK↓D 데드크로스"
    return sc,lb,k,d

def calc_ichimoku(df):
    h,l,c = df["high"],df["low"],df["close"]
    ichi = ta.trend.IchimokuIndicator(h,l,window1=9,window2=26,window3=52)
    tenkan,kijun = ichi.ichimoku_conversion_line(),ichi.ichimoku_base_line()
    sa,sb = ichi.ichimoku_a(),ichi.ichimoku_b()
    price,tv,kv = c.iloc[-1],tenkan.iloc[-1],kijun.iloc[-1]
    if len(df)<36: return 0,"데이터 부족",None,None,None,None
    sa_now,sb_now = sa.iloc[-26],sb.iloc[-26]
    cloud_top,cloud_bot = max(sa_now,sb_now),min(sa_now,sb_now)
    bull = sa.iloc[-1]>sb.iloc[-1]
    sc=0; dt=[]
    if price>cloud_top: sc+=0.35; dt.append("구름위↑")
    elif price<cloud_bot: sc-=0.35; dt.append("구름아래↓")
    else: dt.append("구름안")
    if not (np.isnan(tv) or np.isnan(kv)):
        if tv>kv: sc+=0.30; dt.append("전환>기준↑")
        else: sc-=0.30; dt.append("전환<기준↓")
    sc += 0.20 if bull else -0.20; dt.append("미래구름"+("녹↑" if bull else "적↓"))
    if not np.isnan(kv):
        if price>kv: sc+=0.15; dt.append("가격>기준선")
        else: sc-=0.15; dt.append("가격<기준선")
    return np.clip(sc,-1,1),"\n".join(dt),tenkan,kijun,sa,sb

def calc_market_structure(df, lookback=80):
    h = df["high"].values[-lookback:]
    l = df["low"].values[-lookback:]
    peaks,_ = find_peaks(h, distance=5, prominence=h.std()*0.3)
    troughs,_ = find_peaks(-l, distance=5, prominence=l.std()*0.3)
    if len(peaks)<2 or len(troughs)<2:
        return 0,"구조 불명확",peaks,troughs
    sh1,sh2 = h[peaks[-2]],h[peaks[-1]]
    sl1,sl2 = l[troughs[-2]],l[troughs[-1]]
    hh,hl = sh2>sh1, sl2>sl1
    lh,ll = sh2<sh1, sl2<sl1
    if hh and hl:   return 0.7,"상승구조 HH+HL↑",peaks,troughs
    elif lh and ll: return -0.7,"하락구조 LH+LL↓",peaks,troughs
    elif hh and ll: return 0,"확산 횡보 HH+LL",peaks,troughs
    elif lh and hl: return 0,"수렴 횡보 LH+HL",peaks,troughs
    else:           return 0,"구조 불명확",peaks,troughs

def calc_volume(df):
    vol,close = df["volume"],df["close"]
    avg = vol.rolling(20).mean().iloc[-1]
    if np.isnan(avg) or avg==0: return 0,"계산 불가"
    ratio = vol.iloc[-1]/avg
    d = 1 if close.iloc[-1]>close.iloc[-2] else -1
    if ratio>2.0: return 0.9*d,f"거래량 {ratio:.1f}x\n강한 {'매수' if d>0 else '매도'}"
    elif ratio>1.5: return 0.6*d,f"거래량 {ratio:.1f}x\n{'매수' if d>0 else '매도'} 확인"
    elif ratio>1.2: return 0.3*d,f"거래량 {ratio:.1f}x\n약한 신호"
    else: return 0,f"거래량 {ratio:.1f}x\n평균 수준"

def calc_vwap(df):
    vwap = ta.volume.VolumeWeightedAveragePrice(
        df["high"], df["low"], df["close"], df["volume"], window=14
    ).volume_weighted_average_price()
    price = df["close"].iloc[-1]
    vv = vwap.iloc[-1]
    if np.isnan(vv): return 0, "계산 불가", vwap
    diff = (price - vv) / vv * 100
    if diff >  0.5: sc, lb = min(0.9, 0.5 + diff*0.1), f"VWAP 위 +{diff:.2f}%\n매수 우위"
    elif diff < -0.5: sc, lb = max(-0.9, -0.5 + diff*0.1), f"VWAP 아래 {diff:.2f}%\n매도 우위"
    else: sc, lb = 0, f"VWAP 근접 {diff:+.2f}%\n중립"
    return sc, lb, vwap

def calc_funding(rate):
    p = rate*100
    if p>0.15:   return -0.9,f"+{p:.4f}%\n롱 극단 과열"
    elif p>0.07: return -0.5,f"+{p:.4f}%\n롱 과열"
    elif p>0.02: return -0.2,f"+{p:.4f}%\n롱 소폭 우세"
    elif p<-0.15:return  0.9,f"{p:.4f}%\n숏 극단 과열"
    elif p<-0.07:return  0.5,f"{p:.4f}%\n숏 과열"
    elif p<-0.02:return  0.2,f"{p:.4f}%\n숏 소폭 우세"
    else:        return  0, f"{p:.4f}%\n중립"


# ══════════════════════════════════════════════
# ATR 손절 + 지지저항
# ══════════════════════════════════════════════

def calc_atr(df, period=14):
    return ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=period
    ).average_true_range()

def find_sr_levels(df, price, lookback=120, tol=0.008):
    h,l = df["high"].values[-lookback:],df["low"].values[-lookback:]
    peaks,_ = find_peaks(h, distance=5, prominence=h.std()*0.25)
    troughs,_ = find_peaks(-l, distance=5, prominence=l.std()*0.25)
    res,sup = h[peaks].tolist(), l[troughs].tolist()
    mag = 10**max(3,int(np.log10(price))-1)
    for m in range(-6,7):
        lv = round(price/mag)*mag+m*mag
        (res if lv>price else sup).append(float(lv))
    def cluster(lvs):
        if not lvs: return []
        lvs=sorted(set(lvs)); groups=[[lvs[0]]]
        for lv in lvs[1:]:
            if (lv-groups[-1][-1])/max(groups[-1][-1],1)<tol: groups[-1].append(lv)
            else: groups.append([lv])
        return [float(np.mean(g)) for g in groups]
    r_lvs = [(lv,"저항") for lv in cluster(res) if lv>price*0.997]
    s_lvs = [(lv,"지지") for lv in cluster(sup) if lv<price*1.003]
    all_lvs = r_lvs+s_lvs
    all_lvs.sort(key=lambda x: abs(x[0]-price))
    return all_lvs[:10]

def detect_candle(df):
    last = df.iloc[-1]
    body = abs(last["close"]-last["open"]); total = last["high"]-last["low"]
    if total==0: return "캔들 없음",0,""
    uw = last["high"]-max(last["open"],last["close"])
    lw = min(last["open"],last["close"])-last["low"]
    br = body/total; bull = last["close"]>=last["open"]
    if lw>=body*2 and uw<=body*0.5 and br<0.45:
        return ("🔨 불리시 해머",0.9,"아래꼬리 길고 양봉") if bull else ("🔨 해머(음봉)",0.6,"아래꼬리 길고 음봉")
    if uw>=body*2 and lw<=body*0.5 and br<0.45:
        return ("⭐ 슈팅스타",-0.9,"위꼬리 길고 음봉") if not bull else ("⭐ 역망치(양봉)",-0.5,"위꼬리 길고 양봉")
    if br<0.08: return "十 도지",0,"방향 결정 대기"
    if bull and br>0.70: return "📈 장대양봉",0.7,f"강한 매수세 {br:.0%}"
    if not bull and br>0.70: return "📉 장대음봉",-0.7,f"강한 매도세 {br:.0%}"
    return ("📊 일반 양봉",0.2,f"{br:.0%}") if bull else ("📊 일반 음봉",-0.2,f"{br:.0%}")

def get_atr_stop(df, entry, is_long, sr_levels, mult=1.5):
    atr_val = calc_atr(df).iloc[-1]
    atr_stop = entry-atr_val*mult if is_long else entry+atr_val*mult
    sups = sorted([(lv,t) for lv,t in sr_levels if lv<entry], reverse=True)
    ress = sorted([(lv,t) for lv,t in sr_levels if lv>entry])
    if is_long and sups:
        sr_stop = sups[0][0]*0.9965
        sl = min(atr_stop, sr_stop) if sr_stop>entry*0.90 else atr_stop
    elif not is_long and ress:
        sr_stop = ress[0][0]*1.0035
        sl = max(atr_stop, sr_stop) if sr_stop<entry*1.10 else atr_stop
    else: sl = atr_stop
    return sl, atr_val

def get_trend_filter_1h(df_1h):
    ema_s, _, _ = calc_ema(df_1h)
    ms_s, *_    = calc_market_structure(df_1h)
    score = ema_s*0.6 + ms_s*0.4
    if score > 0.20:    return "bull",    "📈 1H 상승 — 롱 방향 유리"
    elif score < -0.20: return "bear",    "📉 1H 하락 — 숏 방향 유리"
    else:               return "neutral", "↔ 1H 횡보 — 양방향 가능"

def get_5m_confirm(df_5m, direction):
    """5분봉이 15분 신호 방향을 확인하는지 체크 — 단타 정확도 보강"""
    ema_s, _, _    = calc_ema(df_5m, periods=(9, 21, 50, 100))
    rsi_s, _, _, _ = calc_rsi(df_5m)
    stoch_s, _, _, _ = calc_stoch_rsi(df_5m)
    score = ema_s*0.5 + rsi_s*0.3 + stoch_s*0.2
    if direction == "LONG":
        ok = score > 0.05
    else:
        ok = score < -0.05
    label = f"5분 {'✅ 확인' if ok else '❌ 미확인'} (score {score:+.2f})"
    return ok, label


def detect_chart_patterns(df_4h, df_1h, df_15m, price):
    """
    Multi-timeframe chart pattern detector.
    Returns list of dicts: {name, tf, desc, bias, level}. Max 3 patterns.
    """
    patterns = []

    tf_configs = [
        (df_4h,  "4H",  80),
        (df_1h,  "1H",  80),
        (df_15m, "15m", 100),
    ]

    for df, tf_name, lookback in tf_configs:
        if df is None or len(df) < lookback:
            continue

        h = df["high"].values[-lookback:]
        l = df["low"].values[-lookback:]

        peaks,   _ = find_peaks(h, distance=5, prominence=h.std() * 0.3)
        troughs, _ = find_peaks(-l, distance=5, prominence=l.std() * 0.3)

        # Only consider pivots that appear in the last 20-30 bars of the slice
        recent_cutoff = lookback - 30

        def _recent(idx_arr):
            return [i for i in idx_arr if i >= recent_cutoff]

        recent_peaks   = _recent(peaks.tolist())
        recent_troughs = _recent(troughs.tolist())

        # ── 3드라이브 하락 (3 descending troughs → bullish reversal)
        if len(troughs) >= 3:
            t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
            if t3 >= recent_cutoff and l[t1] > l[t2] > l[t3]:
                patterns.append({
                    "name": "3드라이브 하락",
                    "tf": tf_name,
                    "desc": f"3연속 저점 하강 → 강세 반전 가능 (최저 ${l[t3]:,.0f})",
                    "bias": "bull",
                    "level": float(l[t3]),
                })

        # ── 3드라이브 상승 (3 ascending peaks → bearish reversal)
        if len(peaks) >= 3:
            p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
            if p3 >= recent_cutoff and h[p1] < h[p2] < h[p3]:
                patterns.append({
                    "name": "3드라이브 상승",
                    "tf": tf_name,
                    "desc": f"3연속 고점 상승 → 약세 반전 가능 (최고 ${h[p3]:,.0f})",
                    "bias": "bear",
                    "level": float(h[p3]),
                })

        # ── 더블바텀 (2 troughs within 1.5%)
        if len(troughs) >= 2:
            ta_, tb_ = troughs[-2], troughs[-1]
            if tb_ >= recent_cutoff:
                diff = abs(l[ta_] - l[tb_]) / max(l[ta_], l[tb_])
                if diff <= 0.015:
                    avg_lv = (l[ta_] + l[tb_]) / 2
                    patterns.append({
                        "name": "더블바텀",
                        "tf": tf_name,
                        "desc": f"두 저점 근접 (${avg_lv:,.0f}) → 지지 강화, 강세 신호",
                        "bias": "bull",
                        "level": float(avg_lv),
                    })

        # ── 더블탑 (2 peaks within 1.5%)
        if len(peaks) >= 2:
            pa_, pb_ = peaks[-2], peaks[-1]
            if pb_ >= recent_cutoff:
                diff = abs(h[pa_] - h[pb_]) / max(h[pa_], h[pb_])
                if diff <= 0.015:
                    avg_lv = (h[pa_] + h[pb_]) / 2
                    patterns.append({
                        "name": "더블탑",
                        "tf": tf_name,
                        "desc": f"두 고점 근접 (${avg_lv:,.0f}) → 저항 강화, 약세 신호",
                        "bias": "bear",
                        "level": float(avg_lv),
                    })

        # ── 역헤드앤숄더 (middle trough is lowest of 3 → bullish)
        if len(troughs) >= 3:
            s1, head, s2 = troughs[-3], troughs[-2], troughs[-1]
            if s2 >= recent_cutoff and l[head] < l[s1] and l[head] < l[s2]:
                patterns.append({
                    "name": "역헤드앤숄더",
                    "tf": tf_name,
                    "desc": f"역H&S 형성, 헤드 ${l[head]:,.0f} → 강세 반전 신호",
                    "bias": "bull",
                    "level": float(l[head]),
                })

        # ── 헤드앤숄더 (middle peak is highest of 3 → bearish)
        if len(peaks) >= 3:
            s1, head, s2 = peaks[-3], peaks[-2], peaks[-1]
            if s2 >= recent_cutoff and h[head] > h[s1] and h[head] > h[s2]:
                patterns.append({
                    "name": "헤드앤숄더",
                    "tf": tf_name,
                    "desc": f"H&S 형성, 헤드 ${h[head]:,.0f} → 약세 반전 신호",
                    "bias": "bear",
                    "level": float(h[head]),
                })

    # Deduplicate by name (keep first/higher timeframe occurrence), max 3
    seen = set()
    unique = []
    for p in patterns:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique.append(p)
        if len(unique) >= 3:
            break

    return unique


def get_pattern_chart_annotations(df, price, lookback=100):
    """
    15m 차트에 직접 그릴 패턴 피벗 + 넥라인 데이터 반환.
    Returns list of annotation dicts for build_chart.
    """
    annotations = []
    lb = min(lookback, len(df))
    h  = df["high"].values[-lb:]
    l  = df["low"].values[-lb:]
    ts = df["ts"].values[-lb:]
    n  = len(h)
    recent = n - 30

    peaks,   _ = find_peaks(h, distance=5, prominence=h.std() * 0.3)
    troughs, _ = find_peaks(-l, distance=5, prominence=l.std() * 0.3)

    # ── 더블바텀
    if len(troughs) >= 2:
        ta_, tb_ = int(troughs[-2]), int(troughs[-1])
        if tb_ >= recent and abs(l[ta_] - l[tb_]) / max(l[ta_], l[tb_]) <= 0.015:
            mid_peaks = [pk for pk in peaks if ta_ < pk < tb_]
            nk = float(h[mid_peaks[0]]) if mid_peaks else price * 1.015
            target = nk + (nk - min(l[ta_], l[tb_]))
            mid_pt = (ts[mid_peaks[0]], float(h[mid_peaks[0]])) if mid_peaks else None
            line_pts = [(ts[ta_], float(l[ta_]))]
            if mid_pt: line_pts.append(mid_pt)
            line_pts.append((ts[tb_], float(l[tb_])))
            annotations.append({
                "name": "더블바텀", "bias": "bull",
                "pivots": [(ts[ta_], float(l[ta_]), "①"), (ts[tb_], float(l[tb_]), "②")],
                "neckline": nk, "target": target,
                "fail": min(l[ta_], l[tb_]) * 0.995,
                "marker_color": "#5de8a0", "anchor": "bottom center",
                "zone_start": ts[ta_], "zone_end": ts[tb_],
                "line_points": line_pts,
            })

    # ── 더블탑
    if len(peaks) >= 2:
        pa_, pb_ = int(peaks[-2]), int(peaks[-1])
        if pb_ >= recent and abs(h[pa_] - h[pb_]) / max(h[pa_], h[pb_]) <= 0.015:
            mid_trs = [tr for tr in troughs if pa_ < tr < pb_]
            nk = float(l[mid_trs[0]]) if mid_trs else price * 0.985
            target = nk - (max(h[pa_], h[pb_]) - nk)
            mid_pt = (ts[mid_trs[0]], float(l[mid_trs[0]])) if mid_trs else None
            line_pts = [(ts[pa_], float(h[pa_]))]
            if mid_pt: line_pts.append(mid_pt)
            line_pts.append((ts[pb_], float(h[pb_])))
            annotations.append({
                "name": "더블탑", "bias": "bear",
                "pivots": [(ts[pa_], float(h[pa_]), "①"), (ts[pb_], float(h[pb_]), "②")],
                "neckline": nk, "target": target,
                "fail": max(h[pa_], h[pb_]) * 1.005,
                "marker_color": "#e85d5d", "anchor": "top center",
                "zone_start": ts[pa_], "zone_end": ts[pb_],
                "line_points": line_pts,
            })

    # ── 3드라이브 하락
    if len(troughs) >= 3:
        t1, t2, t3 = int(troughs[-3]), int(troughs[-2]), int(troughs[-1])
        if t3 >= recent and l[t1] > l[t2] > l[t3]:
            target = l[t3] + (l[t1] - l[t3]) * 0.618
            annotations.append({
                "name": "3드라이브 하락", "bias": "bull",
                "pivots": [(ts[t1], float(l[t1]), "①"), (ts[t2], float(l[t2]), "②"), (ts[t3], float(l[t3]), "③")],
                "neckline": None, "target": target,
                "fail": l[t3] * 0.995,
                "marker_color": "#5de8a0", "anchor": "bottom center",
                "zone_start": ts[t1], "zone_end": ts[t3],
                "line_points": [(ts[t1], float(l[t1])), (ts[t2], float(l[t2])), (ts[t3], float(l[t3]))],
            })

    # ── 3드라이브 상승
    if len(peaks) >= 3:
        p1, p2, p3 = int(peaks[-3]), int(peaks[-2]), int(peaks[-1])
        if p3 >= recent and h[p1] < h[p2] < h[p3]:
            target = h[p3] - (h[p3] - h[p1]) * 0.618
            annotations.append({
                "name": "3드라이브 상승", "bias": "bear",
                "pivots": [(ts[p1], float(h[p1]), "①"), (ts[p2], float(h[p2]), "②"), (ts[p3], float(h[p3]), "③")],
                "neckline": None, "target": target,
                "fail": h[p3] * 1.005,
                "marker_color": "#e85d5d", "anchor": "top center",
                "zone_start": ts[p1], "zone_end": ts[p3],
                "line_points": [(ts[p1], float(h[p1])), (ts[p2], float(h[p2])), (ts[p3], float(h[p3]))],
            })

    # ── 역헤드앤숄더
    if len(troughs) >= 3:
        s1, hd, s2 = int(troughs[-3]), int(troughs[-2]), int(troughs[-1])
        if s2 >= recent and l[hd] < l[s1] and l[hd] < l[s2]:
            mid_peaks = [pk for pk in peaks if s1 < pk < s2]
            nk = float(np.mean([h[pk] for pk in mid_peaks])) if mid_peaks else price * 1.02
            target = nk + (nk - l[hd])
            annotations.append({
                "name": "역헤드앤숄더", "bias": "bull",
                "pivots": [(ts[s1], float(l[s1]), "①숄"), (ts[hd], float(l[hd]), "②헤"), (ts[s2], float(l[s2]), "③숄")],
                "neckline": nk, "target": target,
                "fail": l[hd] * 0.995,
                "marker_color": "#5de8a0", "anchor": "bottom center",
                "zone_start": ts[s1], "zone_end": ts[s2],
                "line_points": [(ts[s1], float(l[s1])), (ts[hd], float(l[hd])), (ts[s2], float(l[s2]))],
            })

    # ── 헤드앤숄더
    if len(peaks) >= 3:
        s1, hd, s2 = int(peaks[-3]), int(peaks[-2]), int(peaks[-1])
        if s2 >= recent and h[hd] > h[s1] and h[hd] > h[s2]:
            mid_trs = [tr for tr in troughs if s1 < tr < s2]
            nk = float(np.mean([l[tr] for tr in mid_trs])) if mid_trs else price * 0.98
            target = nk - (h[hd] - nk)
            annotations.append({
                "name": "헤드앤숄더", "bias": "bear",
                "pivots": [(ts[s1], float(h[s1]), "①숄"), (ts[hd], float(h[hd]), "②헤"), (ts[s2], float(h[s2]), "③숄")],
                "neckline": nk, "target": target,
                "fail": h[hd] * 1.005,
                "marker_color": "#e85d5d", "anchor": "top center",
                "zone_start": ts[s1], "zone_end": ts[s2],
                "line_points": [(ts[s1], float(h[s1])), (ts[hd], float(h[hd])), (ts[s2], float(h[s2]))],
            })

    return annotations[:2]


def detect_momentum_exhaustion(df_15m, df_5m):
    """
    Count consecutive same-direction candles from the last 10 bars.
    Check 5m first, then 15m. If count >= 4 return exhaustion signal.
    """
    empty = {"exhausted": False, "direction": "", "count": 0, "tf": "", "label": ""}

    for df, tf_name in [(df_5m, "5m"), (df_15m, "15m")]:
        if df is None or len(df) < 5:
            continue
        last10 = df.tail(10)
        candles = [(row["close"] >= row["open"]) for _, row in last10.iterrows()]
        # Count consecutive from the end
        last_dir = candles[-1]
        count = 0
        for c in reversed(candles):
            if c == last_dir:
                count += 1
            else:
                break
        if count >= 4:
            direction = "bull" if last_dir else "bear"
            dir_str   = "상승" if last_dir else "하락"
            opp       = "숏" if last_dir else "롱"
            label     = f"⚠️ {tf_name} {dir_str} {count}연속 — {opp} 신호 신중"
            return {
                "exhausted": True,
                "direction": direction,
                "count": count,
                "tf": tf_name,
                "label": label,
            }

    return empty


# ══════════════════════════════════════════════
# 차트
# ══════════════════════════════════════════════

def build_chart(df, ema_series, bb_u, bb_l, bb_m,
                tenkan, kijun, sa, sb,
                rsi_s, macd_l, macd_sig, macd_hist,
                stoch_k, stoch_d,
                sr_levels, ms_peaks, ms_troughs, rec, name,
                vwap=None, tf_label="4H", pattern_annotations=None):
    n=100; dp=df.tail(n).copy(); idx=dp.index

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.52,0.14,0.18,0.16],
        subplot_titles=(f"{name} {tf_label}", "거래량", "RSI / Stoch RSI", "MACD"),
    )

    # ── 캔들
    fig.add_trace(go.Candlestick(
        x=dp["ts"], open=dp["open"], high=dp["high"], low=dp["low"], close=dp["close"],
        name=name, increasing_line_color="#5de8a0", decreasing_line_color="#e85d5d",
        increasing_fillcolor="#5de8a0", decreasing_fillcolor="#e85d5d", line_width=1,
    ), row=1, col=1)

    # ── 볼린저밴드
    if bb_u is not None:
        for s,n_,dash in [(bb_u,"BBU","solid"),(bb_l,"BBL","solid"),(bb_m,"BBM","dot")]:
            kw = dict(fill="tonexty",fillcolor="rgba(120,120,200,0.06)") if n_=="BBL" else {}
            fig.add_trace(go.Scatter(x=dp["ts"],y=s.loc[idx],name=n_,
                line=dict(color="rgba(120,120,200,0.4)",width=1,dash=dash),
                showlegend=False, **kw), row=1, col=1)

    # ── 일목 구름
    if sa is not None:
        ca,cb = sa.shift(26).loc[idx], sb.shift(26).loc[idx]
        fig.add_trace(go.Scatter(x=dp["ts"],y=ca,name="Senkou A",
            line=dict(color="rgba(93,232,160,0.5)",width=1),showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=dp["ts"],y=cb,name="Senkou B",
            line=dict(color="rgba(232,93,93,0.5)",width=1),
            fill="tonexty",fillcolor="rgba(93,232,160,0.07)",showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=dp["ts"],y=tenkan.loc[idx],name="전환선",
            line=dict(color="#00BFFF",width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dp["ts"],y=kijun.loc[idx],name="기준선",
            line=dict(color="#FF8C00",width=1.2)), row=1, col=1)

    # ── EMA (ema_series 키 기준으로 순회 — 스윙/단타 모두 대응)
    _ema_colors = {9:"#FFD700", 21:"#FF6B9D", 50:"#00BFFF", 100:"#88DDFF", 200:"#C77DFF"}
    for p in ema_series:
        clr = _ema_colors.get(p, "#AAAAAA")
        fig.add_trace(go.Scatter(x=dp["ts"],y=ema_series[p].loc[idx],
            name=f"EMA{p}",line=dict(color=clr,width=1.0)), row=1, col=1)

    # ── VWAP (단타용)
    if vwap is not None:
        fig.add_trace(go.Scatter(x=dp["ts"], y=vwap.loc[idx], name="VWAP",
            line=dict(color="#FF00FF", width=1.5, dash="dot")), row=1, col=1)

    # ── 지지저항 (상위 4개만)
    for lv,lt in sr_levels[:4]:
        c = "rgba(232,93,93,0.55)" if lt=="저항" else "rgba(93,232,160,0.55)"
        fig.add_hline(y=lv,line_color=c,line_dash="dash",line_width=1,
            annotation_text=f"  {lt} ${lv:,.0f}",
            annotation_font_size=9,annotation_font_color=c,row=1,col=1)

    # ── 패턴 마킹 (배경 음영 + 연결선 + 피벗 마커 + 넥라인 + 목표가)
    if pattern_annotations:
        for ann in pattern_annotations[:2]:
            mc  = ann["marker_color"]
            anc = ann["anchor"]
            is_bull = ann["bias"] == "bull"

            # 1) 배경 음영 — 패턴 구간 강조
            if ann.get("zone_start") is not None and ann.get("zone_end") is not None:
                shade = "rgba(0,200,100,0.10)" if is_bull else "rgba(255,80,80,0.10)"
                fig.add_vrect(
                    x0=ann["zone_start"], x1=ann["zone_end"],
                    fillcolor=shade, layer="below", line_width=0,
                    row=1, col=1,
                )

            # 2) 피봇 연결선 — 패턴 모양 직접 표현
            if ann.get("line_points"):
                lx = [p[0] for p in ann["line_points"]]
                ly = [p[1] for p in ann["line_points"]]
                line_col = "#5de8a0" if is_bull else "#e85d5d"
                fig.add_trace(go.Scatter(
                    x=lx, y=ly, mode="lines",
                    line=dict(color=line_col, width=2, dash="solid"),
                    name=f"{ann['name']} 선", showlegend=False,
                ), row=1, col=1)

            # 3) 피봇 포인트 마커 (◆)
            px = [p[0] for p in ann["pivots"]]
            py = [p[1] for p in ann["pivots"]]
            pt_labels = [p[2] for p in ann["pivots"]]
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers+text",
                text=pt_labels, textposition=anc,
                marker=dict(symbol="diamond", size=11, color=mc,
                            line=dict(color="#fff", width=1)),
                name=ann["name"], showlegend=True,
                textfont=dict(color=mc, size=11),
            ), row=1, col=1)

            # 4) 넥라인
            if ann.get("neckline"):
                nk_col = "#e8c85d"
                fig.add_hline(y=ann["neckline"], line_color=nk_col,
                    line_dash="dashdot", line_width=1.5,
                    annotation_text=f"  넥라인 ${ann['neckline']:,.0f}",
                    annotation_font_color=nk_col, annotation_font_size=9, row=1, col=1)

            # 5) 목표가
            if ann.get("target"):
                tg_col = mc
                fig.add_hline(y=ann["target"], line_color=tg_col,
                    line_dash="dot", line_width=1,
                    annotation_text=f"  목표 ${ann['target']:,.0f}",
                    annotation_font_color=tg_col, annotation_font_size=9, row=1, col=1)

    # ── TP/SL 라인
    if rec:
        il = rec["direction"]=="LONG"
        ec = "#5de8a0" if il else "#e85d5d"
        for y,lbl,c,dash in [
            (rec["entry"],f"진입 ${rec['entry']:,.0f}",ec,"solid"),
            (rec["sl"],f"SL ${rec['sl']:,.0f}","#ff3333","dot"),
            (rec["tp1"],f"TP1 ${rec['tp1']:,.0f}","#4499ff","dot"),
            (rec["tp2"],f"TP2 ${rec['tp2']:,.0f}","#88bbff","dot"),
        ]:
            fig.add_hline(y=y,line_color=c,line_width=1.5 if lbl.startswith("진입") else 1,
                line_dash=dash,annotation_text=f"  {lbl}",
                annotation_font_color=c,annotation_font_size=10,row=1,col=1)

    # ── 거래량
    vc = ["#5de8a0" if c>=o else "#e85d5d" for c,o in zip(dp["close"],dp["open"])]
    fig.add_trace(go.Bar(x=dp["ts"],y=dp["volume"],marker_color=vc,
        name="거래량",showlegend=False), row=2, col=1)

    # ── RSI (Stoch RSI 차트에서 제거 — 신호 그리드에서 확인)
    fig.add_trace(go.Scatter(x=dp["ts"],y=rsi_s.loc[idx],name="RSI",
        line=dict(color="#DDA0DD",width=1.5)), row=3, col=1)
    for y,c in [(70,"rgba(232,93,93,0.4)"),(30,"rgba(93,232,160,0.4)"),(50,"rgba(150,150,150,0.2)")]:
        fig.add_hline(y=y,line_dash="dash",line_color=c,line_width=1,row=3,col=1)
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(232,93,93,0.05)",row=3,col=1,line_width=0)
    fig.add_hrect(y0=0,y1=30,fillcolor="rgba(93,232,160,0.05)",row=3,col=1,line_width=0)

    # ── MACD
    hist_colors = ["#5de8a0" if v>=0 else "#e85d5d" for v in macd_hist.loc[idx]]
    fig.add_trace(go.Bar(x=dp["ts"],y=macd_hist.loc[idx],marker_color=hist_colors,
        name="MACD Hist",showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=dp["ts"],y=macd_l.loc[idx],name="MACD",
        line=dict(color="#00BFFF",width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=dp["ts"],y=macd_sig.loc[idx],name="Signal",
        line=dict(color="#FF8C00",width=1.2)), row=4, col=1)
    fig.add_hline(y=0,line_dash="dot",line_color="rgba(150,150,150,0.3)",row=4,col=1)

    fig.update_layout(
        template="plotly_dark", height=540,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.02,x=0,font=dict(size=8),itemwidth=40),
        margin=dict(l=4,r=90,t=22,b=4),
        autosize=True,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)")
    # Y축을 실제 캔들 범위로 제한 (원거리 S/R 레벨로 축 왜곡 방지)
    pad = (dp["high"].max() - dp["low"].min()) * 0.06
    fig.update_yaxes(range=[dp["low"].min() - pad, dp["high"].max() + pad], row=1, col=1)
    return fig


def build_pattern_detail_chart(df, ann, display):
    """
    패턴 구간만 줌인한 상세 차트.
    캔들 + 배경음영 + 연결선 + 마커 + 넥라인 + 목표가 만 표시.
    """
    # 패턴 구간에서 앞뒤 25봉 컨텍스트
    ctx = 25
    zone_start = ann.get("zone_start")
    zone_end   = ann.get("zone_end")

    if zone_start is not None and zone_end is not None:
        mask = (df["ts"] >= zone_start) & (df["ts"] <= zone_end)
        idxs = df.index[mask]
        if len(idxs) > 0:
            i0 = max(0, idxs[0] - ctx)
            i1 = min(len(df) - 1, idxs[-1] + ctx)
            dp = df.iloc[i0:i1+1].copy()
        else:
            dp = df.tail(80).copy()
    else:
        dp = df.tail(80).copy()

    mc      = ann["marker_color"]
    is_bull = ann["bias"] == "bull"
    bias_kr = "📈 상승 반전" if is_bull else "📉 하락 반전"
    shade   = "rgba(0,200,100,0.13)" if is_bull else "rgba(255,80,80,0.13)"
    line_col = "#5de8a0" if is_bull else "#e85d5d"

    fig = go.Figure()

    # 캔들
    fig.add_trace(go.Candlestick(
        x=dp["ts"], open=dp["open"], high=dp["high"], low=dp["low"], close=dp["close"],
        name=display,
        increasing_line_color="#5de8a0", decreasing_line_color="#e85d5d",
        increasing_fillcolor="#5de8a0", decreasing_fillcolor="#e85d5d",
        line_width=1,
    ))

    # 배경 음영
    if zone_start is not None and zone_end is not None:
        fig.add_vrect(x0=zone_start, x1=zone_end,
                      fillcolor=shade, layer="below", line_width=0)

    # 피봇 연결선
    if ann.get("line_points"):
        lx = [p[0] for p in ann["line_points"]]
        ly = [p[1] for p in ann["line_points"]]
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode="lines",
            line=dict(color=line_col, width=2.5, dash="solid"),
            name="패턴 선", showlegend=False,
        ))

    # 피봇 마커 (◆ + 가격 레이블)
    for ts_p, pr_p, lbl in ann["pivots"]:
        fig.add_trace(go.Scatter(
            x=[ts_p], y=[pr_p],
            mode="markers+text",
            text=[f"{lbl}<br>${pr_p:,.0f}"],
            textposition=ann["anchor"],
            marker=dict(symbol="diamond", size=14, color=mc,
                        line=dict(color="#fff", width=1.5)),
            showlegend=False,
            textfont=dict(color=mc, size=11),
        ))

    # 넥라인
    if ann.get("neckline"):
        nk = ann["neckline"]
        fig.add_hline(y=nk, line_color="#e8c85d", line_dash="dashdot", line_width=2,
                      annotation_text=f"  넥라인  ${nk:,.0f}",
                      annotation_font_color="#e8c85d", annotation_font_size=11)

    # 목표가
    if ann.get("target"):
        tg = ann["target"]
        fig.add_hline(y=tg, line_color=mc, line_dash="dot", line_width=1.5,
                      annotation_text=f"  목표  ${tg:,.0f}",
                      annotation_font_color=mc, annotation_font_size=11)

    # 손절가 (fail)
    if ann.get("fail"):
        fl = ann["fail"]
        fig.add_hline(y=fl, line_color="#ff4444", line_dash="dot", line_width=1.2,
                      annotation_text=f"  패턴 무효  ${fl:,.0f}",
                      annotation_font_color="#ff4444", annotation_font_size=10)

    # 음영 범례 텍스트
    legend_color = "rgba(0,200,100,0.7)" if is_bull else "rgba(255,80,80,0.7)"
    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text=f"{'🟢' if is_bull else '🔴'} 색 구간 = {ann['name']} 형성 영역",
        showarrow=False,
        font=dict(size=11, color=legend_color),
        align="left",
        bgcolor="rgba(0,0,0,0.45)", bordercolor=legend_color, borderwidth=1,
    )

    pad = (dp["high"].max() - dp["low"].min()) * 0.08
    fig.update_layout(
        template="plotly_dark",
        height=320,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_rangeslider_visible=False,
        title=dict(text=f"  {ann['name']} — {bias_kr} 패턴 (15분)",
                   font=dict(size=13, color=mc), x=0),
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=9)),
        margin=dict(l=4, r=100, t=36, b=4),
        autosize=True,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                     range=[dp["low"].min() - pad, dp["high"].max() + pad])
    return fig


# ══════════════════════════════════════════════
# 단타 전용 — 지지저항 존 탐지
# ══════════════════════════════════════════════

def find_sr_zones(df, price, lookback=120, tolerance=0.005):
    """
    Returns list of (center_price, zone_type) tuples.
    zone_type is "지지" or "저항".
    Uses find_peaks on highs for resistance, on -lows for support.
    Clusters nearby levels within tolerance*2.
    Also adds round-number levels (±3 multiples).
    Returns top 8 sorted by distance from price.
    """
    h = df["high"].values[-lookback:]
    l = df["low"].values[-lookback:]

    # Peak-based levels
    peaks, _ = find_peaks(h, distance=5, prominence=h.std() * 0.25)
    troughs, _ = find_peaks(-l, distance=5, prominence=l.std() * 0.25)

    res_raw = h[peaks].tolist()
    sup_raw = l[troughs].tolist()

    # Round-number levels (±3 multiples)
    mag = 10 ** max(3, int(np.log10(price)) - 1)
    for m in range(-3, 4):
        lv = round(price / mag) * mag + m * mag
        if lv > price:
            res_raw.append(float(lv))
        else:
            sup_raw.append(float(lv))

    tol2 = tolerance * 2

    def cluster(lvs):
        if not lvs:
            return []
        lvs = sorted(lvs)  # keep duplicates for counting
        groups = [[lvs[0]]]
        for lv in lvs[1:]:
            if (lv - groups[-1][-1]) / max(groups[-1][-1], 1) < tol2:
                groups[-1].append(lv)
            else:
                groups.append([lv])
        return [(float(np.mean(g)), len(g)) for g in groups]

    r_lvs = [(lv, "저항", cnt) for lv, cnt in cluster(res_raw) if lv > price * 0.997]
    s_lvs = [(lv, "지지", cnt) for lv, cnt in cluster(sup_raw) if lv < price * 1.003]

    all_zones = r_lvs + s_lvs
    all_zones.sort(key=lambda x: abs(x[0] - price))
    return all_zones[:8]


def check_support_reaction(df_5m, support_lv, tolerance=0.004):
    """
    Returns (confirmed: bool, detail: str).
    Checks last 8 candles of df_5m for bullish reaction at support_lv.
    """
    last8 = df_5m.tail(8).copy()
    zone_low = support_lv * (1 - tolerance)
    zone_high = support_lv * (1 + tolerance)

    # Check if any candle touched the zone
    touched = last8[last8["low"] <= zone_high]
    if touched.empty:
        return False, "지지구간 미접촉"

    current_close = df_5m["close"].iloc[-1]

    # Check if support is broken
    if current_close < zone_low:
        return False, "지지 이탈"

    signals = []

    # Check for bullish reaction candles in touched candles
    for _, row in touched.iterrows():
        body = abs(row["close"] - row["open"])
        if body == 0:
            continue
        lower_wick = min(row["open"], row["close"]) - row["low"]
        bull = row["close"] >= row["open"]
        if bull and lower_wick > body * 0.5:
            signals.append("매수 반응 캔들")
            break

    # RSI check on last 30 candles
    rsi_series = ta.momentum.RSIIndicator(df_5m["close"].tail(30), window=14).rsi()
    if len(rsi_series.dropna()) >= 2:
        last_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
        if last_rsi > prev_rsi and last_rsi < 55:
            signals.append("RSI 상승 전환")

    # Price recovered above zone
    if current_close > zone_high:
        signals.append("지지구간 위 회복")

    # Volume spike check on touched candles
    vol_avg = df_5m["volume"].tail(20).mean()
    recent_vol_max = df_5m["volume"].tail(8).max()
    if recent_vol_max > vol_avg * 1.3:
        signals.append("거래량 스파이크")

    if signals:
        return True, " · ".join(signals[:3])
    else:
        return False, "접촉했으나 반응 미확인"


def check_resistance_reaction(df_5m, resistance_lv, tolerance=0.004):
    """
    Returns (confirmed: bool, detail: str).
    Checks last 8 candles of df_5m for bearish reaction at resistance_lv.
    """
    last8 = df_5m.tail(8).copy()
    zone_low = resistance_lv * (1 - tolerance)
    zone_high = resistance_lv * (1 + tolerance)

    # Check if any candle's high touched zone
    touched = last8[last8["high"] >= zone_low]
    if touched.empty:
        return False, "저항구간 미접촉"

    current_close = df_5m["close"].iloc[-1]

    # Check if resistance is broken (price above zone)
    if current_close > zone_high:
        return False, "저항 돌파"

    signals = []

    # Check for bearish reaction candles in touched candles
    for _, row in touched.iterrows():
        body = abs(row["close"] - row["open"])
        if body == 0:
            continue
        upper_wick = row["high"] - max(row["open"], row["close"])
        bear = row["close"] < row["open"]
        if bear and upper_wick > body * 0.5:
            signals.append("매도 반응 캔들")
            break

    # RSI check on last 30 candles
    rsi_series = ta.momentum.RSIIndicator(df_5m["close"].tail(30), window=14).rsi()
    if len(rsi_series.dropna()) >= 2:
        last_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
        if last_rsi < prev_rsi and last_rsi > 45:
            signals.append("RSI 하락 전환")

    # Price stayed below zone
    if current_close < zone_low:
        signals.append("저항구간 아래 유지")

    # Volume spike check on touched candles
    vol_avg = df_5m["volume"].tail(20).mean()
    recent_vol_max = df_5m["volume"].tail(8).max()
    if recent_vol_max > vol_avg * 1.3:
        signals.append("거래량 스파이크")

    if signals:
        return True, " · ".join(signals[:3])
    else:
        return False, "접촉했으나 반응 미확인"


@st.cache_data(ttl=240)
def get_btc_4h_direction():
    """Compute BTC 4H direction for use as ETH correlation filter."""
    df = fetch_ohlcv("BTC/USDT:USDT", "4h", limit=150)
    if df is None:
        return None
    ema_s = calc_ema(df, periods=(9, 21, 50, 200))[0]
    ms_s  = calc_market_structure(df)[0]
    ich_s = calc_ichimoku(df)[0]
    score = ema_s * 0.4 + ms_s * 0.35 + ich_s * 0.25
    if score > 0.20:    return "bull"
    elif score < -0.20: return "bear"
    else:               return "neutral"


def scalp_decision_engine(df_4h, df_1h, df_15m, df_5m, funding_rate, btc_direction=None):
    """
    Four-stage cascade decision engine for scalp trading.
    Returns a result dict with decision, labels, levels, and trade parameters.
    """
    price = df_15m["close"].iloc[-1]

    # ── Defaults for new fields
    pattern = "대기"
    pattern_label = "—"
    btc_conflict = False
    btc_conflict_label = ""
    zone_strength_map = {}
    projected_trade = None

    # ── Data freshness check
    last_ts = df_15m["ts"].iloc[-1]
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    data_age_min = (now_utc - last_ts).total_seconds() / 60
    data_stale = data_age_min > 18  # 15분봉 캔들 주기(15분) + 3분 버퍼

    # ── Market regime (ADX + ATR volatility)
    try:
        adx_series = ta.trend.ADXIndicator(df_15m["high"], df_15m["low"], df_15m["close"], window=14).adx()
        adx_val = adx_series.iloc[-1]
    except:
        adx_val = 25.0

    atr_series = calc_atr(df_15m, period=14)
    atr_now = atr_series.iloc[-1]
    atr_avg = atr_series.rolling(20).mean().iloc[-1]
    atr_ratio = atr_now / atr_avg if (atr_avg and not np.isnan(atr_avg) and atr_avg > 0) else 1.0

    if atr_ratio > 2.0:
        regime = "HIGH_VOLATILITY"
        regime_label = "⚠️ 고변동성"
    elif adx_val > 25:
        regime = "TRENDING"
        regime_label = "📈 추세장"
    elif adx_val < 18:
        regime = "RANGING"
        regime_label = "↔ 횡보장"
    else:
        regime = "NORMAL"
        regime_label = "— 보통"

    # ── Stage 1: 4H Direction
    ema_s = calc_ema(df_4h, periods=(9, 21, 50, 200))[0]
    ms_s = calc_market_structure(df_4h)[0]
    ich_s = calc_ichimoku(df_4h)[0]
    direction_score = ema_s * 0.4 + ms_s * 0.35 + ich_s * 0.25

    if direction_score > 0.20:
        direction = "bull"
        direction_label = "📈 4H 상승"
    elif direction_score < -0.20:
        direction = "bear"
        direction_label = "📉 4H 하락"
    else:
        direction = "neutral"
        direction_label = "↔ 4H 횡보"

    # ── Stage 2: 1H Structure
    ms_s_1h = calc_market_structure(df_1h)[0]
    ema_s_1h = calc_ema(df_1h, periods=(9, 21, 50, 100))[0]
    rsi_s_1h = calc_rsi(df_1h)[0]
    struct_score = ms_s_1h * 0.5 + ema_s_1h * 0.3 + rsi_s_1h * 0.2

    if direction == "bull":
        aligned = struct_score > 0
    elif direction == "bear":
        aligned = struct_score < 0
    else:
        aligned = False

    if aligned:
        structure_label = "✅ 1H 방향 일치"
    elif direction != "neutral":
        structure_label = "⚠️ 1H 방향 불일치"
    else:
        structure_label = "↔ 1H 중립"

    # ── Stage 3: 15m S/R Zone
    sr_zones = find_sr_zones(df_15m, price)

    at_zone = False
    zone_price = None
    zone_type = None
    zone_label = "🔍 주요 구간과 거리 있음"

    # Find nearest support (for bull/neutral) and resistance (for bear/neutral)
    nearest_support = None
    nearest_resistance = None

    for lv, zt, strength in sr_zones:
        zone_strength_map[lv] = strength
        dist = abs(lv - price) / price
        if zt == "지지" and dist <= 0.008:
            if nearest_support is None or dist < abs(nearest_support[0] - price) / price:
                nearest_support = (lv, zt)
        if zt == "저항" and dist <= 0.008:
            if nearest_resistance is None or dist < abs(nearest_resistance[0] - price) / price:
                nearest_resistance = (lv, zt)

    def _strength_dots(s):
        return "●" * min(s, 3) + "○" * (3 - min(s, 3))

    if direction == "bull":
        if nearest_support is not None:
            at_zone = True
            zone_price, zone_type = nearest_support
            zone_label = f"📍 지지구간 ${zone_price:,.0f} {_strength_dots(zone_strength_map.get(zone_price, 1))}"
        elif nearest_resistance is not None:
            at_zone = True
            zone_price, zone_type = nearest_resistance
            zone_label = f"📍 저항구간 ${zone_price:,.0f} {_strength_dots(zone_strength_map.get(zone_price, 1))}"
    elif direction == "bear":
        if nearest_resistance is not None:
            at_zone = True
            zone_price, zone_type = nearest_resistance
            zone_label = f"📍 저항구간 ${zone_price:,.0f} {_strength_dots(zone_strength_map.get(zone_price, 1))}"
        elif nearest_support is not None:
            at_zone = True
            zone_price, zone_type = nearest_support
            zone_label = f"📍 지지구간 ${zone_price:,.0f} {_strength_dots(zone_strength_map.get(zone_price, 1))}"
    else:  # neutral — pick whichever is closer
        candidates = []
        if nearest_support is not None:
            candidates.append(nearest_support)
        if nearest_resistance is not None:
            candidates.append(nearest_resistance)
        if candidates:
            best = min(candidates, key=lambda x: abs(x[0] - price))
            at_zone = True
            zone_price, zone_type = best
            if zone_type == "지지":
                zone_label = f"📍 지지구간 ${zone_price:,.0f} {_strength_dots(zone_strength_map.get(zone_price, 1))}"
            else:
                zone_label = f"📍 저항구간 ${zone_price:,.0f} {_strength_dots(zone_strength_map.get(zone_price, 1))}"

    # ── 차트 패턴 감지
    chart_patterns = detect_chart_patterns(df_4h, df_1h, df_15m, price)

    # ── 모멘텀 소진 감지
    momentum = detect_momentum_exhaustion(df_15m, df_5m)

    # ── S/R 접근 감지 (0.8%~1.5% 이내)
    approaching_zone = False
    approach_price = None
    approach_type = None
    for lv, zt, strength in sr_zones:
        dist = abs(lv - price) / price
        if 0.008 < dist <= 0.015:
            approaching_zone = True
            approach_price = lv
            approach_type = zt
            break

    # ── Pattern classification
    if zone_type == "지지":
        if direction == "bull":
            pattern = "눌림목"
            pattern_label = "📉→📈 눌림목"
        elif direction == "neutral":
            pattern = "지지반응"
            pattern_label = "📍 지지반응"
        else:
            pattern = "역추세반등"
            pattern_label = "⚠️ 역추세반등"
    elif zone_type == "저항":
        if direction == "bear":
            pattern = "저항반응"
            pattern_label = "📍 저항반응"
        elif direction == "neutral":
            pattern = "저항반응"
            pattern_label = "📍 저항반응"
        else:
            pattern = "역추세저항"
            pattern_label = "⚠️ 역추세저항"
    else:
        pattern = "대기"
        pattern_label = "—"

    # ── BTC correlation (ETH only)
    if btc_direction is not None:
        if direction == "bull" and btc_direction == "bear":
            btc_conflict = True
            btc_conflict_label = "⚠️ BTC 하락 중 — ETH 롱 신중"
        elif direction == "bear" and btc_direction == "bull":
            btc_conflict = True
            btc_conflict_label = "⚠️ BTC 상승 중 — ETH 숏 신중"
        elif direction == btc_direction and direction != "neutral":
            btc_conflict_label = "✅ BTC 방향 일치"

    # ── Stage 4: 5m Trigger
    trigger_ok = False
    trigger_detail = ""
    trigger_label = "-"

    if at_zone and df_5m is not None:
        if zone_type == "지지":
            trigger_ok, trigger_detail = check_support_reaction(df_5m, zone_price)
        else:
            trigger_ok, trigger_detail = check_resistance_reaction(df_5m, zone_price)

        if trigger_ok:
            trigger_label = f"✅ 반응 확인 — {trigger_detail}"
        else:
            trigger_label = "⏳ 반응 대기 중"

    steps = [direction_label, structure_label, zone_label, trigger_label]
    if btc_conflict_label:
        steps.append(btc_conflict_label)
    if momentum["exhausted"]:
        steps.append(momentum["label"])

    # ── Decision logic
    sl = tp1 = tp2 = rr1 = rr2 = invalidation = None
    entry_zone = None

    if direction == "neutral" and not at_zone:
        decision = "NO TRADE"
        reason = "4H 방향 불명확, 주요 구간 없음"
    elif direction != "neutral" and not aligned and not at_zone:
        decision = "NO TRADE"
        reason = "상위/하위 시간봉 방향 불일치"
    elif at_zone and not trigger_ok:
        decision = "WAIT"
        reason = "구간 도달 — 5분봉 반응 확인 대기"
        entry_zone = (zone_price * 0.997, zone_price * 1.003) if zone_price else None
        # Projected entry for WAIT
        if zone_price is not None:
            proj_entry = zone_price
            is_long_proj = zone_type == "지지"
            sr_proj = find_sr_levels(df_15m, proj_entry)
            sl_proj, _ = get_atr_stop(df_15m, proj_entry, is_long_proj, sr_proj, mult=0.8)
            sups_proj = sorted([(lv, t) for lv, t in sr_proj if lv < proj_entry], reverse=True)
            ress_proj = sorted([(lv, t) for lv, t in sr_proj if lv > proj_entry])
            risk_proj = proj_entry - sl_proj if is_long_proj else sl_proj - proj_entry
            tp1_proj = (ress_proj[0][0] if ress_proj else proj_entry + risk_proj * 2.0) if is_long_proj else (sups_proj[0][0] if sups_proj else proj_entry - risk_proj * 2.0)
            projected_trade = {
                "entry": proj_entry,
                "sl": sl_proj,
                "tp1": tp1_proj,
                "direction": "LONG" if is_long_proj else "SHORT",
            }
    elif at_zone and trigger_ok:
        # Determine trade direction from zone type
        if zone_type == "지지":
            trade_direction = "LONG"
        else:
            trade_direction = "SHORT"

        # High volatility override
        if regime == "HIGH_VOLATILITY":
            decision = "NO TRADE"
            reason = "고변동성 — 손절 폭 계산 불안정. 변동성 완화 후 진입"
        else:
            is_long = trade_direction == "LONG"
            sr_levels_15m = find_sr_levels(df_15m, price)
            sl, _ = get_atr_stop(df_15m, price, is_long, sr_levels_15m, mult=0.8)

            sups = sorted([(lv, t) for lv, t in sr_levels_15m if lv < price], reverse=True)
            ress = sorted([(lv, t) for lv, t in sr_levels_15m if lv > price])

            if is_long:
                risk = price - sl
                tp1 = ress[0][0] if ress else price + risk * 2.0
                tp2 = ress[1][0] if len(ress) > 1 else price + risk * 3.5
            else:
                risk = sl - price
                tp1 = sups[0][0] if sups else price - risk * 2.0
                tp2 = sups[1][0] if len(sups) > 1 else price - risk * 3.5

            # Fee + slippage (Bybit taker 0.055% × 2 + slippage 0.05%) = 0.16% round trip
            fee_rate = 0.0016
            net_gain_tp1 = abs(tp1 - price) - price * fee_rate
            net_loss = abs(price - sl) + price * fee_rate
            rr1 = net_gain_tp1 / net_loss if net_loss > 0 else 0
            net_gain_tp2 = abs(tp2 - price) - price * fee_rate
            rr2 = net_gain_tp2 / net_loss if net_loss > 0 else 0

            # Minimum profit filter: TP1 must be > 0.3% from entry (fee erosion prevention)
            tp1_pct = abs(tp1 - price) / price
            if tp1_pct < 0.003:
                decision = "NO TRADE"
                reason = f"목표가 근접 — TP1 +{tp1_pct*100:.2f}% (수수료 {0.16:.2f}% 미달)"
                sl = tp1 = tp2 = rr1 = rr2 = None
            elif rr1 < 1.5:
                decision = "NO TRADE"
                reason = f"손익비 부족 (R:R {rr1:.1f})"
                sl = tp1 = tp2 = rr1 = rr2 = None
            else:
                decision = trade_direction
                reason = f"{zone_type} 구간 반응 확인 — {trigger_detail}"
                invalidation = sl
                entry_zone = (zone_price * 0.997, zone_price * 1.003) if zone_price else None
    else:
        # direction is clear but no zone nearby
        decision = "WAIT"
        reason = "주요 구간 접근 대기"

    return dict(
        decision=decision,
        reason=reason,
        price=price,
        direction=direction,
        direction_label=direction_label,
        structure_label=structure_label,
        zone_label=zone_label,
        trigger_label=trigger_label,
        steps=steps,
        entry_zone=entry_zone,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        rr1=rr1,
        rr2=rr2,
        invalidation=invalidation,
        zone_price=zone_price,
        zone_type=zone_type,
        funding_rate=funding_rate,
        regime=regime,
        regime_label=regime_label,
        data_stale=data_stale,
        pattern=pattern,
        pattern_label=pattern_label,
        btc_conflict=btc_conflict,
        btc_conflict_label=btc_conflict_label,
        chart_patterns=chart_patterns,
        momentum=momentum,
        approaching_zone=approaching_zone,
        approach_price=approach_price,
        approach_type=approach_type,
        projected_trade=projected_trade if (at_zone and not trigger_ok and zone_price) else None,
    )


# ══════════════════════════════════════════════
# 지표 신호 그리드
# ══════════════════════════════════════════════

def signal_card_html(name, score, detail):
    if score > 0.25:    bg,bd,c,d = "#0d3b26","#2d8a57","#5de8a0","▲ 롱"
    elif score < -0.25: bg,bd,c,d = "#3b0d0d","#8a2d2d","#e85d5d","▼ 숏"
    else:               bg,bd,c,d = "#1a1a2e","#334466","#e8c85d","— 중립"
    st_w = "강" if abs(score) >= 0.7 else "중" if abs(score) >= 0.4 else "약"
    det = detail.replace('\n', '<br>')
    return (
        '<div class="signal-card" style="background:' + bg + ';border:1px solid ' + bd + '">'
        '<div style="font-size:9px;color:#888;font-weight:700;margin-bottom:3px">' + name + '</div>'
        '<div style="font-size:18px;font-weight:800;color:' + c + '">' + d + '</div>'
        '<div style="font-size:9px;color:#555">' + st_w + ' (' + f'{score:+.2f}' + ')</div>'
        '<div style="font-size:9px;color:#999;margin-top:5px;line-height:1.4">' + det + '</div>'
        '</div>'
    )

def render_signal_grid(cards):
    inner = "".join(signal_card_html(n, s, d) for n, s, d in cards)
    st.markdown('<div class="signal-grid">' + inner + '</div>', unsafe_allow_html=True)


def render_scalp_decision_card(result, symbol):
    """Render the scalp decision card with trade parameters."""
    decision = result["decision"]
    price = result["price"]
    now_str = datetime.now(KST).strftime("%H:%M")
    regime_label = result.get("regime_label", "")
    data_stale = result.get("data_stale", False)

    if data_stale:
        st.warning("⚠️ 데이터가 3분 이상 경과했어요 — 갱신 확인 필요")

    # Colors per decision
    if decision == "LONG":
        bg, bd, col, em = "#0d2e1e", "#2d8a57", "#5de8a0", "🟢"
    elif decision == "SHORT":
        bg, bd, col, em = "#2e0d0d", "#8a2d2d", "#e85d5d", "🔴"
    elif decision == "WAIT":
        bg, bd, col, em = "#1e1e0a", "#6a6a20", "#e8e05d", "⏳"
    else:  # NO TRADE
        bg, bd, col, em = "#1a1a1a", "#444", "#888", "⛔"

    decision_display = decision.replace("_", " ")

    # Build steps list HTML
    steps_html = "".join(
        f'<div style="font-size:12px;color:#bbb;padding:3px 0;border-left:3px solid {bd};padding-left:8px;margin-bottom:4px">{s}</div>'
        for s in result["steps"]
    )

    # 패턴 카드 (ASCII 모양 + 가격 레벨)
    # 패턴 요약 (차트에 직접 마킹 — 여기서는 간단 텍스트만)
    patterns = result.get("chart_patterns", [])
    if patterns:
        p_str = " · ".join(f'**{p["tf"]} {p["name"]}**' for p in patterns[:2])
        st.caption(f"📊 {p_str} 감지 — 차트에서 ◆ 마커 확인")

    html = (
        '<div style="background:' + bg + ';border:2px solid ' + bd + ';border-radius:14px;'
        'padding:14px 18px 12px;margin-bottom:8px">'
        '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px">'
        '<span style="font-size:1.6rem;font-weight:900;color:' + col + '">'
        + em + ' ' + decision_display +
        '</span>'
        '<span style="font-size:11px;color:#888;margin-left:4px">' + symbol + ' · ' + result.get("pattern_label", "—") + ' · 15분'
        + (f' · <span style="color:#e8c85d">{regime_label}</span>' if regime_label else '') +
        '</span>'
        '<span style="font-size:12px;font-weight:700;color:#e8c85d;margin-left:auto;white-space:nowrap">'
        + now_str + ' 기준'
        + '</span>'
        '</div>'
        + steps_html +
        '<div style="font-size:12px;color:#ccc;margin-top:8px;border-top:1px solid rgba(255,255,255,0.07);padding-top:8px">'
        + result["reason"] +
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    # Trade metrics if LONG or SHORT
    if decision in ("LONG", "SHORT") and result["sl"] is not None:
        sl = result["sl"]
        tp1 = result["tp1"]
        tp2 = result["tp2"]
        rr1 = result["rr1"]
        rr2 = result["rr2"]
        invalidation = result["invalidation"]
        zone_p = result.get("zone_price")
        zone_type_kr = result.get("zone_type", "구간")

        sl_c = (sl - price) / price * 100
        tp1_c = (tp1 - price) / price * 100
        tp2_c = (tp2 - price) / price * 100

        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("▶ 진입가", f"${price:,.2f}", "현재가")
        m2.metric("✂ 손절", f"${sl:,.2f}", f"{sl_c:+.2f}%")
        m3.metric("🎯 TP1", f"${tp1:,.2f}", f"{tp1_c:+.2f}% · R:R {rr1:.1f}")
        m4.metric("🎯 TP2", f"${tp2:,.2f}", f"{tp2_c:+.2f}% · R:R {rr2:.1f}")

        # 📋 시나리오
        zone_str = f"{zone_type_kr} ${zone_p:,.0f}" if zone_p else zone_type_kr
        dir_str = "상승" if decision == "LONG" else "하락"
        pat_str = " · ".join(f'{p["tf"]} {p["name"]}' for p in result.get("chart_patterns", [])[:2])
        pat_prefix = f"📊 {pat_str}  \n" if pat_str else ""
        scenario = (
            pat_prefix +
            f"**{zone_str} 반응 확인 → 지금이 진입 구간.**  \n"
            f"4H·1H 모두 {dir_str} 방향. "
            f"TP1 ${tp1:,.0f} 목표, ${sl:,.0f} 이탈 시 즉시 손절 (실질 R:R {rr1:.1f})."
        )
        st.info("📋 " + scenario)

    elif decision == "WAIT" and result["entry_zone"] is not None:
        proj = result.get("projected_trade")
        patterns = result.get("chart_patterns", [])
        pat_str = " · ".join(f'{p["tf"]} {p["name"]}' for p in patterns[:2]) if patterns else ""

        ez = result["entry_zone"]
        zone_p = result.get("zone_price")
        zone_type_kr = result.get("zone_type", "구간")
        direction = result.get("direction", "neutral")
        dir_str = {"bull": "상승", "bear": "하락", "neutral": "횡보"}.get(direction, "횡보")

        if proj:
            sl_pct = (proj["sl"] - zone_p) / zone_p * 100
            tp1_pct = (proj["tp1"] - zone_p) / zone_p * 100
            risk = abs(zone_p - proj["sl"])
            rr = abs(proj["tp1"] - zone_p) / risk if risk > 0 else 0
            pat_prefix2 = f"📊 {pat_str} 패턴 진행 중  \n" if pat_str else ""
            scenario = (
                pat_prefix2 +
                f"**{zone_type_kr} ${zone_p:,.0f} 반응 확인 시 {proj['direction']} 추천**  \n"
                f"예상 진입 `${proj['entry']:,.0f}` | SL `${proj['sl']:,.2f}` ({sl_pct:+.2f}%) | "
                f"TP1 `${proj['tp1']:,.0f}` ({tp1_pct:+.2f}%) · R:R {rr:.1f}  \n"
                f"지금은 반응 캔들 미확인 → 대기. ${ez[0]:,.0f}~${ez[1]:,.0f} 구간 주시 (4H {dir_str})"
            )
        else:
            pat_prefix2 = f"📊 {pat_str} 패턴 진행 중  \n" if pat_str else ""
            scenario = (
                pat_prefix2 +
                f"**{zone_type_kr} ${zone_p:,.0f} 근접, 반응 캔들 대기 중.**  \n"
                f"${ez[0]:,.0f}~${ez[1]:,.0f} 구간에서 아랫꼬리+거래량 확인 후 진입. (4H {dir_str})"
            )
        st.info("📋 " + scenario)

    else:
        reason = result.get("reason", "")
        direction = result.get("direction", "neutral")
        dir_str = {"bull": "상승", "bear": "하락", "neutral": "횡보"}.get(direction, "횡보")

        # 📋 시나리오
        if "고변동성" in reason:
            scenario = "**고변동성으로 손절 계산 불안정 → 관망.** 변동성이 줄어들 때까지 신규 진입 금지."
        elif "구간 접근" in reason or "주요 구간" in reason:
            scenario = f"**4H {dir_str} 추세 진행 중, 아직 진입 구간 미도달 → 대기.** 주요 지지/저항 구간이 가까워지면 알림."
        elif "방향 불일치" in reason:
            scenario = "**상위·하위 시간봉 방향이 엇갈림 → 관망.** 두 방향이 일치할 때까지 진입하지 않는 게 안전해요."
        elif "방향 불명확" in reason:
            scenario = "**4H 추세 방향이 불명확 → 관망.** 횡보 구간에서는 진입보다 방향 확인이 우선이에요."
        else:
            scenario = f"**{reason} → 관망.** 억지로 진입하기보다 다음 기회를 기다리세요."
        st.warning("📋 " + scenario)

        # Approaching zone info
        if result.get("approaching_zone") and not result.get("zone_price"):
            approach_p = result.get("approach_price")
            approach_t = result.get("approach_type")
            if approach_p and approach_t:
                dist_pct = abs(approach_p - result["price"]) / result["price"] * 100
                st.info(f"👀 **{approach_t} ${approach_p:,.0f} 접근 중** ({dist_pct:.1f}% 거리) — 구간 도달 시 반응 확인")


# ══════════════════════════════════════════════
# 단타 분석 (4H 방향 / 1H 구조 / 15분 존 / 5분 트리거)
# ══════════════════════════════════════════════

def render_analysis_scalp(symbol, display, funding_rate, btc_direction=None):
    with st.spinner(f"{display} 분석 중..."):
        df_4h  = fetch_ohlcv(symbol, "4h",  limit=300)
        df_1h  = fetch_ohlcv(symbol, "1h",  limit=200)
        df_15m = fetch_ohlcv(symbol, "15m", limit=300)
        df_5m  = fetch_ohlcv(symbol, "5m",  limit=200)

    if df_15m is None or df_1h is None or df_4h is None:
        st.error(f"{display} 데이터 로딩 실패")
        return

    price  = df_15m["close"].iloc[-1]
    change = (price - df_15m["close"].iloc[-2]) / df_15m["close"].iloc[-2] * 100

    result = scalp_decision_engine(df_4h, df_1h, df_15m, df_5m, funding_rate, btc_direction=btc_direction)

    # 패턴 어노테이션은 결정 카드 아래에도 쓰이므로 먼저 계산
    pat_annotations = get_pattern_chart_annotations(df_15m, price)

    render_scalp_decision_card(result, display)

    c1, c2 = st.columns(2)
    c1.metric(f"{display}", f"${price:,.2f}", f"{change:+.2f}%")
    c2.metric("펀딩비", f"{funding_rate*100:+.4f}%",
              "롱과열" if funding_rate*100 > 0.05 else ("숏과열" if funding_rate*100 < -0.05 else "중립"))

    # ── 패턴 상세 차트 (감지된 패턴이 있을 때만 표시)
    if pat_annotations:
        for ann in pat_annotations[:2]:
            fig_pat = build_pattern_detail_chart(df_15m, ann, display)
            st.plotly_chart(fig_pat, use_container_width=True,
                            config={"responsive": True, "displayModeBar": False})

    with st.expander("📈 15분 차트 & 지표 신호", expanded=False):
        ema_s, ema_d, ema_ser = calc_ema(df_15m, periods=(9, 21, 50))
        bb_s, bb_d, bb_u, bb_l, bb_m = calc_bb(df_15m)
        rsi_s, rsi_d, _, rsi_ser = calc_rsi(df_15m)
        macd_s, macd_d, ml, msl, mhl = calc_macd(df_15m)
        stoch_s, stoch_d, sk, sd = calc_stoch_rsi(df_15m)
        vwap_s, vwap_d, vwap_ser = calc_vwap(df_15m)
        vol_s, vol_d = calc_volume(df_15m)
        fun_s, fun_d = calc_funding(funding_rate)
        sr_levels = find_sr_levels(df_15m, price)

        # 지표 신호 그리드 (차트 위에)
        render_signal_grid([
            ("RSI",       rsi_s,   rsi_d),
            ("MACD",      macd_s,  macd_d),
            ("Stoch RSI", stoch_s, stoch_d),
            ("VWAP",      vwap_s,  vwap_d),
            ("EMA 정렬",  ema_s,   ema_d),
            ("볼린저밴드", bb_s,   bb_d),
            ("거래량",    vol_s,   vol_d),
            ("펀딩비",    fun_s,   fun_d),
        ])

        chart_rec = None
        if result["decision"] in ("LONG", "SHORT") and result["sl"] is not None:
            chart_rec = dict(
                direction=result["decision"], entry=price,
                sl=result["sl"], tp1=result["tp1"], tp2=result["tp2"],
                long_prob=0, counter=False, rec_lev=0,
                risk_pct=0, rr1=result["rr1"], rr2=result["rr2"],
                reasons=[], atr=0,
            )

        fig = build_chart(df_15m, ema_ser, bb_u, bb_l, bb_m,
                          None, None, None, None,
                          rsi_ser, ml, msl, mhl, sk, sd,
                          sr_levels, [], [], chart_rec, display,
                          vwap=vwap_ser, tf_label="15분",
                          pattern_annotations=pat_annotations)
        st.plotly_chart(fig, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})


# ══════════════════════════════════════════════
# 시장 개요
# ══════════════════════════════════════════════

def render_market_overview():
    st.markdown("### 📊 시장 개요")

    # ── 현재 세션
    now_kst = datetime.now(KST)
    h = now_kst.hour
    sessions = []
    if 9 <= h < 18:   sessions.append("🇯🇵 아시아")
    if 17 <= h or h < 2: sessions.append("🇬🇧 런던")
    if 22 <= h or h < 7: sessions.append("🇺🇸 뉴욕")
    session_str = " · ".join(sessions) if sessions else "🌙 비활성 시간대"
    st.caption(f"현재 세션: {session_str}  ·  {now_kst.strftime('%H:%M KST')}")

    st.divider()

    # ── 공포/탐욕 + 도미넌스
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**😱 공포/탐욕 지수**")
        fg_val, fg_label = fetch_fear_greed()
        if fg_val is not None:
            if fg_val <= 25:   fc = "#5de8a0"; interp = "극단 공포 — 과매도 구간. 반등 가능성 주시"
            elif fg_val <= 45: fc = "#a0e85d"; interp = "공포 — 시장 심리 위축. 롱 신중"
            elif fg_val <= 55: fc = "#e8c85d"; interp = "중립 — 방향성 없음"
            elif fg_val <= 75: fc = "#e8a05d"; interp = "탐욕 — 과열 주의. 숏 신중"
            else:              fc = "#e85d5d"; interp = "극단 탐욕 — 조정 가능성. 고점 진입 주의"
            st.markdown(
                f'<div style="background:#1a1a2e;border-radius:10px;padding:12px 16px;text-align:center">'
                f'<div style="font-size:2rem;font-weight:900;color:{fc}">{fg_val}</div>'
                f'<div style="font-size:11px;color:{fc};font-weight:600">{fg_label}</div>'
                f'<div style="background:#333;border-radius:4px;height:6px;margin:8px 0 6px">'
                f'<div style="background:{fc};width:{fg_val}%;height:6px;border-radius:4px"></div></div>'
                f'<div style="font-size:10px;color:#999">{interp}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.caption("로딩 실패")

    with col2:
        st.markdown("**🏛 BTC 도미넌스**")
        btc_dom, eth_dom = fetch_dominance()
        if btc_dom is not None:
            if btc_dom >= 60:   dom_lbl, dom_col = "BTC 독주 — ETH 포함 알트 전반 약세", "#e85d5d"
            elif btc_dom >= 50: dom_lbl, dom_col = "BTC 우세 — 알트 횡보 가능성", "#e8a05d"
            elif btc_dom >= 40: dom_lbl, dom_col = "알트 시즌 가능성 — ETH 상대 강세", "#e8c85d"
            else:               dom_lbl, dom_col = "알트 시즌 — ETH 유리", "#5de8a0"
            for nm, val, color in [("BTC", btc_dom, "#F7931A"), ("ETH", eth_dom, "#627EEA")]:
                st.markdown(
                    f'<div style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">'
                    f'<span style="color:#aaa">{nm}</span>'
                    f'<span style="font-weight:700;color:{color}">{val}%</span></div>'
                    f'<div style="background:#333;border-radius:4px;height:5px">'
                    f'<div style="background:{color};width:{val}%;height:5px;border-radius:4px"></div>'
                    f'</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:10px;color:{dom_col}">{dom_lbl}</div>', unsafe_allow_html=True)
        else:
            st.caption("로딩 실패")

    st.divider()

    # ── OI
    st.markdown("**📊 미결제약정 (OI)**")
    col_oi1, col_oi2 = st.columns(2)
    for col_oi, sym_name, sym_usdt in [(col_oi1, "BTC", "BTCUSDT"), (col_oi2, "ETH", "ETHUSDT")]:
        oi_val, oi_chg = fetch_open_interest(sym_usdt)
        with col_oi:
            if oi_val is not None:
                chg_col = "#5de8a0" if oi_chg >= 0 else "#e85d5d"
                chg_arrow = "▲" if oi_chg >= 0 else "▼"
                if oi_val > 1e9:
                    oi_str = f"${oi_val/1e9:.2f}B"
                else:
                    oi_str = f"${oi_val/1e6:.0f}M"
                st.markdown(
                    f'<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px">'
                    f'<div style="font-size:10px;color:#888;margin-bottom:4px">{sym_name} OI</div>'
                    f'<div style="font-size:1.2rem;font-weight:700;color:#ccc">{oi_str}</div>'
                    f'<div style="font-size:11px;color:{chg_col}">{chg_arrow} {abs(oi_chg):.2f}% (1H)</div>'
                    f'<div style="font-size:9px;color:#666;margin-top:4px">'
                    f'OI 증가+가격 상승 = 신규 롱 유입 가능. OI 감소+가격 상승 = 숏 청산 가능</div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.caption(f"{sym_name} OI 로딩 실패")

    st.divider()

    # ── L/S 비율
    st.markdown("**⚖️ 롱/숏 비율 (Bybit)**")
    col_ls1, col_ls2 = st.columns(2)
    for col_ls, sym_name, sym_usdt in [(col_ls1, "BTC", "BTCUSDT"), (col_ls2, "ETH", "ETHUSDT")]:
        lp, sp = fetch_ls_ratio(sym_usdt)
        with col_ls:
            if lp is not None:
                lc = "#5de8a0" if lp >= 50 else "#e85d5d"
                sc2 = "#e85d5d" if lp >= 50 else "#5de8a0"
                st.markdown(
                    f'<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px">'
                    f'<div style="font-size:10px;color:#888;margin-bottom:6px">{sym_name}</div>'
                    f'<div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">'
                    f'<span style="color:{lc};font-weight:700;font-size:14px">{lp:.0f}%</span>'
                    f'<div style="flex:1;background:#333;border-radius:4px;height:6px">'
                    f'<div style="background:{lc};width:{lp}%;height:6px;border-radius:4px 0 0 4px"></div></div>'
                    f'<span style="color:{sc2};font-weight:700;font-size:14px">{sp:.0f}%</span></div>'
                    f'<div style="display:flex;justify-content:space-between;font-size:9px;color:#555">'
                    f'<span>롱</span><span>숏</span></div>'
                    f'</div>', unsafe_allow_html=True)

    st.divider()

    # ── 매크로 캘린더
    st.markdown("**📅 이번 주 USD 주요 이벤트**")
    st.caption("고영향 이벤트 발표 전후 변동성 급증 주의 — 발표 30분 전 신규 진입 자제 권장")
    events = fetch_macro_events()
    if events:
        for ev in events[:8]:
            dot = "🔴" if ev["impact"] == "High" else "🟡"
            st.markdown(
                f'<div style="padding:6px 0;border-bottom:1px solid #1a1a2e">'
                f'<div style="font-size:10px;color:#aaa">{dot} {ev["date"]} {ev["time"]}</div>'
                f'<div style="font-size:11px;font-weight:600">{ev["title"]}</div>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.caption("캘린더 로딩 실패")


# ══════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ 단타 분석기")
        st.caption("Bybit USDT Perp · 60초 자동 갱신")
        st.divider()

        fg_val, fg_label = fetch_fear_greed()
        if fg_val is not None:
            if fg_val <= 25:   fc = "#5de8a0"
            elif fg_val <= 45: fc = "#a0e85d"
            elif fg_val <= 55: fc = "#e8c85d"
            elif fg_val <= 75: fc = "#e8a05d"
            else:              fc = "#e85d5d"
            st.markdown(
                f'<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;text-align:center">'
                f'<div style="font-size:10px;color:#888;margin-bottom:3px">공포/탐욕</div>'
                f'<div style="font-size:1.6rem;font-weight:900;color:{fc}">{fg_val}</div>'
                f'<div style="font-size:10px;color:{fc}">{fg_label}</div>'
                f'</div>', unsafe_allow_html=True)

        st.divider()
        btc_dom, eth_dom = fetch_dominance()
        if btc_dom is not None:
            st.markdown(f'<div style="font-size:11px;color:#aaa">BTC 도미넌스 <span style="color:#F7931A;font-weight:700">{btc_dom}%</span></div>', unsafe_allow_html=True)

        st.divider()
        events = fetch_macro_events()
        high_events = [e for e in events if e["impact"] == "High"]
        if high_events:
            st.warning(f"⚠️ 고영향 이벤트 {len(high_events)}건")
            for ev in high_events[:3]:
                st.caption(f"🔴 {ev['date']} {ev['time']} {ev['title']}")


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
render_sidebar()

st.markdown("## ⚡ 선물 단타 분석기")
st.caption(f"OKX Perpetual · 4H→1H→15m→5m 분석 · 60초 자동갱신 · {datetime.now(KST).strftime('%H:%M KST')}")

col_refresh, _ = st.columns([1, 6])
with col_refresh:
    if st.button("🔄 즉시 새로고침", use_container_width=True):
        st.cache_data.clear(); st.rerun()

st.divider()

btc_fr = fetch_funding("BTC/USDT:USDT")
eth_fr = fetch_funding("ETH/USDT:USDT")

tab_trade, tab_market = st.tabs(["⚡ 단타 분석", "📊 시장 개요"])

with tab_trade:
    # Macro warning
    events = fetch_macro_events()
    high_events = [e for e in events if e["impact"] == "High"]
    if high_events:
        titles = " · ".join(e["title"] for e in high_events[:2])
        st.warning(f"⚠️ 고영향 이벤트 예정 — {titles} · 발표 전 신규 진입 주의")

    btc_4h_dir = get_btc_4h_direction()

    col_btc, col_eth = st.columns(2)
    with col_btc:
        render_analysis_scalp("BTC/USDT:USDT", "BTC", btc_fr)
    with col_eth:
        render_analysis_scalp("ETH/USDT:USDT", "ETH", eth_fr, btc_direction=btc_4h_dir)

with tab_market:
    render_market_overview()

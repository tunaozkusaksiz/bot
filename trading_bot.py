#!/usr/bin/env python3
"""
Asama 1 - Kripto KAGIT (paper) ticaret botu

- Canli fiyati Binance'in halka acik, cografi engeli OLMAYAN veri ucundan ceker
  (data-api.binance.vision) -> GitHub Actions'ta sorunsuz calisir.
- EMA(20/50) + RSI + ATR sinyalleri; trailing ATR stop; risk yonetimi;
  dusus (drawdown) emniyet salteri; islem arasi bekleme (cooldown).
- Borsaya BAGLANMAZ: alim-satim, kod icinde tutulan SAHTE bir portfoye islenir
  (state.json'da nakit + pozisyonlar). Gercek para ya da borsa anahtari gerekmez.
- Opsiyonel: Gemini API ile sirket haberi + kuresel risk katmani.
- Telegram'a ozet rapor atar.

UYARI: Bu bir demo/egitim aracidir, yatirim tavsiyesi degildir.
"""

import os
import json
import time
import requests
import pandas as pd

# =============================== AYARLAR ===============================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOGEUSDT"]

COIN_NAMES = {
    "BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "SOLUSDT": "Solana",
    "BNBUSDT": "BNB Binance", "XRPUSDT": "XRP Ripple", "ADAUSDT": "Cardano",
    "AVAXUSDT": "Avalanche", "LINKUSDT": "Chainlink", "LTCUSDT": "Litecoin",
    "DOGEUSDT": "Dogecoin",
}

# BIST hisseleri (Yahoo Finance sembolleri, ".IS" ekli). Sinyal-only: Midas'tan elle uygulanir.
BIST_SYMBOLS  = ["GARAN.IS", "AKBNK.IS", "ISCTR.IS", "THYAO.IS", "ASELS.IS",
                 "KCHOL.IS", "TUPRS.IS", "SISE.IS", "EREGL.IS", "BIMAS.IS"]
BIST_INTERVAL = "1h"     # BIST icin saatlik mum (ucretsiz veride en dengeli)
BIST_RANGE    = "1mo"    # son 1 ay

INTERVAL        = "15m"     # mum periyodu (15 dk). Sakin mod icin "1h"
KLINES_LIMIT    = 200
_IVMIN          = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}.get(INTERVAL, 60)
LOOKBACK_24H    = max(2, round(24 * 60 / _IVMIN))   # ~24 saatlik geri bakis (mum sayisi)
EMA_FAST        = 20
EMA_SLOW        = 50
RSI_PERIOD      = 14
RSI_FLOOR       = 45
RSI_CEIL        = 70
ATR_PERIOD      = 14
ATR_STOP_MULT   = 3.0
RISK_PER_TRADE  = 0.02
MAX_POSITIONS   = 5
COOLDOWN_HOURS  = 6
MAX_DRAWDOWN_HALT = 0.20
STARTING_CASH   = 10000.0   # sahte baslangic sermayesi (USDT)
FEE_PCT         = 0.10      # her islemde simule edilen komisyon (%)
MIN_TRADE_USDT  = 10.0

STATE_FILE = "state.json"
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() != "false"

GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")    # opsiyonel (haber/risk katmani)
GEMINI_MODEL = "gemini-2.5-flash"                 # istersen "gemini-3.5-flash" yapabilirsin
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT      = os.getenv("TELEGRAM_CHAT_ID", "")

# Halka acik, cografi engellenmeyen market-data ucu
PUBLIC_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


# =============================== VERI / GOSTERGE ===============================
def fetch_klines(symbol, interval=INTERVAL, limit=KLINES_LIMIT):
    r = requests.get(PUBLIC_KLINES_URL,
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["openTime", "open", "high", "low", "close", "volume",
                                         "closeTime", "qav", "trades", "tbbav", "tbqav", "ignore"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def compute_indicators(df):
    close, high, low = df["close"], df["high"], df["low"]
    ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).fillna(50.0)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    return {
        "price": float(close.iloc[-1]),
        "ema_fast": float(ema_fast.iloc[-1]),
        "ema_slow": float(ema_slow.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "atr": float(atr.iloc[-1]),
    }


# =============================== BIST (Yahoo, sinyal-only) ===============================
def fetch_yahoo(symbol, interval=BIST_INTERVAL, rng=BIST_RANGE):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                     params={"interval": interval, "range": rng},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"],
                       "low": q["low"], "close": q["close"]}).dropna().reset_index(drop=True)
    return df


def bist_signals():
    """Her BIST hissesi icin guncel sinyal (AL / SAT-cik / bekle). Emir gondermez."""
    out = []
    for sym in BIST_SYMBOLS:
        name = sym.replace(".IS", "")
        try:
            df = fetch_yahoo(sym)
            if len(df) < EMA_SLOW + 2:
                out.append(f"{name}  ·  yeterli veri yok"); continue
            ind = compute_indicators(df)
            if ind["ema_fast"] > ind["ema_slow"] and RSI_FLOOR <= ind["rsi"] <= RSI_CEIL:
                out.append(f"🟢 {name}: AL sinyali (trend yukarı, RSI {ind['rsi']:.0f})")
            elif ind["ema_fast"] < ind["ema_slow"]:
                out.append(f"🔴 {name}: SAT/çık uyarısı (trend aşağı)")
            else:
                out.append(f"⚪ {name}: bekle (RSI {ind['rsi']:.0f})")
        except Exception as e:
            print(f"{sym} BIST veri hatası: {e}")
            out.append(f"{name}  ·  veri alınamadı")
    return out


# =============================== KARAR MANTIGI ===============================
def want_to_buy(ind, risk_state):
    if risk_state == "HIGH":
        return False, "küresel risk YÜKSEK"
    if ind["ema_fast"] <= ind["ema_slow"]:
        return False, "trend aşağı (EMA20<EMA50)"
    if ind["rsi"] < RSI_FLOOR:
        return False, f"RSI zayıf ({ind['rsi']:.0f})"
    if ind["rsi"] > RSI_CEIL:
        return False, f"RSI aşırı alım ({ind['rsi']:.0f})"
    return True, "trend yukarı + RSI uygun"


def want_to_sell(ind, high_since_entry):
    if ind["ema_fast"] < ind["ema_slow"]:
        return True, "trend döndü (EMA20<EMA50)"
    stop = high_since_entry - ATR_STOP_MULT * ind["atr"]
    if ind["price"] <= stop:
        return True, f"trailing stop ({stop:.4f})"
    return False, "tut"


# =============================== HABER / RISK (Gemini) ===============================
def gemini_json(system, user):
    if not GEMINI_KEY:
        return None
    try:
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{GEMINI_MODEL}:generateContent")
        r = requests.post(
            url,
            headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"parts": [{"text": user}]}],
                  "generationConfig": {"temperature": 0, "maxOutputTokens": 300,
                                       "responseMimeType": "application/json"}},
            timeout=30)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        return json.loads(text)
    except Exception as e:
        print("Gemini hatasi:", e)
        return None


def gdelt_headlines(query, maxrecords=15, timespan="1d"):
    try:
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc",
                         params={"query": query, "mode": "ArtList", "format": "json",
                                 "maxrecords": maxrecords, "timespan": timespan,
                                 "sort": "DateDesc"},
                         timeout=20)
        r.raise_for_status()
        return [a.get("title", "") for a in r.json().get("articles", []) if a.get("title")]
    except Exception as e:
        print("GDELT hatasi:", e)
        return []


def fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=15)
        r.raise_for_status()
        return int(r.json()["data"][0]["value"])
    except Exception:
        return None


def global_risk_state(btc_change_24h):
    market = "NORMAL"
    if btc_change_24h is not None:
        if btc_change_24h <= -10:
            market = "HIGH"
        elif btc_change_24h <= -5:
            market = "ELEVATED"
    fg = fear_greed()
    if fg is not None and fg <= 15 and market == "NORMAL":
        market = "ELEVATED"

    ai, reason = "NORMAL", ""
    if GEMINI_KEY:
        heads = gdelt_headlines('(war OR invasion OR conflict OR sanctions OR '
                                '"central bank" OR "rate hike" OR crisis OR attack)')
        if heads:
            sys = ("You are a market risk classifier. From recent global news headlines, "
                   "judge CURRENT market risk for risk assets. Respond ONLY with JSON: "
                   '{"risk":"NORMAL|ELEVATED|HIGH","reason":"short"}. '
                   "HIGH = a major geopolitical/financial shock unfolding right now.")
            res = gemini_json(sys, "Headlines:\n- " + "\n- ".join(heads[:15]))
            if res and res.get("risk") in ("NORMAL", "ELEVATED", "HIGH"):
                ai, reason = res["risk"], res.get("reason", "")

    order = {"NORMAL": 0, "ELEVATED": 1, "HIGH": 2}
    final = market if order[market] >= order[ai] else ai
    detail = f"piyasa={market}, ai={ai}, F&G={fg}" + (f" ({reason})" if reason else "")
    return final, detail


def negative_news(coin_name):
    if not GEMINI_KEY:
        return False, ""
    heads = gdelt_headlines(f"{coin_name} crypto", maxrecords=10, timespan="1d")
    if not heads:
        return False, ""
    sys = ("You classify crypto news. Given headlines about a coin, decide if there is MAJOR "
           "NEGATIVE news (hack, ban, lawsuit, exploit, delisting, fraud) that warrants avoiding "
           'a new buy now. Respond ONLY with JSON: {"negative":true|false,"reason":"short"}.')
    res = gemini_json(sys, f"Coin: {coin_name}\nHeadlines:\n- " + "\n- ".join(heads[:10]))
    if res and isinstance(res.get("negative"), bool):
        return res["negative"], res.get("reason", "")
    return False, ""


# =============================== TELEGRAM / STATE ===============================
def notify(text):
    if not (TG_TOKEN and TG_CHAT):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      json={"chat_id": TG_CHAT, "text": text[:4000]}, timeout=20)
    except Exception as e:
        print("Telegram hatasi:", e)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# =============================== ANA AKIS (KAGIT PORTFOY) ===============================
def main():
    mode = "DENEME (sadece rapor, portföy değişmez)" if DRY_RUN else "CANLI kağıt-portföy"
    print(f"=== Bot çalışıyor — {mode} ===")

    state = load_state()
    meta = state.setdefault("_meta", {"cash": STARTING_CASH, "peak_equity": STARTING_CASH})

    data, btc_change = {}, None
    for sym in SYMBOLS:
        try:
            df = fetch_klines(sym)
            data[sym] = compute_indicators(df)
            if sym == "BTCUSDT" and len(df) > LOOKBACK_24H:
                btc_change = (df["close"].iloc[-1] / df["close"].iloc[-1 - LOOKBACK_24H] - 1) * 100
        except Exception as e:
            print(f"{sym} veri hatası: {e}")

    if not data:
        notify("⚠️ Veri çekilemedi (fiyat verisine erişilemiyor)."); print("Veri yok."); return

    risk, risk_detail = global_risk_state(btc_change)
    fee = FEE_PCT / 100.0
    cash = meta["cash"]            # çalışma başı nakit (özet satırı için)
    avail = cash                   # işlem bütçesi (döngü içinde değişir)

    positions_val = sum((state.get(s) or {}).get("qty", 0.0) * data[s]["price"]
                        for s in data if (state.get(s) or {}).get("qty", 0.0) > 0)
    equity = cash + positions_val
    if meta.get("peak_equity") is None or equity > meta["peak_equity"]:
        meta["peak_equity"] = equity
    dd = (equity / meta["peak_equity"] - 1) if meta["peak_equity"] else 0.0
    halt_new = dd <= -MAX_DRAWDOWN_HALT
    open_positions = sum(1 for s in SYMBOLS if (state.get(s) or {}).get("qty", 0.0) > 0)

    acts, holds, waits = [], [], []      # işlemler / elde tutulanlar / bekleyenler

    for sym in SYMBOLS:
        if sym not in data:
            continue
        ind, base = data[sym], sym.replace("USDT", "")
        st = state.setdefault(sym, {"in_position": False, "entry": None,
                                    "high": None, "qty": 0.0, "last_trade": 0})
        in_pos = st.get("qty", 0.0) > 0

        if in_pos:
            st["high"] = max(st.get("high") or ind["price"], ind["price"])
            pnl = (ind["price"] / st["entry"] - 1) * 100 if st["entry"] else 0.0
            sell, why = want_to_sell(ind, st["high"])
            if sell:
                if DRY_RUN:
                    acts.append(f"🔴 {base}: SATARDIM — {why}  (K/Z %{pnl:+.1f})")
                else:
                    avail += st["qty"] * ind["price"] * (1 - fee)
                    acts.append(f"🔴 {base}: SATILDI — {why}  (K/Z %{pnl:+.1f})")
                    st.update({"in_position": False, "entry": None, "high": None,
                               "qty": 0.0, "last_trade": time.time()})
                    open_positions -= 1
            else:
                holds.append(f"{base}:  K/Z %{pnl:+.1f}  ·  RSI {ind['rsi']:.0f}")
            continue

        buy, why = want_to_buy(ind, risk)
        if not buy:
            waits.append(f"{base}  ·  {why}"); continue
        if halt_new:
            waits.append(f"{base}  ·  sinyal var ama düşüş limiti aktif"); continue
        if open_positions >= MAX_POSITIONS:
            waits.append(f"{base}  ·  sinyal var ama pozisyon limiti dolu"); continue
        if time.time() - st.get("last_trade", 0) < COOLDOWN_HOURS * 3600:
            waits.append(f"{base}  ·  sinyal var ama bekleme süresinde"); continue
        neg, nreason = negative_news(COIN_NAMES.get(sym, base))
        if neg:
            waits.append(f"{base}  ·  alım iptal: olumsuz haber ({nreason})"); continue

        stop_dist = ATR_STOP_MULT * ind["atr"]
        if stop_dist <= 0:
            waits.append(f"{base}  ·  ATR sıfır, atlandı"); continue
        qty = (equity * RISK_PER_TRADE) / stop_dist
        cost = qty * ind["price"] * (1 + fee)
        if cost > avail * 0.95:
            qty = (avail * 0.95) / (ind["price"] * (1 + fee))
            cost = qty * ind["price"] * (1 + fee)
        if qty * ind["price"] < MIN_TRADE_USDT:
            waits.append(f"{base}  ·  sinyal var ama nakit yetmiyor"); continue

        if DRY_RUN:
            acts.append(f"🟢 {base}: ALIRDIM — {qty:.6f} @ {ind['price']:.4f}  (~{cost:.0f} USDT)")
            avail -= cost           # bütçeyi ve 5-pozisyon limitini gözet (state'e yazılmaz)
            open_positions += 1
        else:
            avail -= cost
            st.update({"in_position": True, "entry": ind["price"], "high": ind["price"],
                       "qty": qty, "last_trade": time.time()})
            open_positions += 1
            acts.append(f"🟢 {base}: ALINDI — {qty:.6f} @ {ind['price']:.4f}  (~{cost:.0f} USDT)")

    if not DRY_RUN:
        meta["cash"] = avail
        save_state(state)

    news_status = "açık (Gemini)" if GEMINI_KEY else "KAPALI (Gemini anahtarı eklenmemiş)"
    rflag = {"NORMAL": "🟢", "ELEVATED": "🟠", "HIGH": "🔴"}.get(risk, "⚪")

    out = [f"🤖 KRİPTO BOT — {mode}  ·  {INTERVAL} mum",
           f"{rflag} Küresel risk: {risk}   ({risk_detail})",
           f"🧠 Haber/analiz katmanı: {news_status}",
           f"💰 Portföy ≈ {equity:,.0f} USDT   (nakit {cash:,.0f} + pozisyon {positions_val:,.0f})",
           f"📈 Tepe {meta['peak_equity']:,.0f}  ·  düşüş %{dd*100:.1f}"
           + ("   ⛔ yeni alım durdu" if halt_new else "")]

    if acts:
        out.append("\n🔁 İşlemler\n" + "\n".join(acts))
    if holds:
        out.append("\n📦 Elde tutulanlar\n" + "\n".join(holds))
    if waits:
        out.append(f"\n⏳ Şu an alım yok ({len(waits)} coin)\n" + "\n".join(waits))

    bist = bist_signals()
    if bist:
        out.append("\n📈 BİST sinyalleri (Midas'tan elle uygula)\n" + "\n".join(bist))

    report = "\n".join(out)
    notify(report); print(report)


if __name__ == "__main__":
    main()

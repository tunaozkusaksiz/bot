#!/usr/bin/env python3
"""
Asama 1 - Kripto otomatik islem botu (Binance TESTNET / sahte para)

- Canli (public) fiyat verisiyle EMA(20/50) + RSI + ATR sinyalleri uretir
- Emirleri SADECE Binance testnet'e gonderir (gercek para yok)
- Trailing ATR stop, pozisyon boyutlandirma, dusus (drawdown) emniyet salteri,
  islem arasi bekleme (cooldown)
- Opsiyonel: Claude API ile sirket haberi + kuresel risk katmani
- Telegram'a ozet rapor atar
- Durumu state.json'da tutar (GitHub Actions her calismada geri commit eder)

UYARI: Bu bir demo/egitim aracidir, yatirim tavsiyesi degildir.
"""

import os
import json
import time
import math
import requests
import pandas as pd
import numpy as np
from binance.client import Client

# =============================== AYARLAR ===============================
# Takip edilen kripto listesi (USDT pariteleri). Istedigini ekle/cikar.
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOGEUSDT"]

COIN_NAMES = {  # haber aramasi icin okunabilir isimler
    "BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "SOLUSDT": "Solana",
    "BNBUSDT": "BNB Binance", "XRPUSDT": "XRP Ripple", "ADAUSDT": "Cardano",
    "AVAXUSDT": "Avalanche", "LINKUSDT": "Chainlink", "LTCUSDT": "Litecoin",
    "DOGEUSDT": "Dogecoin",
}

INTERVAL        = "1h"    # sinyal mum periyodu
KLINES_LIMIT    = 200     # gosterge gecmisi icin cekilecek mum sayisi
EMA_FAST        = 20
EMA_SLOW        = 50
RSI_PERIOD      = 14
RSI_FLOOR       = 45      # altinda alim yok (zayif momentum)
RSI_CEIL        = 70      # ustunde alim yok (asiri alim)
ATR_PERIOD      = 14
ATR_STOP_MULT   = 3.0     # trailing stop = en yuksek - 3*ATR
RISK_PER_TRADE  = 0.02    # islem basina sermayenin %2'si riske edilir
MAX_POSITIONS   = 5       # ayni anda en fazla pozisyon
COOLDOWN_HOURS  = 6       # ayni coin'de iki islem arasi minimum sure
MAX_DRAWDOWN_HALT = 0.20  # equity tepe noktasindan %20 dusunce yeni alim durur
MIN_NOTIONAL_FALLBACK = 10.0  # USDT

STATE_FILE = "state.json"
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() != "false"

# --- Sirlar (GitHub Secrets'tan gelir) ---
BINANCE_KEY     = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET  = os.getenv("BINANCE_API_SECRET", "")
TG_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT         = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")    # opsiyonel (haber/risk katmani)
GEMINI_MODEL = "gemini-2.5-flash"                 # istersen "gemini-3.5-flash" yapabilirsin

# Public market data: ABD IP'lerinde 451 vermeyen ayna uc nokta
PUBLIC_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


# =============================== VERI / GOSTERGE ===============================
def fetch_klines(symbol, interval=INTERVAL, limit=KLINES_LIMIT):
    r = requests.get(PUBLIC_KLINES_URL,
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=20)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=["openTime", "open", "high", "low", "close", "volume",
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
    rs = avg_gain / avg_loss          # avg_loss=0 & gain>0 -> inf -> RSI 100
    rsi = (100 - (100 / (1 + rs))).fillna(50.0)  # 0/0 (duz piyasa) -> 50

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


# =============================== KARAR MANTIGI ===============================
def want_to_buy(ind, risk_state):
    if risk_state == "HIGH":
        return False, "kuresel risk YUKSEK"
    if ind["ema_fast"] <= ind["ema_slow"]:
        return False, "trend asagi (EMA20<EMA50)"
    if ind["rsi"] < RSI_FLOOR:
        return False, f"RSI zayif ({ind['rsi']:.0f})"
    if ind["rsi"] > RSI_CEIL:
        return False, f"RSI asiri alim ({ind['rsi']:.0f})"
    return True, "trend yukari + RSI uygun"


def want_to_sell(ind, high_since_entry):
    if ind["ema_fast"] < ind["ema_slow"]:
        return True, "trend dondu (EMA20<EMA50)"
    stop = high_since_entry - ATR_STOP_MULT * ind["atr"]
    if ind["price"] <= stop:
        return True, f"trailing stop ({stop:.4f})"
    return False, "tut"


def position_size(equity, free_usdt, ind, step, min_notional):
    stop_dist = ATR_STOP_MULT * ind["atr"]
    if stop_dist <= 0:
        return 0.0
    risk_amount = equity * RISK_PER_TRADE
    qty = risk_amount / stop_dist
    cost = qty * ind["price"]
    max_cost = free_usdt * 0.95
    if cost > max_cost:
        qty = max_cost / ind["price"]
    qty = round_step(qty, step)
    if qty * ind["price"] < max(min_notional, MIN_NOTIONAL_FALLBACK):
        return 0.0
    return qty


def step_decimals(step):
    s = f"{step:.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def round_step(qty, step):
    if step and step > 0:
        q = math.floor(qty / step) * step
        return float(f"{q:.{step_decimals(step)}f}")
    return float(f"{qty:.6f}")


# =============================== BINANCE (TESTNET) ===============================
def get_client():
    c = Client(BINANCE_KEY, BINANCE_SECRET, testnet=True)
    return c


def get_balances(client):
    acct = client.get_account()
    return {b["asset"]: float(b["free"]) for b in acct["balances"]}


def safe_filters(client, symbol):
    """(tradable, stepSize, minNotional) - hata olursa guvenli varsayilan."""
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            return False, 0.0, MIN_NOTIONAL_FALLBACK
        tradable = info.get("status") == "TRADING"
        step, min_n = 0.0, MIN_NOTIONAL_FALLBACK
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
            if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
                min_n = float(f.get("minNotional", f.get("notional", MIN_NOTIONAL_FALLBACK)))
        return tradable, step, min_n
    except Exception as e:
        print(f"{symbol} filtre hatasi: {e}")
        return False, 0.0, MIN_NOTIONAL_FALLBACK


def place_market(client, symbol, side, qty):
    return client.create_order(symbol=symbol, side=side, type="MARKET", quantity=qty)


# =============================== HABER / RISK (Claude) ===============================
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
        d = r.json()["data"][0]
        return int(d["value"])
    except Exception:
        return None


def global_risk_state(btc_change_24h):
    # 1) Piyasa temelli sinyal (her zaman calisir)
    market = "NORMAL"
    if btc_change_24h is not None:
        if btc_change_24h <= -10:
            market = "HIGH"
        elif btc_change_24h <= -5:
            market = "ELEVATED"
    fg = fear_greed()
    if fg is not None and fg <= 15 and market == "NORMAL":
        market = "ELEVATED"

    # 2) AI/haber temelli sinyal (opsiyonel)
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


# =============================== ANA AKIS ===============================
def main():
    dry = " (DENEME: emir gonderilmez)" if DRY_RUN else ""
    print(f"=== Bot calisiyor{dry} ===")

    if not (BINANCE_KEY and BINANCE_SECRET):
        notify("HATA: Binance testnet anahtarlari yok (BINANCE_API_KEY/SECRET).")
        print("HATA: Binance anahtarlari yok."); return

    client = get_client()
    try:
        balances = get_balances(client)
    except Exception as e:
        notify(f"Binance testnet'e baglanilamadi: {e}")
        print("Baglanti hatasi:", e); return

    state = load_state()
    meta = state.setdefault("_meta", {"peak_equity": None})

    # Veri + gosterge
    data, btc_change = {}, None
    for sym in SYMBOLS:
        try:
            df = fetch_klines(sym)
            data[sym] = compute_indicators(df)
            if sym == "BTCUSDT" and len(df) > 25:
                btc_change = (df["close"].iloc[-1] / df["close"].iloc[-25] - 1) * 100
        except Exception as e:
            print(f"{sym} veri hatasi: {e}")

    if not data:
        notify("Veri cekilemedi (borsa erisilemiyor olabilir)."); return

    risk, risk_detail = global_risk_state(btc_change)

    # Equity + mevcut pozisyonlar
    free_usdt = balances.get("USDT", 0.0)
    holdings, equity = {}, free_usdt
    for sym, ind in data.items():
        qty = balances.get(sym.replace("USDT", ""), 0.0)
        val = qty * ind["price"]
        if val >= 1.0:
            holdings[sym] = qty
        equity += val

    if meta["peak_equity"] is None or equity > meta["peak_equity"]:
        meta["peak_equity"] = equity
    dd = (equity / meta["peak_equity"] - 1) if meta["peak_equity"] else 0.0
    halt_new = dd <= -MAX_DRAWDOWN_HALT

    lines = [f"Kuresel risk: {risk}  ({risk_detail})",
             f"Equity ~ {equity:.2f} USDT | tepe {meta['peak_equity']:.2f} | dusus {dd*100:.1f}%"
             + ("  YENI ALIM DURDURULDU" if halt_new else "")]

    open_positions = len(holdings)

    for sym in SYMBOLS:
        if sym not in data:
            continue
        ind, base = data[sym], sym.replace("USDT", "")
        st = state.setdefault(sym, {"in_position": False, "entry": None,
                                    "high": None, "last_trade": 0})
        in_pos = sym in holdings

        # Durum uzlastirma (gercek bakiye esastir)
        if in_pos:
            st["in_position"] = True
            st["entry"] = st.get("entry") or ind["price"]
            st["high"] = max(st.get("high") or ind["price"], ind["price"])
        else:
            st.update({"in_position": False, "entry": None, "high": None})

        # --- SAT? ---
        if in_pos:
            sell, why = want_to_sell(ind, st["high"])
            if sell:
                _, step, _ = safe_filters(client, sym)
                qty = round_step(holdings[sym], step)
                if DRY_RUN:
                    lines.append(f"[SAT-deneme] {base}: {why} @ {ind['price']:.4f}")
                elif qty > 0:
                    try:
                        place_market(client, sym, "SELL", qty)
                        st.update({"in_position": False, "entry": None,
                                   "high": None, "last_trade": time.time()})
                        lines.append(f"[SATILDI] {base}: {why} @ {ind['price']:.4f}")
                    except Exception as e:
                        lines.append(f"[hata] {base} satis: {e}")
                else:
                    lines.append(f"{base}: satilacak miktar cok kucuk")
            else:
                lines.append(f"[tut] {base} (RSI {ind['rsi']:.0f})")
            continue

        # --- AL? ---
        buy, why = want_to_buy(ind, risk)
        if not buy:
            lines.append(f"- {base}: alim yok ({why})")
            continue
        if halt_new:
            lines.append(f"- {base}: sinyal var ama dusus limiti aktif")
            continue
        if open_positions >= MAX_POSITIONS:
            lines.append(f"- {base}: sinyal var ama pozisyon limiti dolu")
            continue
        if time.time() - st.get("last_trade", 0) < COOLDOWN_HOURS * 3600:
            lines.append(f"- {base}: sinyal var ama cooldown")
            continue
        neg, nreason = negative_news(COIN_NAMES.get(sym, base))
        if neg:
            lines.append(f"- {base}: alim iptal, olumsuz haber ({nreason})")
            continue
        tradable, step, min_n = safe_filters(client, sym)
        if not tradable:
            lines.append(f"- {base}: testnet'te islem gormuyor, atlandi")
            continue
        qty = position_size(equity, free_usdt, ind, step, min_n)
        if qty <= 0:
            lines.append(f"- {base}: sinyal var ama butce/min tutar yetmiyor")
            continue
        cost = qty * ind["price"]
        if DRY_RUN:
            lines.append(f"[AL-deneme] {base}: {why} | ~{qty} @ {ind['price']:.4f} (~{cost:.1f} USDT)")
        else:
            try:
                place_market(client, sym, "BUY", qty)
                st.update({"in_position": True, "entry": ind["price"],
                           "high": ind["price"], "last_trade": time.time()})
                free_usdt -= cost
                open_positions += 1
                lines.append(f"[ALINDI] {base}: {why} | {qty} @ {ind['price']:.4f} (~{cost:.1f} USDT)")
            except Exception as e:
                lines.append(f"[hata] {base} alis: {e}")

    save_state(state)
    report = f"Kripto bot raporu{dry}\n" + "\n".join(lines)
    notify(report)
    print(report)


if __name__ == "__main__":
    main()

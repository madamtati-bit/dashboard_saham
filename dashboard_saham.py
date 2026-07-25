import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ======================
# PENGATURAN AWAL
# ======================
st.set_page_config(page_title="Dashboard Saham IDX Pro", layout="wide")
st.title("📈 Dashboard Analisis Saham IDX - VERSI LENGKAP")
st.subheader(f"Update Terakhir: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} WIB")

# ======================
# DAFTAR SAHAM (SIAP DIEDIT)
# Anda bisa menambah/menghapus/mengganti kode di sini kapan saja
# ======================
DAFTAR_SAHAM = [
    # --- PERBANKAN ---
    "BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BTPN.JK",
    # --- TELEKOMUNIKASI & INFRASTRUKTUR ---
    "TLKM.JK", "ISAT.JK", "EXCL.JK", "FREN.JK",
    # --- BARANG KONSUMEN & MAKANAN ---
    "ASII.JK", "HMSP.JK", "ICBP.JK", "UNVR.JK", "MYOR.JK", "ROTI.JK",
    # --- INDUSTRI & BAHAN BANGUNAN ---
    "AKRA.JK", "SMGR.JK", "INTP.JK",
    # --- PERTAMBANGAN & ENERGI ---
    "ANTM.JK", "PTBA.JK", "ADRO.JK", "MEDC.JK", "TINS.JK",
    # --- PERTANIAN & PROPERTI ---
    "LSIP.JK", "SGRO.JK", "PWON.JK", "CTRA.JK"
]

# ======================
# FUNGSI MENGHITUNG INDIKATOR TEKNIS
# ======================
def hitung_indikator(df):
    close = np.asarray(df['Close']).ravel()
    high = np.asarray(df['High']).ravel()
    low = np.asarray(df['Low']).ravel()
    volume = np.asarray(df['Volume']).ravel()
    n = len(close)
    
    # EMA 9 & 20
    ema9 = np.zeros(n)
    ema9[0] = close[0]
    for i in range(1, n):
        ema9[i] = (2/(9+1)) * close[i] + (1 - 2/(9+1)) * ema9[i-1]
    df['EMA9'] = ema9
    
    ema20 = np.zeros(n)
    ema20[0] = close[0]
    for i in range(1, n):
        ema20[i] = (2/(20+1)) * close[i] + (1 - 2/(20+1)) * ema20[i-1]
    df['EMA20'] = ema20
    
    # VWAP
    harga_rata = (high + low + close) / 3
    total = np.cumsum(harga_rata * volume)
    vol_total = np.cumsum(volume)
    df['VWAP'] = total / np.where(vol_total == 0, 1, vol_total)
    
    # RSI 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_avg = pd.Series(gain).rolling(14).mean().fillna(0).values
    loss_avg = pd.Series(loss).rolling(14).mean().fillna(0).values
    rs = gain_avg / np.where(loss_avg == 0, 0.001, loss_avg)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic 14,3
    low14 = pd.Series(low).rolling(14).min().fillna(0).values
    high14 = pd.Series(high).rolling(14).max().fillna(0).values
    pctk = 100 * ((close - low14) / np.where(high14 - low14 == 0, 1, high14 - low14))
    df['%K'] = pctk
    df['%D'] = pd.Series(pctk).rolling(3).mean().fillna(0).values
    
    # MACD 12,26,9
    ema12 = np.zeros(n)
    ema12[0] = close[0]
    for i in range(1, n):
        ema12[i] = (2/(12+1)) * close[i] + (1 - 2/(12+1)) * ema12[i-1]
    ema26 = np.zeros(n)
    ema26[0] = close[0]
    for i in range(1, n):
        ema26[i] = (2/(26+1)) * close[i] + (1 - 2/(26+1)) * ema26[i-1]
    macd = ema12 - ema26
    df['MACD'] = macd
    signal = np.zeros(n)
    signal[0] = macd[0]
    for i in range(1, n):
        signal[i] = (2/(9+1)) * macd[i] + (1 - 2/(9+1)) * signal[i-1]
    df['SIGNAL'] = signal
    
    # Bollinger Bands
    sma20 = pd.Series(close).rolling(20).mean().fillna(0).values
    std20 = pd.Series(close).rolling(20).std().fillna(0).values
    df['UPPER'] = sma20 + 2 * std20
    df['LOWER'] = sma20 - 2 * std20
    
    # CCI 20
    tp = (high + low + close) / 3
    ma_tp = pd.Series(tp).rolling(20).mean().fillna(0).values
    mad = pd.Series(tp).rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x)))).fillna(0).values
    df['CCI'] = (tp - ma_tp) / np.where(0.015 * mad == 0, 0.001, 0.015 * mad)
    
    # ATR 14
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(hl, hc), lc)
    df['ATR'] = pd.Series(tr).rolling(14).mean().fillna(0).values
    
    # Volume rata-rata
    df['VOL_AVG'] = pd.Series(volume).rolling(20).mean().fillna(0).values
    
    return df

# ======================
# FUNGSI PENENTUAN SINYAL
# ======================
def dapatkan_sinyal(c, e9, e20, vw, r, k, d, m, sig, vol, vola, lowb, cc):
    skor = 0
    alasan = []
    if c > e9 and e9 > e20: skor +=1; alasan.append("✅ Tren naik EMA")
    if c > vw: skor +=1; alasan.append("✅ Harga di atas VWAP")
    if 30 <= r <= 65: skor +=1; alasan.append("✅ RSI sehat")
    if k > d and k < 50: skor +=1; alasan.append("✅ Momentum naik")
    if m > sig: skor +=1; alasan.append("✅ MACD positif")
    if vol >= 1.5 * vola: skor +=1; alasan.append("✅ Volume tinggi")
    if c >= lowb: skor +=1; alasan.append("✅ Aman Bollinger")
    if cc > -100: skor +=1; alasan.append("✅ CCI membaik")

    if skor >=7: return "🟢 BELI KUAT", skor, alasan
    elif skor >=5: return "🟡 BELI WASPADA", skor, alasan
    elif skor >=3: return "⚪ TAHAN", skor, alasan
    else: return "🔴 JUAL/HINDARI", skor, alasan

# ======================
# TAMPILAN SISI KIRI
# ======================
pilihan = st.sidebar.multiselect("Pilih Saham Analisis", DAFTAR_SAHAM, default=["BBRI.JK", "BBCA.JK", "BMRI.JK"])
refresh = st.sidebar.button("🔄 Perbarui Data Pasar")

# ======================
# PROSES PENGAMBILAN DATA
# ======================
if refresh or 'data' not in st.session_state:
    hasil = []
    with st.spinner("Mengambil data & menghitung analisis..."):
        for kode in pilihan:
            try:
                data = yf.download(kode, period="3mo", interval="1d", progress=False, auto_adjust=False)
                data = data[['Open','High','Low','Close','Volume']].copy()
                data = hitung_indikator(data)
                terkini = data.iloc[-1]
                
                # Ambil semua nilai
                c = terkini['Close'].item()
                e9 = terkini['EMA9'].item()
                e20 = terkini['EMA20'].item()
                vw = terkini['VWAP'].item()
                r = terkini['RSI'].item()
                k_val = terkini['%K'].item()
                d_val = terkini['%D'].item()
                m = terkini['MACD'].item()
                sig = terkini['SIGNAL'].item()
                vol = terkini['Volume'].item()
                vola = terkini['VOL_AVG'].item()
                lowb = terkini['LOWER'].item()
                cc = terkini['CCI'].item()
                atr = terkini['ATR'].item()
                
                sinyal, skor_angka, alasan = dapatkan_sinyal(c, e9, e20, vw, r, k_val, d_val, m, sig, vol, vola, lowb, cc)
                hasil.append({
                    "Kode": kode.replace(".JK",""),
                    "Harga": f"Rp{c:,.0f}",
                    "RSI": f"{r:.1f}",
                    "MACD": f"{m:.2f}",
                    "ATR": f"{atr:.0f}",
                    "Skor": f"{skor_angka}/8",
                    "SINYAL": sinyal,
                    "Keterangan": ", ".join(alasan)
                })
            except Exception as e:
                st.warning(f"Gagal ambil {kode}: {str(e)}")
    st.session_state.data = pd.DataFrame(hasil)

# ======================
# TAMPILAN TABEL UTAMA
# ======================
if not st.session_state.data.empty:
    st.subheader("📋 Hasil Analisis Semua Saham Pilihan")
    st.dataframe(st.session_state.data.sort_values("SINYAL", ascending=False), use_container_width=True)
else:
    st.error("Tidak ada data. Coba tekan tombol Perbarui Data.")

st.markdown("---")

# ======================
# GRAFIK & RINCIAN PER SAHAM
# ======================
if not st.session_state.data.empty:
    pilih = st.selectbox("🔍 Lihat Rincian Lengkap & Grafik", DAFTAR_SAHAM)
    datagraf = yf.download(pilih, period="3mo", interval="1d", progress=False, auto_adjust=False)
    datagraf = datagraf[['Open','High','Low','Close','Volume']].copy()
    datagraf = hitung_indikator(datagraf)
    terkini = datagraf.iloc[-1]
    
    c = terkini['Close'].item()
    atr = terkini['ATR'].item()
    e9 = terkini['EMA9'].item()
    e20 = terkini['EMA20'].item()
    vw = terkini['VWAP'].item()
    r = terkini['RSI'].item()
    k_val = terkini['%K'].item()
    d_val = terkini['%D'].item()
    m = terkini['MACD'].item()
    sig = terkini['SIGNAL'].item()
    vol = terkini['Volume'].item()
    vola = terkini['VOL_AVG'].item()
    lowb = terkini['LOWER'].item()
    cc = terkini['CCI'].item()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"📊 Ringkasan: {pilih.replace('.JK','')}")
        st.write(f"💵 Harga Sekarang: **Rp{c:,.0f}**")
        st.write(f"🛑 Stop Loss Aman: Rp{c - 1.5*atr:,.0f}")
        st.write(f"🎯 Target Wajar: Rp{c + 2*atr:,.0f}")
        sinyal, skor_angka, alasan = dapatkan_sinyal(c, e9, e20, vw, r, k_val, d_val, m, sig, vol, vola, lowb, cc)
        st.info(f"KESIMPULAN: {sinyal} (Nilai: {skor_angka}/8)")
        for a in alasan: st.write(a)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=datagraf.index, open=datagraf['Open'], high=datagraf['High'], low=datagraf['Low'], close=datagraf['Close'], name='Harga'))
        fig.add_trace(go.Scatter(x=datagraf.index, y=datagraf['EMA9'], name='EMA 9', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=datagraf.index, y=datagraf['EMA20'], name='EMA 20', line=dict(color='blue', width=1.5)))
        fig.update_layout(height=400, xaxis_rangeslider_visible=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("💡 Panduan: Gabungkan sinyal teknis ini dengan berita fundamental perusahaan untuk keputusan paling tepat.")

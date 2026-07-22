import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Süper Lig Kahini", page_icon="⚽", layout="centered")

# --- CSS İLE GÖRSELLİK ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 10px 24px;
        border-radius: 12px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODELİ YÜKLE ---
@st.cache_resource
def model_yukle():
    try:
        return joblib.load('final_voting_model.pkl')
    except:
        return None

paket = model_yukle()

if paket is None:
    st.error("Model dosyası (final_voting_model.pkl) bulunamadı! Lütfen önce model kodunu çalıştırın.")
    st.stop()

model = paket['model']
scaler = paket['scaler'] # Scaler'ı çektik
stats = paket['takim_guc_sozlugu']
isim_id = paket['isimden_idye']
id_isim = paket['id_den_isime']

# --- BAŞLIK ---
st.title("⚽ Süper Lig Kupon Sihirbazı")
st.markdown("**Yöntem:** Ensemble Learning (RF + SVM + LR Voting Classifier)")
st.info(
    "Bu eğitim amaçlı sistem, geçmiş Süper Lig maçlarından türetilen "
    "takım performans istatistiklerini analiz eder."
)

# --- KULLANICI GİRİŞLERİ ---
st.markdown("### 🏟️ Maç Seçimi")
col1, col2 = st.columns(2)

takim_listesi = sorted(list(isim_id.keys()))

with col1:
    ev_sahibi = st.selectbox("Ev Sahibi Takım", takim_listesi, index=0)

with col2:
    misafir = st.selectbox("Deplasman Takımı", takim_listesi, index=1)

# --- ANALİZ BUTONU ---
if st.button("🔮 MAÇI ANALİZ ET", use_container_width=True):
    if ev_sahibi == misafir:
        st.warning("⚠️ Lütfen iki farklı takım seçiniz!")
    else:
        # 1. Verileri Hazırla
        ev_id = isim_id[ev_sahibi]
        mis_id = isim_id[misafir]
        
        ev_stat = stats.get(ev_id, {'Puan_Ort': 0, 'Saldiri': 0, 'Savunma': 0})
        mis_stat = stats.get(mis_id, {'Puan_Ort': 0, 'Saldiri': 0, 'Savunma': 0})
        
        # DataFrame oluştur (Sütun sırası eğitimle aynı olmalı)
        input_data = pd.DataFrame({
            'Ev_Puan_Ort': [ev_stat['Puan_Ort']],
            'Dep_Puan_Ort': [mis_stat['Puan_Ort']],
            'Ev_Saldiri': [ev_stat['Saldiri']],
            'Dep_Savunma': [mis_stat['Savunma']],
            'Guc_Farki': [ev_stat['Puan_Ort'] - mis_stat['Puan_Ort']],
            'Ev_sahibi_ID': [ev_id],
            'Misafir_ID': [mis_id]
        })
        
        # 2. ÖLÇEKLENDİRME (SCALING) - KRİTİK ADIM
        # Model scaled veriyle eğitildi, o yüzden inputu da scale ediyoruz
        input_scaled = scaler.transform(input_data)
        
        # 3. Tahmin
        olasiliklar = model.predict_proba(input_scaled)[0]
        tahmin = model.predict(input_scaled)[0]
        
        prob_beraberlik = olasiliklar[0]
        prob_ev = olasiliklar[1]
        prob_dep = olasiliklar[2]
        
        # --- SONUÇ EKRANI ---
        st.markdown("---")
        st.subheader(f"📊 {ev_sahibi} vs {misafir} Analizi")
        
        # İstatistikler
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ev Puan Ort.", f"{ev_stat['Puan_Ort']:.2f}")
        k2.metric("Ev Hücum", f"{ev_stat['Saldiri']:.2f} Gol")
        k3.metric("Dep Puan Ort.", f"{mis_stat['Puan_Ort']:.2f}")
        k4.metric("Dep Savunma", f"{mis_stat['Savunma']:.2f} Gol")
        
        st.markdown("---")
        
        # Kazananı Yazdır
        if tahmin == 1:
            st.success(f"🏆 **TAHMİN: {ev_sahibi} KAZANIR**")
        elif tahmin == 2:
            st.error(f"🏆 **TAHMİN: {misafir} KAZANIR**")
        else:
            st.info(f"⚖️ **TAHMİN: BERABERLİK**")
            
        # Barlar
        st.write("### 📈 Olasılık Dağılımı")
        
        col_bar1, col_bar2, col_bar3 = st.columns(3)
        with col_bar1:
            st.write(f"🏠 **Ev Sahibi**")
            st.progress(prob_ev)
            st.caption(f"%{prob_ev*100:.1f}")
            
        with col_bar2:
            st.write(f"⚖️ **Beraberlik**")
            st.progress(prob_beraberlik)
            st.caption(f"%{prob_beraberlik*100:.1f}")
            
        with col_bar3:
            st.write(f"✈️ **Deplasman**")
            st.progress(prob_dep)
            st.caption(f"%{prob_dep*100:.1f}")
            
        # Yorum
        guven = max(prob_ev, prob_beraberlik, prob_dep)
        if guven > 0.55:
            st.success(f"🤖 Yapay Zeka bu tahminden **%{guven*100:.1f}** oranında emin.")
        else:
            st.warning(f"🤖 Maç çok ortada görünüyor. En yüksek ihtimal **%{guven*100:.1f}** ile belirlendi.")

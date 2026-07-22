# model_olustur.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

print("⏳ Model oluşturuluyor... Lütfen bekleyiniz.")

# 1. Veriyi Yükle
try:
    df = pd.read_csv('egitim_verisi_final.csv')
except FileNotFoundError:
    print("HATA: 'egitim_verisi_final.csv' dosyası bulunamadı!")
    exit()

# 2. İstatistikleri Hesapla
stats = {} 
ev_ids = df['Ev_sahibi_ID'].values; dep_ids = df['Misafir_ID'].values
ev_gols = df['Ev_sahibi_gol'].values; dep_gols = df['Misafir_takim_gol'].values
sonuclar = df['Sonuc'].values

for i, (ev_id, dep_id, ev_gol, dep_gol, sonuc) in enumerate(zip(ev_ids, dep_ids, ev_gols, dep_gols, sonuclar)):
    if ev_id not in stats: stats[ev_id] = [0, 0, 0, 0]
    if dep_id not in stats: stats[dep_id] = [0, 0, 0, 0]
    
    # Güncelle
    p_ev = 3 if sonuc == 1 else (1 if sonuc == 0 else 0)
    p_dep = 3 if sonuc == 2 else (1 if sonuc == 0 else 0)
    
    stats[ev_id][0]+=p_ev; stats[ev_id][1]+=ev_gol; stats[ev_id][2]+=dep_gol; stats[ev_id][3]+=1
    stats[dep_id][0]+=p_dep; stats[dep_id][1]+=dep_gol; stats[dep_id][2]+=ev_gol; stats[dep_id][3]+=1

takim_guc_sozlugu = {}
for tid, val in stats.items():
    if val[3] > 0:
        takim_guc_sozlugu[tid] = {
            'Puan_Ort': val[0]/val[3],
            'Saldiri': val[1]/val[3],
            'Savunma': val[2]/val[3]
        }

isimden_idye = {}
id_den_isime = {}
takimlar = df[['Ev_sahibi', 'Ev_sahibi_ID']].drop_duplicates()
for index, row in takimlar.iterrows():
    isimden_idye[row['Ev_sahibi']] = row['Ev_sahibi_ID']
    id_den_isime[row['Ev_sahibi_ID']] = row['Ev_sahibi']

# 3. Modeli Eğit
df['Ev_Puan_Ort'] = df['Ev_sahibi_ID'].map(lambda x: takim_guc_sozlugu.get(x, {}).get('Puan_Ort', 0))
df['Dep_Puan_Ort'] = df['Misafir_ID'].map(lambda x: takim_guc_sozlugu.get(x, {}).get('Puan_Ort', 0))
df['Ev_Saldiri'] = df['Ev_sahibi_ID'].map(lambda x: takim_guc_sozlugu.get(x, {}).get('Saldiri', 0))
df['Dep_Savunma'] = df['Misafir_ID'].map(lambda x: takim_guc_sozlugu.get(x, {}).get('Savunma', 0))
df['Guc_Farki'] = df['Ev_Puan_Ort'] - df['Dep_Puan_Ort']

X = df[['Ev_Puan_Ort', 'Dep_Puan_Ort', 'Ev_Saldiri', 'Dep_Savunma', 'Guc_Farki', 'Ev_sahibi_ID', 'Misafir_ID']]
y = df['Sonuc']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf1 = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=4, random_state=42)
clf2 = SVC(probability=True, kernel='rbf', C=1.0)
clf3 = LogisticRegression(max_iter=2000)

voting_model = VotingClassifier(estimators=[('rf', clf1), ('svm', clf2), ('lr', clf3)], voting='soft')
voting_model.fit(X_scaled, y)

paket = {
    'model': voting_model,
    'scaler': scaler,
    'takim_guc_sozlugu': takim_guc_sozlugu,
    'isimden_idye': isimden_idye,
    'id_den_isime': id_den_isime
}

joblib.dump(paket, 'final_voting_model.pkl')
print("✅ BAŞARILI! Model dosyası yenilendi: 'final_voting_model.pkl'")

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Panduan Istilah | Brent Oil Forecast", page_icon="📖", layout="wide")

st.markdown("""
<style>
    .glossary-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(212, 160, 23, 0.25);
    }
    .term-card {
        background: rgba(212, 160, 23, 0.06);
        border-left: 4px solid #d4a017;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.9rem 0;
    }
    .term-title { font-size: 1.15rem; font-weight: 700; color: #f2f2f2; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glossary-hero">
    <h2>📖 Panduan Istilah & Parameter Input</h2>
    <p style="color:#d4a017;">Penjelasan setiap kontrol yang ada di dashboard prediksi harga Brent, dalam bahasa yang mudah dipahami.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Dashboard ini memakai istilah-istilah pasar keuangan/geopolitik yang mungkin belum familiar.
Halaman ini menjelaskan **apa arti tiap input**, **kenapa dibutuhkan**, dan **apa efeknya** ke hasil prediksi.
""")

# =========================================================
# BAGIAN 1 — RAMALAN KE DEPAN
# =========================================================
st.markdown("## 🔮 Parameter Ramalan ke Depan")

st.markdown("""
<div class="term-card">
<div class="term-title">📅 Horizon (hari)</div>
Jumlah hari ke depan yang ingin diramalkan, dihitung mulai dari hari terakhir di data kamu.
Contoh: kalau data terakhir tanggal 10 Juli dan horizon diisi <b>30</b>, dashboard akan
meramalkan harga setiap hari dari 11 Juli sampai 9 Agustus.
<br><br>
<b>Cara kerjanya:</b> model ini aslinya cuma bisa menebak <i>satu hari ke depan</i> (t+1).
Untuk meramalkan lebih dari satu hari, dashboard memakai hasil tebakan hari ini sebagai
"data pura-pura" untuk menebak hari berikutnya, dan seterusnya secara berantai.
<br><br>
<b>Efek ke akurasi:</b> makin besar angka horizon, makin panjang rantai tebak-menebaknya,
makin besar juga potensi errornya menumpuk. Horizon 7 hari jauh lebih bisa dipercaya
dibanding horizon 30 atau 90 hari — anggap horizon panjang sebagai <b>skenario kasar</b>,
bukan ramalan presisi.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">🛢️ Tren WTI (%/hari)</div>
<b>WTI (West Texas Intermediate)</b> adalah harga acuan minyak mentah versi Amerika —
"saudara" dari harga Brent yang jadi fokus utama dashboard ini. Brent dan WTI biasanya
bergerak searah karena sama-sama dipengaruhi permintaan/pasokan minyak global.
<br><br>
Karena model butuh tahu harga WTI di hari-hari mendatang (tapi kita tidak punya data
itu), kamu diminta memberi <b>asumsi tren harian</b> dalam persen:
<ul>
<li><b>0%</b> = asumsi harga WTI stagnan (flat), tidak naik/turun</li>
<li><b>+0.5%</b> = asumsi WTI naik 0.5% setiap hari (dalam 30 hari, itu bisa berarti
kenaikan total sekitar 16%, karena efeknya berbunga majemuk/compounding)</li>
<li><b>-0.5%</b> = asumsi WTI turun 0.5% setiap hari</li>
</ul>
Gunakan nilai kecil (biasanya di kisaran -0.5% s/d +0.5%) — angka harian sekecil apa pun
akan membesar drastis kalau horizonnya panjang.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">💵 Tren DXY (%/hari)</div>
<b>DXY (US Dollar Index)</b> mengukur kekuatan Dolar AS dibanding sekeranjang mata uang
utama dunia (Euro, Yen, dll). Ini penting karena minyak dunia diperdagangkan pakai Dolar —
kalau Dolar menguat (DXY naik), minyak biasanya jadi lebih mahal bagi negara-negara dengan
mata uang lain, sehingga permintaan turun dan <b>harga minyak cenderung tertekan (turun)</b>.
Hubungannya sering berkebalikan (inverse) dengan harga minyak.
<br><br>
Slider ini mengasumsikan seberapa persen DXY berubah tiap hari selama periode ramalan.
Sama seperti Tren WTI, gunakan nilai kecil karena efeknya majemuk (compounding) sepanjang horizon.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">📉 Tren VIX (%/hari)</div>
<b>VIX</b> sering dijuluki "indeks ketakutan" (fear index) Wall Street — mengukur seberapa
besar investor memperkirakan gejolak (volatilitas) pasar saham AS dalam waktu dekat.
VIX tinggi = investor cemas/panik, biasanya muncul bareng krisis ekonomi atau geopolitik,
dan sering diikuti gejolak harga komoditas termasuk minyak.
<br><br>
Slider ini mengasumsikan tren harian VIX ke depan. Nilai positif berarti kamu
mengasumsikan kecemasan pasar meningkat; nilai negatif berarti pasar diasumsikan makin tenang.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">🌍 Tren GPR (%/hari)</div>
<b>GPR (Geopolitical Risk Index)</b> adalah indeks yang mengukur tingkat ketegangan
geopolitik global (perang, sanksi, konflik, dll.) berdasarkan analisis pemberitaan dunia.
GPR tinggi biasanya berkorelasi dengan risiko gangguan pasokan minyak — jadi bisa
mendorong harga naik.
<br><br>
Slider ini mengasumsikan tren harian index tersebut ke depan. Kalau kamu memperkirakan
akan ada eskalasi konflik/ketegangan geopolitik, gunakan tren positif; kalau situasi
diperkirakan mereda, gunakan tren negatif.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="term-card">
<div class="term-title">⚠️ Severity & Status Event Geopolitik</div>
Dua kontrol tambahan yang mengasumsikan <b>ada/tidaknya kejadian geopolitik besar</b>
(perang, sanksi, serangan infrastruktur, dll.) selama periode ramalan, dan seberapa
parah dampaknya (skala 1-10). Lihat tabel skala lengkapnya di bagian bawah halaman ini.
</div>
""", unsafe_allow_html=True)

# =========================================================
# BAGIAN 2 — SEVERITY LEGEND
# =========================================================
st.markdown("## ⚠️ Skala Severity Event Geopolitik (1-10)")
st.markdown("""
Dikalibrasi dari pola event nyata dalam dataset riset ini. Skala 1-5 diekstrapolasi
sebagai panduan relatif karena dataset contoh hanya mencatat event-event besar
(severity 6 ke atas) — bukan angka baku, tapi cukup untuk memberi rasa skala.
""")

SEVERITY_LEGEND = {
    1:  ("Sangat Rendah", "Berita rutin pasar, tanpa gangguan pasokan nyata"),
    2:  ("Sangat Rendah", "Rumor/ketegangan kecil, dampak harga minimal"),
    3:  ("Rendah",        "Friksi diplomatik lokal, tidak memengaruhi ekspor"),
    4:  ("Rendah",        "Sanksi simbolis atau pembicaraan OPEC rutin"),
    5:  ("Sedang",        "Krisis diplomatik regional tanpa gangguan produksi"),
    6:  ("Sedang",        "Contoh: krisis diplomatik Qatar, pencabutan sanksi nuklir Iran"),
    7:  ("Tinggi",        "Contoh: tumpahan minyak Deepwater Horizon, intervensi militer Yaman, krisis energi global"),
    8:  ("Tinggi",        "Contoh: embargo minyak Iran/Rusia oleh UE, aneksasi Crimea, blokade Terusan Suez, serangan Laut Merah"),
    9:  ("Sangat Tinggi", "Contoh: Perang Sipil Libya, pembunuhan Jenderal Soleimani, larangan impor minyak Rusia AS, Perang Israel-Hamas"),
    10: ("Ekstrem",       "Contoh: serangan fasilitas Aramco Abqaiq, invasi Rusia ke Ukraina, harga WTI negatif, penutupan Selat Hormuz"),
}
legend_df = pd.DataFrame(
    [{"Severity": k, "Kategori": v[0], "Contoh / Deskripsi": v[1]} for k, v in SEVERITY_LEGEND.items()]
)
st.dataframe(legend_df, use_container_width=True, hide_index=True)

# =========================================================
# BAGIAN 3 — MODE INPUT
# =========================================================
st.markdown("## 📥 Dua Mode Sumber Data")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="term-card">
    <div class="term-title">📤 Upload CSV</div>
    Memakai data historis riil (harga, indikator makro, event) yang kamu unggah.
    Model membaca 30 hari terakhir dari data itu sebagai titik awal ramalan.
    <b>Lebih akurat</b> karena pola lag/volatilitas dihitung dari data sungguhan.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="term-card">
    <div class="term-title">✍️ Input Manual</div>
    Kamu memasukkan kondisi pasar hari ini secara manual (harga, DXY, VIX, GPR, dll.),
    lalu dashboard membuat skenario "flat" 30 hari dari angka itu sebagai titik awal.
    <b>Lebih cepat</b> tapi jauh lebih kasar karena tidak memakai pola historis asli.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="term-card">
<div class="term-title">💡 Ringkasnya</div>
Ramalan ke depan di dashboard ini adalah <b>simulasi skenario</b>, bukan prediksi pasti.
Semakin jauh horizonnya dan semakin ekstrem asumsi tren yang kamu masukkan, semakin besar
kemungkinan hasilnya melenceng dari kenyataan. Gunakan sebagai alat bantu eksplorasi
"bagaimana jika", bukan sebagai patokan keputusan finansial.
</div>
""", unsafe_allow_html=True)

st.page_link("app.py", label="⬅️ Kembali ke Dashboard", icon="🛢️")

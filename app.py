"""
Streamlit — Powder Shade Alignment

Jalankan:  streamlit run app.py
Semua perhitungan ada di core.py (disalin dari notebook, metode tidak diubah).
"""
from io import BytesIO
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Powder Shade Alignment", layout="wide")

DATA_DEFAULT = Path(__file__).parent / "Data Shade Alignment.xlsx"
LABEL_UT = {"theta": "θ — garis a–b terbaik", "hue": "Hue otomatis (optimasi decision tree)"}
LABEL_LEVEL = {True: "Median per undertone (hasil prediksi)", False: "Median semua data"}

st.markdown("""
<style>
.swatch-table {border-collapse: collapse; margin: 0.25rem 0 1rem 0; font-size: 0.85rem;}
.swatch-table th {text-align: left; padding: 4px 10px; font-weight: 600; color: #555;}
.swatch-table td {padding: 4px 10px; vertical-align: middle;}
.swatch-chip {width: 92px; height: 38px; border-radius: 4px; border: 1px solid rgba(0,0,0,.12);}
.swatch-lab {font-size: 0.72rem; color: #666; margin-top: 2px; white-space: nowrap;}
</style>
""", unsafe_allow_html=True)


# =====================================================================================
# Cache: training & rekomendasi
# =====================================================================================
@st.cache_resource(show_spinner="Melatih model Stage 1, 2, 3, undertone, dan level...")
def latih(file_bytes: bytes, sheet: str, versi: str):
    df = core.load_data(BytesIO(file_bytes), sheet_name=sheet)
    return df, core.latih_semua(df, versi)


def katalog_powder_berlabel(T, metode_ut, level_per_ut):
    return core.klasifikasi_powder(T["katalog_powder"], T["model_undertone"], T["model_level"],
                                   metode_ut, level_per_ut)


@st.cache_data(show_spinner="Menghitung rekomendasi...")
def hitung_rekomendasi(file_bytes, sheet, versi, stage, bare, top_k, n_grid, urut, metode_ut, level_per_ut):
    _, T = latih(file_bytes, sheet, versi)
    bare_face = dict(zip(core.KANAL, bare))
    S = T["stage"]
    kat_pow = katalog_powder_berlabel(T, metode_ut, level_per_ut)
    if stage == 1:
        return core.rekomendasi_produk(bare_face, T["katalog_liquid"], S[1]["coef"], S[1]["klas"],
                                       "Liquid", S[1]["data"], top_k=top_k, n_grid=n_grid, urut=urut)
    if stage == 2:
        kat, rng, grid, info = core.rekomendasi_produk(bare_face, kat_pow, S[2]["coef"], S[2]["klas"],
                                                       "Powder", S[2]["data"], top_k=top_k, n_grid=n_grid, urut=urut)
        grid = tambah_kelas_powder(grid, T, metode_ut, level_per_ut, ("L", "a", "b"))
        return kat, rng, grid, info
    kat, rng, grid, info = core.rekomendasi_gabungan(bare_face, T["katalog_liquid"], kat_pow, S[1]["coef"],
                                                     S[3]["coef"], S[3]["klas"], S[3]["data"],
                                                     top_k=top_k, n_grid=n_grid, urut=urut)
    grid = tambah_kelas_powder(grid, T, metode_ut, level_per_ut, ("L_pow", "a_pow", "b_pow"))
    return kat, rng, grid, info


def tambah_kelas_powder(tabel, T, metode_ut, level_per_ut, kolom):
    # Klasifikasi undertone & level untuk titik powder hasil grid (model training powder)
    if tabel.empty:
        return tabel
    tmp = pd.DataFrame(tabel[list(kolom)].values, columns=core.KANAL)
    kls = core.klasifikasi_powder(tmp, T["model_undertone"], T["model_level"], metode_ut, level_per_ut)
    tabel = tabel.copy()
    tabel["pred_Undertone_powder"] = kls["pred_Undertone"].values
    tabel["pred_Level_powder"] = kls["pred_Level"].values
    return tabel


# =====================================================================================
# Helper tampilan
# =====================================================================================
def chip(lab):
    hexc = core.lab_to_hex(np.asarray(lab, float).reshape(1, 3))[0]
    L, a, b = lab
    return (f'<div class="swatch-chip" style="background:{hexc}"></div>'
            f'<div class="swatch-lab">L {L:.1f} · a {a:.1f} · b {b:.1f}</div>')


def swatch_html(kolom, label_baris):
    # kolom: dict {judul: array (N, 3)}
    head = "<tr><th></th>" + "".join(f"<th>{k}</th>" for k in kolom) + "</tr>"
    body = ""
    n = len(label_baris)
    for i in range(n):
        body += f"<tr><td>{label_baris[i]}</td>" + "".join(f"<td>{chip(v[i])}</td>" for v in kolom.values()) + "</tr>"
    st.markdown(f'<table class="swatch-table">{head}{body}</table>', unsafe_allow_html=True)


def fmt_tabel(df, persen=None):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = np.where(df[c], "Match", "Unmatch")
    return df.round(3)


def unduh(df, nama, key):
    st.download_button("Unduh CSV", df.to_csv(index=True).encode("utf-8"), file_name=nama,
                       mime="text/csv", key=key)


# =====================================================================================
# Sidebar
# =====================================================================================
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("File Excel data shade alignment", type=["xlsx"])
    if up is not None:
        file_bytes = up.getvalue()
    elif DATA_DEFAULT.exists():
        file_bytes = DATA_DEFAULT.read_bytes()
        st.caption(f"Memakai file bawaan: {DATA_DEFAULT.name}")
    else:
        st.info("Unggah file Excel untuk mulai.")
        st.stop()
    sheets = pd.ExcelFile(BytesIO(file_bytes)).sheet_names
    sheet = st.selectbox("Sheet", sheets, index=sheets.index("New_data") if "New_data" in sheets else 0)

    st.header("Model")
    versi = st.selectbox("Metode regresi warna", list(core.VERSI_REGRESI),
                         index=0, format_func=lambda v: f"{v}")
    st.caption(f"`{core.VERSI_REGRESI[versi]}`")
    st.caption("Klasifikasi Match/Unmatch: Logistic Regression (fitur dL, da, db, dE), "
               "threshold Youden's J.")

    st.header("Klasifikasi powder")
    metode_ut = st.radio("Undertone powder", list(LABEL_UT), format_func=LABEL_UT.get)
    level_per_ut = st.radio("Leveling powder (metode median)", [True, False], format_func=LABEL_LEVEL.get)

try:
    df_raw, T = latih(file_bytes, sheet, versi)
except Exception as e:  # kolom tidak lengkap, dsb.
    st.error(f"Gagal melatih model dari sheet '{sheet}': {e}")
    st.stop()
S = T["stage"]
file_key = hashlib.md5(file_bytes).hexdigest()

st.title("Powder Shade Alignment")
st.caption(f"Data: {df_raw.shape[0]} baris · regresi versi **{versi}** · undertone powder "
           f"**{LABEL_UT[metode_ut]}** · leveling **{LABEL_LEVEL[level_per_ut]}**")

tab_train, tab_pred = st.tabs(["Training & evaluasi", "Prediksi kulit baru"])


# =====================================================================================
# TAB 1 — Training & evaluasi
# =====================================================================================
def tampil_stage(s, key):
    k = s["klas"]
    c = st.columns(6)
    c[0].metric("Jumlah data", len(s["y"]))
    c[1].metric("Match / Unmatch", f"{int(s['y'].sum())} / {int((s['y'] == 0).sum())}")
    c[2].metric("R² rata-rata", f"{s['persamaan']['R²'].mean():.3f}")
    c[3].metric("AUC", f"{k['auc']:.3f}")
    c[4].metric("Akurasi", f"{k['acc'] * 100:.1f}%")
    c[5].metric("Threshold P(Match)", f"{k['threshold']:.3f}")

    st.subheader("Perbandingan 3 versi regresi")
    st.caption("AUC & akurasi = hasil classifier Match/Unmatch bila prediksi warna memakai versi tersebut.")
    banding = s["banding"].copy()
    st.dataframe(banding.round(4).style.apply(
        lambda r: ["background-color: #e8eeff" if r["Versi"] == versi else "" for _ in r], axis=1),
        hide_index=True)
    with st.expander("Persamaan regresi semua versi"):
        for v, p in s["persamaan_versi"].items():
            st.markdown(f"**{v}** — `{core.VERSI_REGRESI[v]}`")
            st.dataframe(p.round(4), hide_index=True)

    st.subheader(f"Regresi warna versi terpilih: {versi}")
    st.dataframe(s["persamaan"].round(4), hide_index=True)
    st.pyplot(core.plot_aktual_vs_prediksi(s["aktual"], s["pred"], f"{s['label_target']} (versi {versi})"),
              clear_figure=True)

    st.subheader("Model klasifikasi Match/Unmatch")
    pl = core.persamaan_logistik(k)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.code(f"{pl['z']}\n{pl['p']}\n{pl['aturan']}", language=None)
        st.dataframe(k["report"].round(3))
    with col2:
        st.dataframe(pl["koef"].round(4), hide_index=True)
        st.caption("Koefisien memakai satuan asli fitur (tanpa standardisasi), jadi besarnya tidak "
                   "langsung menunjukkan fitur mana yang paling berpengaruh.")
    st.pyplot(core.plot_evaluasi(k, s["nama"]), clear_figure=True)

    with st.expander("Fitur delta (prediksi − bare face) & uji tiap fitur"):
        st.dataframe(s["uji"].round(4), hide_index=True)
        st.pyplot(core.plot_boxplot_fitur(s["delta"], s["y"], f"{s['nama']}: distribusi fitur delta"),
                  clear_figure=True)
    with st.expander("Swatch warna data training"):
        swatch_html(s["swatch"], [f"Baris {i}" for i in s["data"].index])


with tab_train:
    st.caption("Semua akurasi dihitung pada data latih (tanpa cross-validation) dan sampelnya kecil, "
               "jadi lebih tepat dibaca sebagai arah/tren.")
    t1, t2, t3, t4, t5 = st.tabs(["Stage 1 — Liquid", "Stage 2 — Powder", "Stage 3 — Liquid + Powder",
                                  "Ringkasan versi", "Undertone & level powder"])
    with t1:
        tampil_stage(S[1], "s1")
    with t2:
        tampil_stage(S[2], "s2")
    with t3:
        st.info("Langkah 1: after liquid diprediksi dengan regresi Stage 1. "
                "Langkah 2: prediksi after liquid + powder → hasil akhir (regresi baru).")
        tampil_stage(S[3], "s3")
    with t4:
        rv = T["ringkasan_versi"]
        st.dataframe(rv.round(4), hide_index=True)
        for metrik in ["R² rata-rata", "AUC Match/Unmatch", "Akurasi Match/Unmatch"]:
            st.markdown(f"**{metrik}**")
            st.dataframe(rv.pivot(index="Stage", columns="Versi", values=metrik)[list(core.VERSI_REGRESI)].round(4))
    with t5:
        st.subheader("Klasifikasi undertone (C / N / W)")
        st.caption("Model **Powder** dipakai untuk memberi label undertone pada rekomendasi powder. "
                   "Pilih metodenya di sidebar.")
        st.dataframe(T["ringkasan_undertone"].round(4), hide_index=True)
        for nama_sumber, m in T["model_undertone"].items():
            with st.expander(f"Detail {nama_sumber}", expanded=(nama_sumber == "Powder")):
                g = m["garis"]
                teks = [f"[1] Hue manual | Akurasi {m['acc_manual']*100:.1f}%",
                        f"    Hue < {core.BATAS_HUE_MANUAL[0]}° -> Cool | {core.BATAS_HUE_MANUAL[0]}°-"
                        f"{core.BATAS_HUE_MANUAL[1]}° -> Neutral | > {core.BATAS_HUE_MANUAL[1]}° -> Warm", "",
                        f"[2] Hue otomatis | Akurasi {m['acc_hue']*100:.1f}%"]
                teks += ["    " + a for a in m["aturan_hue"]]
                teks += ["    Garis batas: " + core.garis_hue(sd) for sd in m["batas_hue"]]
                teks += ["", f"[3] Garis a-b terbaik | Akurasi {g['acc']*100:.1f}%",
                         f"    θ = {g['theta']}°  ->  skor = {g['w_a']:.4f}·a "
                         f"{'+' if g['w_b'] >= 0 else '-'} {abs(g['w_b']):.4f}·b"]
                teks += ["    " + a for a in m["aturan_skor"]]
                teks += ["    Garis batas: " + core.garis_skor(g["w_a"], g["w_b"], t) for t in g["batas"]]
                st.code("\n".join(teks), language=None)
                st.pyplot(core.plot_undertone(m, nama_sumber), clear_figure=True)

        st.subheader("Klasifikasi level L (metode median)")
        rl = T["ringkasan_level"]
        c1, c2 = st.columns(2)
        c1.markdown("**Akurasi tepat**")
        c1.dataframe(rl.pivot(index="Sumber", columns="Undertone", values="Akurasi")
                     .reindex(columns=["Semua"] + core.UNDERTONES).round(4))
        kol_dekat = f"Akurasi ±{core.TOLERANSI:g}"
        c2.markdown(f"**Akurasi ±{core.TOLERANSI:g} level**")
        c2.dataframe(rl.pivot(index="Sumber", columns="Undertone", values=kol_dekat)
                     .reindex(columns=["Semua"] + core.UNDERTONES).round(4))
        st.markdown("**Aturan level powder**")
        cols = st.columns(4)
        for col, ut in zip(cols, ["Semua"] + core.UNDERTONES):
            if ("Powder", ut) in T["model_level"]:
                col.caption(f"Powder — {ut}")
                col.dataframe(core.tabel_aturan_median(T["model_level"][("Powder", ut)]),
                              hide_index=True)
        with st.expander("Grafik level L"):
            for nama, fig in T["figs_level"].items():
                st.pyplot(fig, clear_figure=False)


# =====================================================================================
# TAB 2 — Prediksi kulit baru
# =====================================================================================
STAGE_NAMA = {1: "Stage 1 — Liquid", 2: "Stage 2 — Powder", 3: "Stage 3 — Liquid + Powder"}

with tab_pred:
    with st.form("input_kulit"):
        st.subheader("Warna kulit baru (bare face)")
        c = st.columns([1, 1, 1, 1.4])
        L = c[0].number_input("L*", value=62.5, step=0.1, format="%.2f")
        a = c[1].number_input("a*", value=11.8, step=0.1, format="%.2f")
        b = c[2].number_input("b*", value=16.4, step=0.1, format="%.2f")
        with c[3]:
            st.markdown("Pratinjau")
            st.markdown(chip((L, a, b)), unsafe_allow_html=True)
        c = st.columns(4)
        stages = c[0].multiselect("Stage yang direkomendasikan", [1, 2, 3], default=[1, 2, 3],
                                  format_func=STAGE_NAMA.get)
        top_k = c[1].slider("Jumlah rekomendasi (top-k)", 3, 20, 5)
        urut = c[2].selectbox("Urutkan berdasarkan", ["P_match", "dE"],
                              format_func=lambda u: "P(Match) tertinggi" if u == "P_match" else "dE terkecil")
        with c[3]:
            n_grid12 = st.number_input("Grid per kanal (Stage 1–2)", 5, 25, 15)
            n_grid3 = st.number_input("Grid per kanal (Stage 3)", 4, 12, 8)
        jalan = st.form_submit_button("Cari rekomendasi", type="primary")

    if jalan:
        st.session_state["input"] = dict(bare=(L, a, b), stages=stages, top_k=top_k, urut=urut,
                                          n_grid12=int(n_grid12), n_grid3=int(n_grid3))

    inp = st.session_state.get("input")
    if not inp:
        st.info("Isi warna kulit lalu tekan **Cari rekomendasi**.")
    else:
        bare = np.array(inp["bare"], float)
        if not inp["stages"]:
            st.warning("Pilih minimal satu stage.")
        kat_pow = katalog_powder_berlabel(T, metode_ut, level_per_ut)
        for stg in inp["stages"]:
            st.divider()
            st.header(STAGE_NAMA[stg])
            n_grid = inp["n_grid3"] if stg == 3 else inp["n_grid12"]
            kat, rng, grid, info = hitung_rekomendasi(file_bytes, sheet, versi, stg, tuple(inp["bare"]),
                                                      inp["top_k"], n_grid, inp["urut"], metode_ut, level_per_ut)
            for p in info["peringatan"]:
                st.warning(p)

            # ---------------- A. Katalog ----------------
            st.subheader("A. Rekomendasi dari katalog")
            satuan = "pasangan" if stg == 3 else "produk"
            st.caption(f"{info['n_match_katalog']} dari {info['n_katalog']} {satuan} katalog diprediksi Match.")
            top = kat.head(inp["top_k"])
            if stg == 3:
                kolom = ["baris_liquid", "baris_powder", "L_liq", "a_liq", "b_liq", "L_pow", "a_pow", "b_pow",
                         "label_Undertone_powder", "label_L_powder", "pred_Undertone_powder", "pred_Level_powder",
                         "after_liq_L", "after_liq_a", "after_liq_b", "pred_L", "pred_a", "pred_b",
                         "dE", "P_match", "Match"]
                st.dataframe(fmt_tabel(top[kolom]), hide_index=True)
                swatch_html({"Bare Face": np.tile(bare, (len(top), 1)),
                             "Liquid": top[["L_liq", "a_liq", "b_liq"]].values,
                             "Powder": top[["L_pow", "a_pow", "b_pow"]].values,
                             "Prediksi after liquid": top[["after_liq_L", "after_liq_a", "after_liq_b"]].values,
                             "Prediksi akhir (liq + pow)": top[["pred_L", "pred_a", "pred_b"]].values},
                            [f"Liq {l} + Pow {q}<br>P={pp:.2f} · {'Match' if m else 'Unmatch'}"
                             for l, q, pp, m in zip(top.baris_liquid, top.baris_powder, top.P_match, top.Match)])
            else:
                st.dataframe(fmt_tabel(top))
                nama = "Liquid" if stg == 1 else "Powder"
                swatch_html({"Bare Face": np.tile(bare, (len(top), 1)),
                             nama: top[core.KANAL].values,
                             f"Prediksi after {nama.lower()}": top[["pred_L", "pred_a", "pred_b"]].values},
                            [f"Baris {i}<br>P={pp:.2f} · {'Match' if m else 'Unmatch'}"
                             for i, pp, m in zip(top.index, top.P_match, top.Match)])
            if stg in (2, 3):
                st.caption(f"Kolom `label_*` = label asli katalog. Kolom `pred_Undertone` dan `pred_Level` = "
                           f"klasifikasi dari model training powder ({LABEL_UT[metode_ut]}; "
                           f"{LABEL_LEVEL[level_per_ut].lower()}).")
            with st.expander(f"Semua {satuan} katalog ({len(kat)})"):
                st.dataframe(fmt_tabel(kat))
                unduh(kat, f"rekomendasi_katalog_stage{stg}.csv", f"dl_kat_{stg}")

            # ---------------- B. Range ----------------
            st.subheader("B. Rekomendasi dari range katalog")
            if stg == 3:
                st.caption(f"Grid {info['n_grid_liq']} liquid × {info['n_grid_pow']} powder = {info['n_grid']:,} "
                           f"pasangan → {info['n_match_grid']:,} Match ({100 * info['n_match_grid'] / info['n_grid']:.1f}%).")
            else:
                st.caption(f"Grid {info['n_grid']:,} kombinasi → {info['n_match_grid']:,} Match "
                           f"({100 * info['n_match_grid'] / info['n_grid']:.1f}%).")
            st.dataframe(rng, hide_index=True)
            if grid.empty:
                st.warning("Tidak ada kombinasi grid yang diprediksi Match.")
            else:
                st.markdown("Contoh kombinasi terbaik dari grid (+ produk katalog terdekat)")
                st.dataframe(fmt_tabel(grid), hide_index=True)
                if stg == 3:
                    swatch_html({"Bare Face": np.tile(bare, (len(grid), 1)),
                                 "Liquid (grid)": grid[["L_liq", "a_liq", "b_liq"]].values,
                                 "Powder (grid)": grid[["L_pow", "a_pow", "b_pow"]].values,
                                 "Prediksi akhir": grid[["pred_L", "pred_a", "pred_b"]].values},
                                [f"#{i + 1}<br>P={pp:.2f}" for i, pp in enumerate(grid.P_match)])
                else:
                    swatch_html({"Bare Face": np.tile(bare, (len(grid), 1)),
                                 "Produk (grid)": grid[core.KANAL].values,
                                 "Prediksi after": grid[["pred_L", "pred_a", "pred_b"]].values},
                                [f"#{i + 1}<br>P={pp:.2f}" for i, pp in enumerate(grid.P_match)])

            # ---------------- C. Simulasi produk tertentu ----------------
            with st.expander("Simulasi produk tertentu (prediksi warna after pemakaian)"):
                if stg in (1, 2):
                    kat_src = T["katalog_liquid"] if stg == 1 else kat_pow
                    nama = "liquid" if stg == 1 else "powder"
                    mode = st.radio("Sumber produk", ["Dari katalog", "Input manual"], horizontal=True, key=f"m{stg}")
                    if mode == "Dari katalog":
                        idx = st.selectbox(f"Baris {nama} di katalog", kat_src.index.tolist(), key=f"k{stg}")
                        prod = kat_src.loc[idx, core.KANAL].values.astype(float)
                    else:
                        cc = st.columns(3)
                        prod = np.array([cc[j].number_input(f"{k} {nama}", value=float(kat_src[k].median()),
                                                            key=f"in{stg}{k}") for j, k in enumerate(core.KANAL)])
                    pred = core.prediksi_lab(S[stg]["coef"], bare, prod)
                    p, dE, m = core.peluang_match(S[stg]["klas"], pred.reshape(1, 3), bare)
                    cc = st.columns(4)
                    cc[0].metric("Prediksi after L*a*b*", f"{pred[0]:.2f} · {pred[1]:.2f} · {pred[2]:.2f}")
                    cc[1].metric("dE ke bare face", f"{dE[0]:.2f}")
                    cc[2].metric("P(Match)", f"{p[0]:.3f}")
                    cc[3].metric("Status", "Match" if m[0] else "Unmatch")
                    if stg == 2:
                        kls = core.klasifikasi_powder(pd.DataFrame([prod], columns=core.KANAL), T["model_undertone"],
                                                      T["model_level"], metode_ut, level_per_ut)
                        st.caption(f"Klasifikasi powder: undertone **{kls['pred_Undertone'].iloc[0]}**, "
                                   f"level **{kls['pred_Level'].iloc[0]:g}**")
                    swatch_html({"Bare Face": bare.reshape(1, 3), nama.capitalize(): prod.reshape(1, 3),
                                 f"Prediksi after {nama}": pred.reshape(1, 3)}, ["Simulasi"])
                else:
                    cc = st.columns(2)
                    i_l = cc[0].selectbox("Baris liquid", T["katalog_liquid"].index.tolist(), key="k3l")
                    i_p = cc[1].selectbox("Baris powder", kat_pow.index.tolist(), key="k3p")
                    liq = T["katalog_liquid"].loc[i_l, core.KANAL].values.astype(float)
                    pw = kat_pow.loc[i_p, core.KANAL].values.astype(float)
                    after = core.prediksi_lab(S[1]["coef"], bare, liq)
                    final = core.prediksi_lab(S[3]["coef"], after, pw)
                    p, dE, m = core.peluang_match(S[3]["klas"], final.reshape(1, 3), bare)
                    cc = st.columns(4)
                    cc[0].metric("Prediksi after liquid", f"{after[0]:.2f} · {after[1]:.2f} · {after[2]:.2f}")
                    cc[1].metric("Prediksi akhir", f"{final[0]:.2f} · {final[1]:.2f} · {final[2]:.2f}")
                    cc[2].metric("dE · P(Match)", f"{dE[0]:.2f} · {p[0]:.3f}")
                    cc[3].metric("Status", "Match" if m[0] else "Unmatch")
                    st.caption(f"Powder baris {i_p}: undertone **{kat_pow.loc[i_p, 'pred_Undertone']}**, "
                               f"level **{kat_pow.loc[i_p, 'pred_Level']:g}** "
                               f"(label asli: {kat_pow.loc[i_p, 'label_Undertone']}, {kat_pow.loc[i_p, 'label_L']:g})")
                    swatch_html({"Bare Face": bare.reshape(1, 3), "Liquid": liq.reshape(1, 3),
                                 "Powder": pw.reshape(1, 3), "Prediksi after liquid": after.reshape(1, 3),
                                 "Prediksi akhir": final.reshape(1, 3)}, ["Simulasi"])

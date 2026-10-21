"""
Inti perhitungan Powder Shade Alignment.

Semua fungsi di sini disalin dari notebook `_UPDATE_Powder_Shade_Alignment_Gabungan.ipynb`
TANPA mengubah metodenya. Yang dibedakan hanya cara keluarannya: di notebook hasil dicetak
dengan print/display/plt.show, di sini hasilnya dikembalikan (return) supaya bisa ditampilkan
oleh Streamlit (app.py).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu
from skimage import color
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (r2_score, root_mean_squared_error, roc_auc_score, roc_curve,
                             accuracy_score, confusion_matrix, classification_report)

# =====================================================================================
# 0. Parameter (sama dengan notebook)
# =====================================================================================
KANAL = ["L", "a", "b"]

VERSI_REGRESI = {
    "berganda":            "hasil = c1*input1 + c2*input2 + c0",
    "sum":                 "hasil = c*(input1 + input2) + c0",
    "sum_tanpa_intercept": "hasil = c*(input1 + input2)",
}

UNDERTONES = ["Cool", "Neutral", "Warm"]
WARNA_UNDERTONE = {"Cool": "#3d5a80", "Neutral": "#8d8d8d", "Warm": "#e07a5f"}

SUMBER_WARNA = {"Powder": "{}_produk", "Liquid": "{}_liquid", "Bare Face": "{}_bare face"}
BATAS_HUE_MANUAL = [42, 52]
TOLERANSI = 1.0      # tebakan level dianggap "dekat" bila selisih level <= TOLERANSI


# =====================================================================================
# Load data
# =====================================================================================
def ke_biner(seri):
    # Ubah kolom Match/Unmatch (True/False, 1/0, "Match"/"Unmatch") menjadi 1/0
    peta = {"match": 1, "unmatch": 0, "true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    return seri.map(lambda v: np.nan if pd.isna(v) else peta.get(str(v).strip().lower(), np.nan))


def load_data(path_or_buffer, sheet_name="New_data"):
    df_raw = pd.read_excel(path_or_buffer, sheet_name=sheet_name)

    # Pastikan semua kolom L, a, b bertipe angka (kadang terbaca sebagai teks dari Excel)
    kolom_lab = [c for c in df_raw.columns if c.split("_")[0] in KANAL and "_" in c]
    for c in kolom_lab:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    kolom_match = [c for c in df_raw.columns if c.startswith("(Match/Unmatch)")]
    for c in kolom_match:
        df_raw[c] = ke_biner(df_raw[c])
    return df_raw


# =====================================================================================
# 1a. Regresi warna
# =====================================================================================
def tanda(x):
    # Format angka dengan tanda di depan, contoh 2.5 -> "+ 2.5000", -2.5 -> "- 2.5000"
    return f"{'+' if x >= 0 else '-'} {abs(x):.4f}"


def fit_regresi(X1, X2, Y, nama_input=("input1", "input2"), nama_target="hasil", versi="berganda"):
    # Regresi per kanal L, a, b. X1, X2, Y berbentuk (N, 3) dengan urutan kolom L, a, b.
    #   versi "berganda"            : hasil = c1*input1 + c2*input2 + c0
    #   versi "sum"                 : hasil = c*(input1 + input2) + c0
    #   versi "sum_tanpa_intercept" : hasil = c*(input1 + input2)
    # Semua versi disimpan dalam format yang sama: coef (3x3) = [c1, c2, c0] per kanal
    # (untuk versi sum, c1 = c2 = c), jadi fungsi lain tidak perlu tahu versinya.
    # Keluaran tambahan dibanding notebook: tabel persamaan (pengganti print).
    n1, n2 = nama_input
    coef = np.zeros((3, 3))
    persamaan = []
    for j, k in enumerate(KANAL):
        x1, x2, y = X1[:, j], X2[:, j], Y[:, j]
        if versi == "berganda":
            reg = LinearRegression().fit(np.column_stack([x1, x2]), y)
            c1, c2, c0 = reg.coef_[0], reg.coef_[1], reg.intercept_
            rumus = f"{c1:.4f}*{n1} {tanda(c2)}*{n2} {tanda(c0)}"
        elif versi in ("sum", "sum_tanpa_intercept"):
            pakai_c0 = versi == "sum"
            reg = LinearRegression(fit_intercept=pakai_c0).fit((x1 + x2).reshape(-1, 1), y)
            c1 = c2 = reg.coef_[0]
            c0 = reg.intercept_ if pakai_c0 else 0.0
            rumus = f"{c1:.4f}*({n1} + {n2})" + (f" {tanda(c0)}" if pakai_c0 else "")
        else:
            raise ValueError(f"Versi tidak dikenal: {versi}. Pilihan: {list(VERSI_REGRESI)}")
        coef[j] = [c1, c2, c0]
        pred_k = c1 * x1 + c2 * x2 + c0
        persamaan.append({"Kanal": k, "Persamaan": f"{k}_{nama_target} = {rumus}",
                          "R²": r2_score(y, pred_k), "RMSE": root_mean_squared_error(y, pred_k)})
    return coef, prediksi_lab(coef, X1, X2), pd.DataFrame(persamaan)


def prediksi_lab(coef, input1, input2):
    # Hitung prediksi L,a,b dari koefisien regresi. input1/input2 berbentuk (..., 3).
    return coef[:, 0] * input1 + coef[:, 1] * input2 + coef[:, 2]


def ambil_lab(data, fmt):
    # Ambil 3 kolom L,a,b sebagai array (N, 3). Contoh fmt: "{}_bare face".
    return data[[fmt.format(k) for k in KANAL]].values.astype(float)


# =====================================================================================
# 1b. Fitur delta & klasifikasi Match/Unmatch
# =====================================================================================
def fitur_match(pred, bare):
    # Fitur untuk classifier: dL, da, db, dE (prediksi dikurangi bare face)
    d = pred - bare
    dE = np.sqrt((d ** 2).sum(axis=1))
    return np.column_stack([d, dE])


def tabel_fitur_delta(pred, bare):
    # Semua fitur delta turunan (dL, da, db, d(a-b), dChroma, dHue, dE, ...) dalam satu tabel
    def turunan(L, a, b):
        return {"L": L, "a": a, "b": b, "a-b": a - b, "2a-b": 2 * a - b, "2b-a": 2 * b - a,
                "b/a": b / a, "Chroma": np.hypot(a, b), "Hue": np.degrees(np.arctan2(b, a)) % 360}
    p, s = turunan(*pred.T), turunan(*bare.T)
    hasil = pd.DataFrame({f"d{n}": p[n] - s[n] for n in p})
    hasil["dE"] = np.sqrt(hasil["dL"] ** 2 + hasil["da"] ** 2 + hasil["db"] ** 2)
    return hasil


def uji_fitur(tabel, y):
    # Uji tiap fitur delta satu per satu: AUC, akurasi, batas (cut-off), dan p-value Mann-Whitney
    y = np.asarray(y).astype(int)
    baris = []
    for col in tabel.columns:
        x = tabel[col].values.astype(float)
        auc = roc_auc_score(y, x)
        arah = ">="
        if auc < 0.5:                       # kalau nilai kecil = Match, balik arahnya
            x, auc, arah = -x, 1 - auc, "<="
        fpr, tpr, thr = roc_curve(y, x)
        ok = np.isfinite(thr)
        batas = thr[ok][np.argmax(tpr[ok] - fpr[ok])]
        acc = accuracy_score(y, (x >= batas).astype(int))
        cutoff = batas if arah == ">=" else -batas
        p_val = mannwhitneyu(x[y == 1], x[y == 0]).pvalue
        baris.append({"Fitur": col, "AUC": auc, "Akurasi": acc,
                      "Aturan Match": f"{col} {arah} {cutoff:.3f}",
                      "p-value": p_val, "Signifikan (p<0.05)": "Ya" if p_val < 0.05 else "-"})
    return pd.DataFrame(baris).sort_values("AUC", ascending=False).reset_index(drop=True)


def fit_klasifikasi(X, y):
    # Logistic Regression + threshold optimal (Youden's J), tanpa print/plot
    y = np.asarray(y).astype(int)
    model = LogisticRegression(class_weight="balanced", random_state=42).fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    fpr, tpr, thr = roc_curve(y, probs)
    ok = np.isfinite(thr) & (thr <= 1)
    threshold = thr[ok][np.argmax(tpr[ok] - fpr[ok])]
    pred = (probs >= threshold).astype(int)
    return {"model": model, "threshold": threshold, "auc": auc,
            "acc": accuracy_score(y, pred), "pred": pred}


def evaluate_match_model(X, y):
    # Sama seperti notebook: fit_klasifikasi + laporan (dikembalikan, bukan dicetak)
    y = np.asarray(y).astype(int)
    hasil = fit_klasifikasi(X, y)
    probs = hasil["model"].predict_proba(X)[:, 1]
    hasil["y"] = y
    hasil["probs"] = probs
    hasil["cm"] = confusion_matrix(y, hasil["pred"])
    hasil["report"] = pd.DataFrame(classification_report(
        y, hasil["pred"], target_names=["Unmatch", "Match"], zero_division=0, output_dict=True)).T
    return hasil


def persamaan_logistik(klas):
    # z = c0 + w1*dL + w2*da + w3*db + w4*dE ; P(Match) = 1/(1+e^-z)
    m = klas["model"]
    w, c0 = m.coef_[0], m.intercept_[0]
    fitur = ["dL", "da", "db", "dE"]
    t = klas["threshold"]
    z_cut = np.log(t / (1 - t))
    suku = " ".join(f"{tanda(wi)}·{f}" for wi, f in zip(w, fitur))
    return {"z": f"z = {c0:.4f} {suku}",
            "p": "P(Match) = 1 / (1 + e^(-z))",
            "aturan": f"Match bila P(Match) ≥ {t:.4f}  ⇔  z ≥ {z_cut:.4f}",
            "koef": pd.DataFrame({"Suku": ["Intercept"] + fitur, "Koefisien": [c0, *w]})}


def bandingkan_versi(X1, X2, Y, bare, y, nama_input, nama_target):
    # Jalankan semua versi regresi -> persamaan + tabel R², RMSE, AUC, akurasi.
    # X1 boleh berupa dict {versi: array} kalau input1 tergantung versi (dipakai di Stage 3).
    baris, semua_coef, semua_persamaan = [], {}, {}
    for versi, rumus in VERSI_REGRESI.items():
        x1 = X1[versi] if isinstance(X1, dict) else X1
        coef, pred, pers = fit_regresi(x1, X2, Y, nama_input, nama_target, versi)
        klas = fit_klasifikasi(fitur_match(pred, bare), y)
        semua_coef[versi] = coef
        semua_persamaan[versi] = pers
        r2 = [r2_score(Y[:, j], pred[:, j]) for j in range(3)]
        rmse = [root_mean_squared_error(Y[:, j], pred[:, j]) for j in range(3)]
        baris.append({"Versi": versi, "Rumus": rumus,
                      "R² L": r2[0], "R² a": r2[1], "R² b": r2[2], "R² rata-rata": np.mean(r2),
                      "RMSE rata-rata": np.mean(rmse),
                      "AUC Match/Unmatch": klas["auc"], "Akurasi Match/Unmatch": klas["acc"]})
    return pd.DataFrame(baris), semua_coef, semua_persamaan


def peluang_match(klasifikasi, pred, bare):
    # P(Match), dE, dan status Match untuk banyak prediksi sekaligus
    X = fitur_match(pred, bare)
    p = klasifikasi["model"].predict_proba(X)[:, 1]
    return p, X[:, 3], p >= klasifikasi["threshold"]


# =====================================================================================
# 1c. Visualisasi (mengembalikan figure)
# =====================================================================================
def lab_to_rgb(lab):
    lab = np.asarray(lab, dtype=float).reshape(-1, 1, 3)
    return np.clip(color.lab2rgb(lab).reshape(-1, 3), 0, 1)


def lab_to_hex(lab):
    return ["#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb) for rgb in lab_to_rgb(lab)]


def plot_aktual_vs_prediksi(aktual, pred, nama):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    idx = np.arange(len(aktual))
    for j, k in enumerate(KANAL):
        axes[j].scatter(idx, aktual[:, j], label="Aktual", color="royalblue", s=40, alpha=0.8)
        axes[j].scatter(idx, pred[:, j], label="Prediksi", color="darkorange", marker="x", s=40)
        axes[j].set_title(f"{k} {nama}", fontweight="bold"); axes[j].set_xlabel("Index data")
        axes[j].grid(True, linestyle="--", alpha=0.5); axes[j].legend()
    plt.tight_layout()
    return fig


def plot_boxplot_fitur(tabel, y, judul):
    data = tabel.copy()
    data["Status"] = np.where(np.asarray(y).astype(int) == 1, "Match", "Unmatch")
    n_cols = 4
    n_rows = int(np.ceil(len(tabel.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.6 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(tabel.columns):
        sns.boxplot(data=data, x="Status", y=col, hue="Status", order=["Unmatch", "Match"],
                    palette={"Unmatch": "#e41a1c", "Match": "#377eb8"}, width=0.5, legend=False, ax=axes[i])
        p_val = mannwhitneyu(data.loc[data.Status == "Match", col], data.loc[data.Status == "Unmatch", col]).pvalue
        axes[i].set_title(f"{col}\np = {p_val:.4f}", fontweight="bold", fontsize=10,
                          color="darkred" if p_val < 0.05 else "black")
        axes[i].set_xlabel(""); axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    for j in range(len(tabel.columns), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(judul, fontweight="bold", fontsize=13)
    plt.tight_layout()
    return fig


def plot_evaluasi(klas, nama):
    # Confusion matrix + aktual vs prediksi (sama seperti evaluate_match_model di notebook)
    y, pred = klas["y"], klas["pred"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    sns.heatmap(klas["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Unmatch", "Match"], yticklabels=["Unmatch", "Match"], ax=axes[0])
    axes[0].set_xlabel("Prediksi"); axes[0].set_ylabel("Aktual"); axes[0].set_title(f"Confusion Matrix: {nama}")
    idx = np.arange(len(y))
    axes[1].scatter(idx, y, color="royalblue", label="Aktual", s=50)
    axes[1].scatter(idx, pred, color="red", marker="x", label="Prediksi", s=50)
    axes[1].set_yticks([0, 1], ["Unmatch", "Match"]); axes[1].set_xlabel("Index Sampel")
    axes[1].set_title(f"Aktual vs Prediksi: {nama}"); axes[1].legend()
    plt.tight_layout()
    return fig


# =====================================================================================
# 1d. Rekomendasi produk
# =====================================================================================
def buat_katalog(data, fmt, kolom_info=None):
    # Daftar produk unik (berdasarkan L, a, b) dari data. Index = nomor baris di data asli.
    kolom = [fmt.format(k) for k in KANAL]
    info = [c for c in (kolom_info or []) if c in data.columns]
    katalog = data[kolom + info].dropna(subset=kolom).drop_duplicates(subset=kolom)
    # kolom info diberi awalan "label_" supaya tidak bentrok dengan kanal L/a/b (misal kolom level "L")
    katalog = katalog.rename(columns={**dict(zip(kolom, KANAL)), **{c: f"label_{c}" for c in info}})
    katalog.index.name = "baris_data"
    return katalog


def buat_grid(katalog, n_grid):
    # Kombinasi L, a, b di dalam rentang min-max katalog (n_grid titik per kanal)
    sumbu = [np.linspace(katalog[k].min(), katalog[k].max(), n_grid) for k in KANAL]
    mesh = np.meshgrid(*sumbu, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def urutkan(tabel, urut):
    if urut == "dE":
        return tabel.sort_values(["Match", "dE"], ascending=[False, True])
    return tabel.sort_values(["P_match", "dE"], ascending=[False, True])


def produk_terdekat(titik, katalog):
    # Untuk tiap titik grid: produk katalog terdekat (jarak Euclidean di ruang LAB)
    jarak = np.linalg.norm(titik[:, None, :] - katalog[KANAL].values[None, :, :], axis=2)
    return katalog.index.values[jarak.argmin(axis=1)], jarak.min(axis=1)


def cek_rentang_bare(bare, data):
    pesan = []
    for j, k in enumerate(KANAL):
        lo, hi = data[f"{k}_bare face"].min(), data[f"{k}_bare face"].max()
        if not lo <= bare[j] <= hi:
            pesan.append(f"{k} bare face = {bare[j]:.2f} di luar rentang data latih ({lo:.1f} - {hi:.1f}) -> hasil ekstrapolasi")
    return pesan


def teks_range(nilai):
    return f"{nilai.min():.2f} - {nilai.max():.2f}" if len(nilai) else "tidak ada"


def tabel_range(titik_match, titik_terbaik, katalog, nama):
    baris = []
    for j, k in enumerate(KANAL):
        baris.append({"Produk": nama, "Kanal": k,
                      "Range katalog": teks_range(katalog[k].values),
                      "Range yang Match": teks_range(titik_match[:, j]),
                      "Range terbaik (top 10%)": teks_range(titik_terbaik[:, j])})
    return pd.DataFrame(baris)


def ambil_top_persen(tabel, persen=10):
    # Ambil sekian persen baris teratas (tabel sudah diurutkan), minimal 1 baris
    return tabel.head(max(1, int(len(tabel) * persen / 100)))


def rekomendasi_produk(bare_face, katalog, coef, klasifikasi, nama_produk, data_latih,
                       top_k=5, n_grid=15, urut="P_match"):
    bare = np.array([bare_face[k] for k in KANAL], dtype=float)
    peringatan = cek_rentang_bare(bare, data_latih)

    # A. Dari katalog ----------
    X = katalog[KANAL].values
    pred = prediksi_lab(coef, bare, X)
    p, dE, match = peluang_match(klasifikasi, pred, bare)
    hasil_katalog = katalog.copy()
    hasil_katalog[["pred_L", "pred_a", "pred_b"]] = pred
    hasil_katalog["dE"], hasil_katalog["P_match"], hasil_katalog["Match"] = dE, p, match
    hasil_katalog = urutkan(hasil_katalog, urut)

    # B. Dari range katalog (grid) ----------
    G = buat_grid(katalog, n_grid)
    pred_g = prediksi_lab(coef, bare, G)
    p_g, dE_g, match_g = peluang_match(klasifikasi, pred_g, bare)
    hasil_grid = pd.DataFrame(G, columns=KANAL)
    hasil_grid[["pred_L", "pred_a", "pred_b"]] = pred_g
    hasil_grid["dE"], hasil_grid["P_match"], hasil_grid["Match"] = dE_g, p_g, match_g
    hasil_grid = urutkan(hasil_grid[hasil_grid.Match], urut)

    hasil_range = tabel_range(G[match_g], ambil_top_persen(hasil_grid)[KANAL].values, katalog, nama_produk)

    hasil_grid = hasil_grid.head(top_k).reset_index(drop=True)
    hasil_grid["katalog_terdekat"], hasil_grid["jarak_ke_katalog"] = produk_terdekat(hasil_grid[KANAL].values, katalog)

    info = {"n_katalog": len(katalog), "n_match_katalog": int(match.sum()),
            "n_grid": len(G), "n_match_grid": int(match_g.sum()), "peringatan": peringatan}
    return hasil_katalog, hasil_range, hasil_grid, info


def rekomendasi_gabungan(bare_face, katalog_liq, katalog_pow, coef_s1, coef_s3, klasifikasi, data_latih,
                         top_k=5, n_grid=8, urut="P_match"):
    bare = np.array([bare_face[k] for k in KANAL], dtype=float)
    peringatan = cek_rentang_bare(bare, data_latih)

    def hitung(LQ, PW):
        # semua pasangan liquid x powder -> after liquid -> hasil akhir
        i_liq = np.repeat(np.arange(len(LQ)), len(PW))
        i_pow = np.tile(np.arange(len(PW)), len(LQ))
        after = prediksi_lab(coef_s1, bare, LQ[i_liq])
        final = prediksi_lab(coef_s3, after, PW[i_pow])
        p, dE, match = peluang_match(klasifikasi, final, bare)
        return i_liq, i_pow, after, final, p, dE, match

    # ---------- A. Dari katalog (semua pasangan liquid x powder) ----------
    LQ, PW = katalog_liq[KANAL].values, katalog_pow[KANAL].values
    i_liq, i_pow, after, final, p, dE, match = hitung(LQ, PW)
    hasil_katalog = pd.DataFrame({
        "baris_liquid": katalog_liq.index.values[i_liq], "baris_powder": katalog_pow.index.values[i_pow],
        "L_liq": LQ[i_liq, 0], "a_liq": LQ[i_liq, 1], "b_liq": LQ[i_liq, 2],
        "L_pow": PW[i_pow, 0], "a_pow": PW[i_pow, 1], "b_pow": PW[i_pow, 2],
        "after_liq_L": after[:, 0], "after_liq_a": after[:, 1], "after_liq_b": after[:, 2],
        "pred_L": final[:, 0], "pred_a": final[:, 1], "pred_b": final[:, 2],
        "dE": dE, "P_match": p, "Match": match})
    for c in katalog_pow.columns:
        if c.startswith("label_") or c.startswith("pred_"):
            hasil_katalog[f"{c}_powder"] = katalog_pow[c].values[i_pow]
    hasil_katalog = urutkan(hasil_katalog, urut)

    # ---------- B. Dari range katalog (grid liquid x grid powder) ----------
    GL, GP = buat_grid(katalog_liq, n_grid), buat_grid(katalog_pow, n_grid)
    i_liq_g, i_pow_g, after_g, final_g, p_g, dE_g, match_g = hitung(GL, GP)
    hasil_grid = pd.DataFrame({
        "L_liq": GL[i_liq_g, 0], "a_liq": GL[i_liq_g, 1], "b_liq": GL[i_liq_g, 2],
        "L_pow": GP[i_pow_g, 0], "a_pow": GP[i_pow_g, 1], "b_pow": GP[i_pow_g, 2],
        "after_liq_L": after_g[:, 0], "after_liq_a": after_g[:, 1], "after_liq_b": after_g[:, 2],
        "pred_L": final_g[:, 0], "pred_a": final_g[:, 1], "pred_b": final_g[:, 2],
        "dE": dE_g, "P_match": p_g, "Match": match_g})
    hasil_grid = urutkan(hasil_grid[hasil_grid.Match], urut)

    terbaik = ambil_top_persen(hasil_grid)
    hasil_range = pd.concat([
        tabel_range(GL[i_liq_g[match_g]], terbaik[["L_liq", "a_liq", "b_liq"]].values, katalog_liq, "Liquid"),
        tabel_range(GP[i_pow_g[match_g]], terbaik[["L_pow", "a_pow", "b_pow"]].values, katalog_pow, "Powder")],
        ignore_index=True)

    hasil_grid = hasil_grid.head(top_k).reset_index(drop=True)
    hasil_grid["liquid_terdekat"], hasil_grid["jarak_liquid"] = produk_terdekat(
        hasil_grid[["L_liq", "a_liq", "b_liq"]].values, katalog_liq)
    hasil_grid["powder_terdekat"], hasil_grid["jarak_powder"] = produk_terdekat(
        hasil_grid[["L_pow", "a_pow", "b_pow"]].values, katalog_pow)

    info = {"n_katalog": len(match), "n_match_katalog": int(match.sum()),
            "n_grid": len(match_g), "n_grid_liq": len(GL), "n_grid_pow": len(GP),
            "n_match_grid": int(match_g.sum()), "peringatan": peringatan}
    return hasil_katalog, hasil_range, hasil_grid, info


# =====================================================================================
# 6. Klasifikasi Undertone (Cool / Neutral / Warm)
# =====================================================================================
def hitung_hue(a, b):
    # Hue dalam derajat (0-360)
    return np.degrees(np.arctan2(b, a)) % 360


def cari_batas(x, y):
    # Decision tree 1 fitur: jumlah daun = jumlah kelas -> batas-batas nilai x
    tree = DecisionTreeClassifier(max_leaf_nodes=len(np.unique(y)), random_state=42)
    tree.fit(x.reshape(-1, 1), y)
    batas = sorted(t for t in tree.tree_.threshold if t != -2)
    return tree, batas, accuracy_score(y, tree.predict(x.reshape(-1, 1)))


def undertone_hue_manual(hue, batas=BATAS_HUE_MANUAL):
    return np.where(hue < batas[0], "Cool", np.where(hue <= batas[1], "Neutral", "Warm"))


def cari_garis_terbaik(a, b, y, langkah=0.5):
    # Coba semua sudut θ (derajat), skor = cos(θ)*a + sin(θ)*b, pilih θ dengan akurasi tertinggi
    terbaik = {"acc": -1}
    for theta in np.arange(0, 180, langkah):
        w_a, w_b = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        skor = w_a * a + w_b * b
        tree, batas, acc = cari_batas(skor, y)
        if acc > terbaik["acc"]:
            terbaik = {"theta": theta, "w_a": w_a, "w_b": w_b, "tree": tree, "batas": batas, "acc": acc}
    return terbaik


def aturan_wilayah(tree, batas, nama, satuan=""):
    # Ubah batas decision tree menjadi aturan teks, contoh: "skor <= -2.03 -> Warm"
    tepi = [-np.inf] + list(batas) + [np.inf]
    aturan = []
    for lo, hi in zip(tepi[:-1], tepi[1:]):
        titik = hi - 1 if lo == -np.inf else (lo + 1 if hi == np.inf else (lo + hi) / 2)
        kelas = tree.predict([[titik]])[0]
        if lo == -np.inf:
            teks = f"{nama} <= {hi:.2f}{satuan}"
        elif hi == np.inf:
            teks = f"{nama} > {lo:.2f}{satuan}"
        else:
            teks = f"{lo:.2f}{satuan} < {nama} <= {hi:.2f}{satuan}"
        aturan.append(f"{teks}  ->  {kelas}")
    return aturan


def garis_hue(sudut):
    return f"Hue = {sudut:.2f}°  <=>  b = {np.tan(np.radians(sudut)):.4f}·a"


def garis_skor(w_a, w_b, t):
    kiri = f"{w_a:.4f}·a {'+' if w_b >= 0 else '-'} {abs(w_b):.4f}·b = {t:.2f}"
    if abs(w_b) < 1e-9:
        return f"{kiri}  <=>  a = {t / w_a:.4f}"
    m, c = -w_a / w_b, t / w_b
    return f"{kiri}  <=>  b = {m:.4f}·a {'+' if c >= 0 else '-'} {abs(c):.4f}"


def gambar_sinar(ax, sudut_list, r):
    for s in sudut_list:
        ax.plot([0, r * np.cos(np.radians(s))], [0, r * np.sin(np.radians(s))], "k--", lw=1.8,
                label=f"b = {np.tan(np.radians(s)):.3f}·a  ({s:.1f}°)")


def gambar_garis(ax, w_a, w_b, batas_list, a_max):
    for t in batas_list:
        if abs(w_b) > 1e-9:
            a_vals = np.linspace(0, a_max, 100)
            ax.plot(a_vals, (t - w_a * a_vals) / w_b, "k--", lw=1.8,
                    label=f"b = {-w_a / w_b:.3f}·a {'+' if t / w_b >= 0 else '-'} {abs(t / w_b):.2f}")
        else:
            ax.axvline(t / w_a, color="k", ls="--", lw=1.8, label=f"a = {t / w_a:.2f}")


def latih_undertone(df_raw):
    # Sama dengan notebook bagian 6: 3 metode x 3 sumber warna. Hasil disimpan per sumber.
    hasil, model = [], {}
    for nama_sumber, fmt in SUMBER_WARNA.items():
        ka, kb = fmt.format("a"), fmt.format("b")
        d = df_raw[df_raw["Undertone"].isin(UNDERTONES)].dropna(subset=[ka, kb])
        if len(d) < 3 or d["Undertone"].nunique() < 2:
            continue
        a, b, y = d[ka].values, d[kb].values, d["Undertone"].values
        hue = hitung_hue(a, b)
        acc_manual = accuracy_score(y, undertone_hue_manual(hue))
        tree_hue, batas_hue, acc_hue = cari_batas(hue, y)
        g = cari_garis_terbaik(a, b, y)
        model[nama_sumber] = {"a": a, "b": b, "y": y, "acc_manual": acc_manual,
                              "tree_hue": tree_hue, "batas_hue": batas_hue, "acc_hue": acc_hue, "garis": g,
                              "aturan_hue": aturan_wilayah(tree_hue, batas_hue, "Hue", "°"),
                              "aturan_skor": aturan_wilayah(g["tree"], g["batas"], "skor")}
        hasil.append({
            "Sumber": nama_sumber, "N": len(d),
            "Akurasi Hue manual": acc_manual,
            "Akurasi Hue otomatis": acc_hue, "Batas Hue otomatis (°)": np.round(batas_hue, 2),
            "Akurasi Garis a-b": g["acc"], "θ (°)": g["theta"],
            "Persamaan skor": f"{g['w_a']:.4f}·a {'+' if g['w_b'] >= 0 else '-'} {abs(g['w_b']):.4f}·b",
            "Batas skor": np.round(g["batas"], 2)})
    return pd.DataFrame(hasil), model


def plot_undertone(m, nama_sumber):
    a, b, y, g = m["a"], m["b"], m["y"], m["garis"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    a_max, b_max = a.max() * 1.1, b.max() * 1.1
    r = np.hypot(a, b).max() * 1.2
    judul = [f"[1] Hue manual {BATAS_HUE_MANUAL}° | Akurasi {m['acc_manual']*100:.1f}%",
             f"[2] Hue otomatis {np.round(m['batas_hue'], 1).tolist()}° | Akurasi {m['acc_hue']*100:.1f}%",
             f"[3] skor = {g['w_a']:.3f}·a {'+' if g['w_b'] >= 0 else '-'} {abs(g['w_b']):.3f}·b"
             f"  (θ={g['theta']}°)\nAkurasi {g['acc']*100:.1f}%"]
    for i, ax in enumerate(axes):
        for ut in UNDERTONES:
            mm = y == ut
            ax.scatter(a[mm], b[mm], c=WARNA_UNDERTONE[ut], label=ut, edgecolors="white", s=70, alpha=0.85)
        if i == 0:
            gambar_sinar(ax, BATAS_HUE_MANUAL, r)
        elif i == 1:
            gambar_sinar(ax, m["batas_hue"], r)
        else:
            gambar_garis(ax, g["w_a"], g["w_b"], g["batas"], a_max)
        ax.set_xlim(0, a_max); ax.set_ylim(0, b_max); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"a {nama_sumber}"); ax.set_ylabel(f"b {nama_sumber}")
        ax.set_title(judul[i], fontsize=10); ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle(f"Undertone dari warna {nama_sumber}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


def prediksi_undertone(model_ut, a, b, metode):
    # metode: "theta" (garis a-b terbaik) atau "hue" (Hue otomatis / optimasi decision tree)
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    if metode == "theta":
        g = model_ut["garis"]
        skor = g["w_a"] * a + g["w_b"] * b
        return g["tree"].predict(skor.reshape(-1, 1))
    hue = hitung_hue(a, b)
    return model_ut["tree_hue"].predict(hue.reshape(-1, 1))


# =====================================================================================
# 7. Klasifikasi Level L (metode median L)
# =====================================================================================
def fit_median(x, level):
    # Median nilai L tiap level -> level diurutkan dari median L terkecil -> batas = titik tengah antar median
    x, level = np.asarray(x, float), np.asarray(level, float)
    levels = np.sort(np.unique(level))
    med = np.array([np.median(x[level == lv]) for lv in levels])
    urut = np.argsort(med, kind="mergesort")
    levels_x, med_x = levels[urut], med[urut]
    batas = (med_x[:-1] + med_x[1:]) / 2
    monoton = bool(np.all(np.diff(med) < 0) or np.all(np.diff(med) > 0))
    return {"levels": levels_x, "median": med_x, "batas": batas, "monoton": monoton}


def predict_median(x, model):
    # Cari posisi x di antara batas -> level yang rentangnya memuat x
    idx = np.searchsorted(model["batas"], np.asarray(x, float), side="right")
    return model["levels"][idx]


def tabel_aturan_median(model):
    tepi = np.r_[-np.inf, model["batas"], np.inf]
    baris = []
    for k, lv in enumerate(model["levels"]):
        lo = "-inf" if np.isinf(tepi[k]) else f"{tepi[k]:.2f}"
        hi = "+inf" if np.isinf(tepi[k + 1]) else f"{tepi[k + 1]:.2f}"
        baris.append({"Level": f"{lv:g}", "Median L": round(float(model["median"][k]), 2),
                      "Aturan": f"{lo} <= L < {hi}"})
    return pd.DataFrame(baris)


def plot_level_L(ax, x, level, judul, label_x, semua_level, warna_level):
    x, level = np.asarray(x, float), np.asarray(level, float)
    model = fit_median(x, level)
    pred = predict_median(x, model)
    acc = (pred == level).mean()
    acc_dekat = (np.abs(pred - level) <= TOLERANSI).mean()

    ax.scatter(x, level, c=[warna_level[lv] for lv in level], edgecolors="white", s=70, zorder=4)
    ax.scatter(model["median"], model["levels"], marker="|", color="black", s=400, linewidths=2.5,
               zorder=6, label="Median L")
    xlim = (x.min() - 1, x.max() + 1)
    tepi = [xlim[0]] + list(model["batas"]) + [xlim[1]]
    for k, lv in enumerate(model["levels"]):
        ax.axvspan(max(tepi[k], xlim[0]), min(tepi[k + 1], xlim[1]), color=warna_level[lv], alpha=0.15, zorder=1)
    for t in model["batas"]:
        ax.axvline(t, color="black", ls="--", lw=1.3, zorder=5)
    ax.set_xlim(xlim)
    ax.set_yticks(semua_level)
    ax.set_xlabel(label_x); ax.set_ylabel("Level L")
    ax.set_title(f"{judul}\nAkurasi {acc*100:.1f}% | ±{TOLERANSI:g} level {acc_dekat*100:.1f}%",
                 fontweight="bold", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6, zorder=2)
    ax.legend(handles=[mpatches.Patch(color=warna_level[lv], alpha=0.35, label=f"Area Lvl {lv:g}")
                       for lv in sorted(model["levels"])],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
    return model, acc, acc_dekat


def latih_level(df_raw):
    # Sama dengan notebook bagian 7.1 & 7.2: model median per (sumber, undertone)
    semua_level = sorted(df_raw["L"].dropna().unique())
    warna_level = {lv: plt.colormaps["tab10"](i % 10) for i, lv in enumerate(semua_level)}
    hasil_level, model_level, figs = [], {}, {}

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (nama_sumber, fmt) in zip(axes, SUMBER_WARNA.items()):
        kol = fmt.format("L")
        d = df_raw.dropna(subset=[kol, "L"])
        model, acc, acc_dekat = plot_level_L(ax, d[kol].values, d["L"].values, nama_sumber, kol,
                                             semua_level, warna_level)
        model_level[(nama_sumber, "Semua")] = model
        hasil_level.append({"Sumber": nama_sumber, "Undertone": "Semua", "N": len(d),
                            "Akurasi": acc, f"Akurasi ±{TOLERANSI:g}": acc_dekat,
                            "Median L": np.round(model["median"], 2), "Batas L": np.round(model["batas"], 2)})
    plt.tight_layout()
    figs["Semua"] = fig

    for nama_sumber, fmt in SUMBER_WARNA.items():
        kol = fmt.format("L")
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True)
        for ax, ut in zip(axes, UNDERTONES):
            d = df_raw[df_raw["Undertone"] == ut].dropna(subset=[kol, "L"])
            if len(d) < 2:
                ax.set_title(f"{ut}: data kurang"); continue
            model, acc, acc_dekat = plot_level_L(ax, d[kol].values, d["L"].values, f"{nama_sumber} — {ut}", kol,
                                                 semua_level, warna_level)
            model_level[(nama_sumber, ut)] = model
            hasil_level.append({"Sumber": nama_sumber, "Undertone": ut, "N": len(d),
                                "Akurasi": acc, f"Akurasi ±{TOLERANSI:g}": acc_dekat,
                                "Median L": np.round(model["median"], 2), "Batas L": np.round(model["batas"], 2)})
        fig.suptitle(f"Level L dari {kol}, per undertone (metode median)", fontweight="bold", fontsize=14)
        plt.tight_layout()
        figs[nama_sumber] = fig
    return pd.DataFrame(hasil_level), model_level, figs


def prediksi_level_L(model_level, L, sumber="Bare Face", undertone=None):
    # Sama dengan notebook 7.4. Kalau undertone diisi -> batas milik undertone tsb, kalau tidak -> semua data.
    L = np.atleast_1d(np.asarray(L, float))
    if undertone is None:
        return predict_median(L, model_level[(sumber, "Semua")])
    undertone = np.broadcast_to(np.asarray(undertone), L.shape)
    out = np.full(len(L), np.nan)
    for ut in np.unique(undertone):
        m = undertone == ut
        kunci = (sumber, ut) if (sumber, ut) in model_level else (sumber, "Semua")
        out[m] = predict_median(L[m], model_level[kunci])
    return out


# =====================================================================================
# Training lengkap (Stage 1, 2, 3 + undertone + level)
# =====================================================================================
def latih_stage(df_raw, versi):
    S = {}

    # ---------- Stage 1 — Bare Face + Liquid ----------
    df_s1 = df_raw[['L_liquid', 'a_liquid', 'b_liquid', 'L_bare face', 'a_bare face', 'b_bare face',
                    'L_after liquid', 'a_after liquid', 'b_after liquid', '(Match/Unmatch)_liquid']].dropna().copy()
    bare_s1 = ambil_lab(df_s1, "{}_bare face")
    produk_s1 = ambil_lab(df_s1, "{}_liquid")
    after_s1 = ambil_lab(df_s1, "{}_after liquid")
    y_s1 = df_s1["(Match/Unmatch)_liquid"].astype(int).values
    banding_s1, coef_versi_s1, pers_versi_s1 = bandingkan_versi(
        bare_s1, produk_s1, after_s1, bare_s1, y_s1, ('bare_face', 'liquid'), "after_liquid")
    coef_s1, pred_s1, pers_s1 = fit_regresi(bare_s1, produk_s1, after_s1, ('bare_face', 'liquid'), "after_liquid", versi)
    delta_s1 = tabel_fitur_delta(pred_s1, bare_s1)
    klas_s1 = evaluate_match_model(fitur_match(pred_s1, bare_s1), y_s1)
    S[1] = dict(nama="Stage 1 — Bare Face + Liquid", data=df_s1, bare=bare_s1, y=y_s1,
                aktual=after_s1, pred=pred_s1, coef=coef_s1, persamaan=pers_s1,
                banding=banding_s1, persamaan_versi=pers_versi_s1, delta=delta_s1,
                uji=uji_fitur(delta_s1, y_s1), klas=klas_s1, label_target="After Liquid",
                swatch={"Liquid": produk_s1, "Bare Face": bare_s1, "After Liquid": after_s1,
                        "Prediksi After Liquid": pred_s1})

    # ---------- Stage 2 — Bare Face + Powder ----------
    df_s2 = df_raw[['L_produk', 'a_produk', 'b_produk', 'L_bare face', 'a_bare face', 'b_bare face',
                    'L_after powder', 'a_after powder', 'b_after powder', '(Match/Unmatch)_powder']].dropna().copy()
    bare_s2 = ambil_lab(df_s2, "{}_bare face")
    produk_s2 = ambil_lab(df_s2, "{}_produk")
    after_s2 = ambil_lab(df_s2, "{}_after powder")
    y_s2 = df_s2["(Match/Unmatch)_powder"].astype(int).values
    banding_s2, coef_versi_s2, pers_versi_s2 = bandingkan_versi(
        bare_s2, produk_s2, after_s2, bare_s2, y_s2, ('bare_face', 'powder'), "after_powder")
    coef_s2, pred_s2, pers_s2 = fit_regresi(bare_s2, produk_s2, after_s2, ('bare_face', 'powder'), "after_powder", versi)
    delta_s2 = tabel_fitur_delta(pred_s2, bare_s2)
    klas_s2 = evaluate_match_model(fitur_match(pred_s2, bare_s2), y_s2)
    S[2] = dict(nama="Stage 2 — Bare Face + Powder", data=df_s2, bare=bare_s2, y=y_s2,
                aktual=after_s2, pred=pred_s2, coef=coef_s2, persamaan=pers_s2,
                banding=banding_s2, persamaan_versi=pers_versi_s2, delta=delta_s2,
                uji=uji_fitur(delta_s2, y_s2), klas=klas_s2, label_target="After Powder",
                swatch={"Powder": produk_s2, "Bare Face": bare_s2, "After Powder": after_s2,
                        "Prediksi After Powder": pred_s2})

    # ---------- Stage 3 — Gabungan Liquid + Powder ----------
    kolom_s3 = ["L_liquid", "a_liquid", "b_liquid", "L_produk", "a_produk", "b_produk",
                "L_bare face", "a_bare face", "b_bare face", 'L_after liquid', 'a_after liquid', 'b_after liquid',
                "L_after liquid_powder", "a_after liquid_powder", "b_after liquid_powder",
                "(Match/Unmatch)_liquid_powder"]
    df_s3 = df_raw[kolom_s3].dropna().copy()
    bare_s3 = ambil_lab(df_s3, "{}_bare face")
    y_s3 = df_s3["(Match/Unmatch)_liquid_powder"].astype(int).values
    liquid_s3 = ambil_lab(df_s3, "{}_liquid")
    powder_s3 = ambil_lab(df_s3, "{}_produk")
    final_s3 = ambil_lab(df_s3, "{}_after liquid_powder")
    after_liq_aktual = ambil_lab(df_s3, "{}_after liquid")

    after_liq_per_versi = {v: prediksi_lab(coef_versi_s1[v], bare_s3, liquid_s3) for v in VERSI_REGRESI}
    banding_s3, coef_versi_s3, pers_versi_s3 = bandingkan_versi(
        after_liq_per_versi, powder_s3, final_s3, bare_s3, y_s3, ("pred_after_liquid", "powder"), "final")

    # Langkah 1: after liquid diprediksi pakai koefisien Stage 1 (versi terpilih)
    after_liq_s3 = prediksi_lab(coef_s1, bare_s3, liquid_s3)
    # Langkah 2: regresi hasil akhir
    coef_s3, pred_s3, pers_s3 = fit_regresi(after_liq_s3, powder_s3, final_s3, ("pred_after_liquid", "powder"), "final", versi)
    delta_s3 = tabel_fitur_delta(pred_s3, bare_s3)
    klas_s3 = evaluate_match_model(fitur_match(pred_s3, bare_s3), y_s3)
    S[3] = dict(nama="Stage 3 — Gabungan Liquid + Powder", data=df_s3, bare=bare_s3, y=y_s3,
                aktual=final_s3, pred=pred_s3, coef=coef_s3, persamaan=pers_s3,
                banding=banding_s3, persamaan_versi=pers_versi_s3, delta=delta_s3,
                uji=uji_fitur(delta_s3, y_s3), klas=klas_s3, label_target="After Liquid + Powder",
                swatch={"Bare Face": bare_s3, "Liquid": liquid_s3, "Powder": powder_s3,
                        "Aktual After Liquid": after_liq_aktual, "Pred After Liquid": after_liq_s3,
                        "Aktual Liq+Pow": final_s3, "Prediksi Liq+Pow": pred_s3})

    ringkasan_versi = pd.concat([banding_s1.assign(Stage="1. Bare Face + Liquid"),
                                 banding_s2.assign(Stage="2. Bare Face + Powder"),
                                 banding_s3.assign(Stage="3. Gabungan Liquid + Powder")], ignore_index=True)
    ringkasan_versi = ringkasan_versi[["Stage"] + [c for c in ringkasan_versi.columns if c != "Stage"]]
    return S, ringkasan_versi


def latih_semua(df_raw, versi):
    S, ringkasan_versi = latih_stage(df_raw, versi)
    ringkasan_ut, model_ut = latih_undertone(df_raw)
    ringkasan_level, model_level, figs_level = latih_level(df_raw)
    katalog_liquid = buat_katalog(df_raw, "{}_liquid")
    katalog_powder = buat_katalog(df_raw, "{}_produk", kolom_info=["Undertone", "L"])
    return {"versi": versi, "stage": S, "ringkasan_versi": ringkasan_versi,
            "ringkasan_undertone": ringkasan_ut, "model_undertone": model_ut,
            "ringkasan_level": ringkasan_level, "model_level": model_level, "figs_level": figs_level,
            "katalog_liquid": katalog_liquid, "katalog_powder": katalog_powder}


def klasifikasi_powder(katalog_pow, model_ut, model_level, metode_ut="theta", level_per_undertone=True):
    """Tambah kolom pred_Undertone & pred_Level ke tabel powder (L, a, b).
    Undertone: model 'Powder' dari bagian 6 (theta atau Hue otomatis).
    Level   : metode median dari bagian 7, batas per undertone hasil prediksi (atau semua data)."""
    out = katalog_pow.copy()
    ut = prediksi_undertone(model_ut["Powder"], out["a"].values, out["b"].values, metode_ut)
    out["pred_Undertone"] = ut
    out["pred_Level"] = prediksi_level_L(model_level, out["L"].values, "Powder",
                                         ut if level_per_undertone else None)
    return out

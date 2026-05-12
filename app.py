from io import BytesIO
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/Users/muhghifari/Documents/Codex/2026-05-13/files-mentioned-by-the-user-datasalesbogor/.matplotlib",
)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = Path("https://raw.githubusercontent.com/MuhGhifari/Prediksi-Penjualan/refs/heads/main/DATASALESBOGOR.csv")

FEATURES = ["DayNum", "Month", "DOW", "Week", "Lag1", "Lag7", "MA7"]
COLORS = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]


st.set_page_config(
    page_title="Prediksi Penjualan Bogor",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_from_upload(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes), sep=";", encoding="utf-8-sig")


def clean_data(df_raw: pd.DataFrame, q_value: float, q_qty: float) -> tuple[pd.DataFrame, dict]:
    df = df_raw.dropna(subset=["Date", "Value", "Qty", "Brand"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    df = df.dropna(subset=["Date", "Value", "Qty"]).copy()

    before = len(df)
    q_val = df["Value"].quantile(q_value)
    q_qt = df["Qty"].quantile(q_qty)
    outlier_mask = (df["Value"] > q_val) | (df["Qty"] > q_qt)
    df = df.loc[~outlier_mask].sort_values("Date").reset_index(drop=True)

    info = {
        "before": before,
        "after": len(df),
        "removed": int(outlier_mask.sum()),
        "value_threshold": q_val,
        "qty_threshold": q_qt,
    }
    return df, info


def apply_filters(
    df: pd.DataFrame,
    channels: list[str],
    brands: list[str],
    stores: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if channels:
        filtered = filtered[filtered["Channel"].isin(channels)]
    if brands:
        filtered = filtered[filtered["Brand"].isin(brands)]
    if stores:
        filtered = filtered[filtered["Nama Store"].isin(stores)]
    return filtered.copy()


def make_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("Date")
        .agg(TotalValue=("Value", "sum"), TotalQty=("Qty", "sum"))
        .reset_index()
        .sort_values("Date")
    )

    daily["DayNum"] = (daily["Date"] - daily["Date"].min()).dt.days
    daily["Month"] = daily["Date"].dt.month
    daily["DOW"] = daily["Date"].dt.dayofweek
    daily["Week"] = daily["Date"].dt.isocalendar().week.astype(int)
    daily["Lag1"] = daily["TotalValue"].shift(1)
    daily["Lag7"] = daily["TotalValue"].shift(7)
    daily["MA7"] = daily["TotalValue"].rolling(7).mean()
    return daily.dropna().reset_index(drop=True)


def train_prediction_models(daily: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    X = daily[FEATURES]
    y = daily["TotalValue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    backtest = pd.DataFrame(
        {"Date": daily.iloc[-len(y_test) :]["Date"].values, "Aktual": y_test.values}
    )
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
            "pred": pred,
            "model": model,
        }
        backtest[name] = pred

    return models, results, backtest


def forecast_sales(
    daily: pd.DataFrame,
    models: dict,
    results: dict,
    days: int,
    selected_model: str,
) -> tuple[str, object, pd.DataFrame]:
    best_name = selected_model
    if selected_model == "Auto Best R2":
        best_name = max(results, key=lambda n: results[n]["R2"])

    best_model = models[best_name]

    last_date = daily["Date"].max()
    last_day = daily["DayNum"].max()
    lag7_buf = list(daily["TotalValue"].iloc[-7:])
    lag1 = daily["TotalValue"].iloc[-1]
    future_preds = []

    for i in range(1, days + 1):
        fd = last_date + pd.Timedelta(days=i)
        row = {
            "DayNum": last_day + i,
            "Month": fd.month,
            "DOW": fd.dayofweek,
            "Week": fd.isocalendar()[1],
            "Lag1": lag1,
            "Lag7": lag7_buf[-7],
            "MA7": np.mean(lag7_buf[-7:]),
        }
        pred = best_model.predict(pd.DataFrame([row]))[0]
        future_preds.append({"Date": fd, "Predicted": pred})
        lag7_buf.append(pred)
        lag1 = pred

    return best_name, best_model, pd.DataFrame(future_preds)


def cluster_stores(df: pd.DataFrame, cluster_choice: str, manual_k: int):
    store_stats = (
        df.groupby(["Kode Store", "Nama Store", "Channel"])
        .agg(
            TotalValue=("Value", "sum"),
            TotalQty=("Qty", "sum"),
            TxCount=("Value", "count"),
            AvgOrder=("Value", "mean"),
            UniqueSKU=("SKU", "nunique"),
        )
        .reset_index()
    )

    feat_clust = store_stats[["TotalValue", "TotalQty", "TxCount", "AvgOrder", "UniqueSKU"]]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feat_clust)

    max_k = min(7, len(store_stats) - 1)
    sil_scores = {}
    if max_k >= 2:
        for k in range(2, max_k + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x_scaled)
            sil_scores[k] = silhouette_score(x_scaled, labels)
        best_k = max(sil_scores, key=sil_scores.get)
        if cluster_choice == "Manual":
            best_k = min(max(manual_k, 2), max_k)
    else:
        best_k = 1

    if best_k == 1:
        store_stats["Cluster"] = 0
        store_stats["PC1"] = 0
        store_stats["PC2"] = 0
        explained = (0, 0)
    else:
        km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        store_stats["Cluster"] = km_best.fit_predict(x_scaled)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(x_scaled)
        store_stats["PC1"] = pcs[:, 0]
        store_stats["PC2"] = pcs[:, 1]
        explained = pca.explained_variance_ratio_

    cluster_profile = (
        store_stats.groupby("Cluster")
        .agg(
            Jumlah_Toko=("Kode Store", "count"),
            Avg_TotalValue=("TotalValue", "mean"),
            Avg_TxCount=("TxCount", "mean"),
            Avg_OrderValue=("AvgOrder", "mean"),
            Avg_SKU=("UniqueSKU", "mean"),
        )
        .reset_index()
    )

    return store_stats, cluster_profile, sil_scores, best_k, explained


def make_dashboard(
    daily: pd.DataFrame,
    future_df: pd.DataFrame,
    results: dict,
    store_stats: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    sil_scores: dict,
    best_k: int,
    explained,
    df: pd.DataFrame,
) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(daily["Date"], daily["TotalValue"] / 1e6, alpha=0.2, color=COLORS[0])
    ax1.plot(daily["Date"], daily["TotalValue"] / 1e6, color=COLORS[0], lw=1.5, label="Aktual")
    ax1.plot(
        future_df["Date"],
        future_df["Predicted"] / 1e6,
        color=COLORS[1],
        lw=2.5,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Prediksi",
    )
    ax1.axvline(daily["Date"].max(), color="gray", linestyle=":", lw=1.5)
    ax1.set_title("Tren Penjualan Harian dan Prediksi", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Nilai Penjualan (Juta Rp)")
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))

    ax2 = fig.add_subplot(gs[1, 0])
    model_names = list(results.keys())
    r2_vals = [results[n]["R2"] for n in model_names]
    bars = ax2.barh(model_names, r2_vals, color=[COLORS[0], COLORS[1], COLORS[2]])
    ax2.set_xlim(min(0, min(r2_vals) - 0.1), max(1, max(r2_vals) + 0.1))
    ax2.set_title("Perbandingan Akurasi Model (R2 Score)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("R2 Score")
    for bar, val in zip(bars, r2_vals):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    ax3 = fig.add_subplot(gs[1, 1])
    mae_vals = [results[n]["MAE"] / 1e6 for n in model_names]
    bars2 = ax3.barh(model_names, mae_vals, color=[COLORS[0], COLORS[1], COLORS[2]])
    ax3.set_title("MAE Model (Juta Rp) - Lebih Kecil Lebih Baik", fontsize=12, fontweight="bold")
    ax3.set_xlabel("MAE (Juta Rp)")
    for bar, val in zip(bars2, mae_vals):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center")

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(
        store_stats["PC1"],
        store_stats["PC2"],
        c=[COLORS[int(c) % len(COLORS)] for c in store_stats["Cluster"]],
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
        s=60,
    )
    for c in range(best_k):
        sub = store_stats[store_stats["Cluster"] == c]
        if len(sub):
            ax4.annotate(
                f"Klaster {c}",
                (sub["PC1"].mean(), sub["PC2"].mean()),
                fontsize=10,
                fontweight="bold",
                color=COLORS[c % len(COLORS)],
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    ax4.set_title(f"Clustering Toko (K-Means, k={best_k})", fontsize=12, fontweight="bold")
    ax4.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax4.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")

    ax5 = fig.add_subplot(gs[2, 1])
    x, width = np.arange(best_k), 0.35
    ax5.bar(x - width / 2, cluster_profile["Avg_TotalValue"] / 1e6, width, color=COLORS[0], label="Avg Total Value")
    ax5r = ax5.twinx()
    ax5r.bar(x + width / 2, cluster_profile["Avg_TxCount"], width, color=COLORS[1], label="Avg Tx Count")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"Klaster {i}" for i in range(best_k)])
    ax5.set_title("Profil Setiap Klaster", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Avg Total Value (Juta Rp)", color=COLORS[0])
    ax5r.set_ylabel("Avg Tx Count", color=COLORS[1])

    ax6 = fig.add_subplot(gs[3, 0])
    ch_brand = df.groupby(["Channel", "Brand"])["Value"].sum().unstack().fillna(0) / 1e6
    ch_brand.plot(kind="bar", ax=ax6, color=COLORS[: len(ch_brand.columns)], edgecolor="white")
    ax6.set_title("Nilai Penjualan per Channel dan Brand", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Channel")
    ax6.set_ylabel("Total Nilai (Juta Rp)")
    ax6.legend(title="Brand", fontsize=8)
    ax6.tick_params(axis="x", rotation=0)

    ax7 = fig.add_subplot(gs[3, 1])
    if sil_scores:
        ax7.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o", color=COLORS[2], lw=2.5)
        ax7.axvline(best_k, color=COLORS[3], linestyle="--", lw=2, label=f"Best k={best_k}")
        ax7.set_xticks(list(sil_scores.keys()))
        ax7.legend(fontsize=10)
    ax7.set_title("Silhouette Score untuk Menentukan Jumlah Klaster", fontsize=12, fontweight="bold")
    ax7.set_xlabel("Jumlah Klaster (k)")
    ax7.set_ylabel("Silhouette Score")

    fig.suptitle(
        "Sales Analytics Dashboard - Bogor Region 2025\nPrediksi Penjualan dan Segmentasi Toko",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    return fig


def build_excel(future_df: pd.DataFrame, store_stats: pd.DataFrame, metrics: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        future_df.to_excel(writer, sheet_name="Prediksi", index=False)
        store_stats.to_excel(writer, sheet_name="Clustering Toko", index=False)
        metrics.to_excel(writer, sheet_name="Evaluasi Model", index=False)
    return buffer.getvalue()


st.title("Prediksi Penjualan dan Clustering Toko")
st.caption("Streamlit app berdasarkan notebook `sales_prediction_bogor (testing).ipynb`.")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    forecast_days = st.slider("Jumlah hari prediksi", min_value=7, max_value=90, value=30)
    q_value = st.slider("Batas outlier Value", 0.90, 1.00, 0.99, 0.01)
    q_qty = st.slider("Batas outlier Qty", 0.90, 1.00, 0.99, 0.01)
    model_choice = st.selectbox(
        "Model prediksi",
        ["Auto Best R2", "Linear Regression", "Random Forest", "Gradient Boosting"],
    )
    cluster_choice = st.radio("Jumlah klaster", ["Auto Silhouette", "Manual"])
    manual_k = st.slider("Manual k", 2, 7, 3, disabled=cluster_choice == "Auto Silhouette")


try:
    if uploaded_file is not None:
        raw_df = load_from_upload(uploaded_file.getvalue())
        source_name = uploaded_file.name
    else:
        raw_df = load_from_path(str(DEFAULT_DATA_PATH))
        source_name = str(DEFAULT_DATA_PATH)
except Exception as exc:
    st.error(f"Gagal memuat data: {exc}")
    st.stop()

df, cleaning_info = clean_data(raw_df, q_value, q_qty)

with st.sidebar:
    st.header("Filter")
    channels = sorted(df["Channel"].dropna().unique().tolist())
    brands = sorted(df["Brand"].dropna().unique().tolist())
    stores = sorted(df["Nama Store"].dropna().unique().tolist())
    selected_channels = st.multiselect("Channel", channels)
    selected_brands = st.multiselect("Brand", brands)
    selected_stores = st.multiselect("Store", stores)

df_filtered = apply_filters(df, selected_channels, selected_brands, selected_stores)
if df_filtered.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

daily = make_daily_features(df_filtered)
if len(daily) < 30:
    st.warning("Data harian terlalu sedikit untuk training model. Kurangi filter atau gunakan data lebih lengkap.")
    st.stop()

models, results, backtest = train_prediction_models(daily)
best_name, best_model, future_df = forecast_sales(
    daily, models, results, forecast_days, model_choice
)
store_stats, cluster_profile, sil_scores, best_k, explained = cluster_stores(
    df_filtered, "Manual" if cluster_choice == "Manual" else "Auto", manual_k
)

metrics = (
    pd.DataFrame(
        [
            {"Model": name, "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"]}
            for name, r in results.items()
        ]
    )
    .sort_values("R2", ascending=False)
    .reset_index(drop=True)
)

st.subheader("Ringkasan Data")
st.write(f"Sumber data: `{source_name}`")
cols = st.columns(5)
cols[0].metric("Baris bersih", f"{len(df_filtered):,}")
cols[1].metric("Total Value", f"Rp {df_filtered['Value'].sum():,.0f}")
cols[2].metric("Total Qty", f"{df_filtered['Qty'].sum():,.0f}")
cols[3].metric("Toko", f"{df_filtered['Kode Store'].nunique():,}")
cols[4].metric("SKU", f"{df_filtered['SKU'].nunique():,}")

with st.expander("Detail cleaning outlier"):
    st.write(f"Baris sebelum outlier filter: **{cleaning_info['before']:,}**")
    st.write(f"Baris setelah outlier filter: **{cleaning_info['after']:,}**")
    st.write(f"Baris terhapus: **{cleaning_info['removed']:,}**")
    st.write(f"Batas Value: **Rp {cleaning_info['value_threshold']:,.0f}**")
    st.write(f"Batas Qty: **{cleaning_info['qty_threshold']:,.0f}**")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Prediksi", "Evaluasi Model", "Clustering Toko", "Data"]
)

with tab1:
    st.subheader("Tren Penjualan dan Prediksi")
    st.write(f"Model yang digunakan: **{best_name}**")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(daily["Date"], daily["TotalValue"] / 1e6, alpha=0.2, color=COLORS[0])
    ax.plot(daily["Date"], daily["TotalValue"] / 1e6, color=COLORS[0], lw=1.6, label="Aktual")
    ax.plot(
        future_df["Date"],
        future_df["Predicted"] / 1e6,
        color=COLORS[1],
        lw=2.4,
        linestyle="--",
        marker="o",
        markersize=4,
        label=f"Prediksi {forecast_days} Hari",
    )
    ax.axvline(daily["Date"].max(), color="gray", linestyle=":", lw=1.5)
    ax.set_ylabel("Total Nilai Penjualan (Juta Rp)")
    ax.legend()
    st.pyplot(fig)

    display_future = future_df.copy()
    display_future["Predicted"] = display_future["Predicted"].round(0)
    st.dataframe(display_future, use_container_width=True)
    st.download_button(
        "Download prediksi CSV",
        display_future.to_csv(index=False).encode("utf-8"),
        "prediksi_penjualan.csv",
        "text/csv",
    )

with tab2:
    st.subheader("Hasil Evaluasi Model")
    st.dataframe(metrics, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.bar_chart(metrics.set_index("Model")[["R2"]])
    c2.bar_chart(metrics.set_index("Model")[["MAE", "RMSE"]])

    st.subheader("Backtest Aktual vs Prediksi")
    backtest_chart = backtest[["Date", "Aktual", best_name]].set_index("Date")
    st.line_chart(backtest_chart)

with tab3:
    st.subheader(f"Clustering Toko - k={best_k}")
    st.dataframe(cluster_profile, use_container_width=True)

    if best_k > 1:
        scatter_df = store_stats[["PC1", "PC2", "Cluster", "Nama Store", "TotalValue"]]
        st.scatter_chart(scatter_df, x="PC1", y="PC2", color="Cluster", size="TotalValue")

    if sil_scores:
        st.subheader("Silhouette Score")
        sil_df = pd.DataFrame(
            {"k": list(sil_scores.keys()), "Silhouette Score": list(sil_scores.values())}
        ).set_index("k")
        st.line_chart(sil_df)

    st.subheader("Hasil Clustering per Toko")
    st.dataframe(store_stats, use_container_width=True)

with tab4:
    st.subheader("Data Bersih")
    st.dataframe(df_filtered, use_container_width=True)
    st.subheader("Data Harian untuk Modeling")
    st.dataframe(daily, use_container_width=True)

st.subheader("Dashboard Lengkap")
dashboard_fig = make_dashboard(
    daily,
    future_df,
    results,
    store_stats,
    cluster_profile,
    sil_scores,
    best_k,
    explained,
    df_filtered,
)
st.pyplot(dashboard_fig)

excel_bytes = build_excel(display_future, store_stats, metrics)
st.download_button(
    "Download semua hasil sebagai Excel",
    excel_bytes,
    "hasil_prediksi_dan_clustering.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

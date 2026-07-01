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
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = Path("DATASALESBOGOR.csv")

FEATURES = ["DayNum", "Month", "DOW", "Week", "Lag1", "Lag7", "MA7"]
COLORS = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
CLUSTER_CMAP = "RdBu"


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


def clean_data(df_raw: pd.DataFrame, q_value: float, q_qty: float) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    # Keep track of original raw index
    raw_df_with_idx = df_raw.copy()
    raw_df_with_idx["_orig_index"] = raw_df_with_idx.index
    
    # 1. Drop missing critical columns
    df = raw_df_with_idx.dropna(subset=["Date", "Value", "Qty", "Brand"]).copy()
    
    # 2. Parse types and drop parsing failures
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    df = df.dropna(subset=["Date", "Value", "Qty"]).copy()

    before = len(df)
    
    # 3. Filter outliers (Only filter out the extreme 2B integer overflow placeholder)
    outlier_mask = df["Qty"] > 2_000_000_000
    df_clean = df.loc[~outlier_mask].copy()
    
    # Reset index and clean temporary columns
    df_clean_final = df_clean.sort_values("Date").reset_index(drop=True)
    if "_orig_index" in df_clean_final.columns:
        df_clean_final = df_clean_final.drop(columns=["_orig_index"])
    
    # The rows that were removed at any stage:
    clean_orig_indices = set(df_clean["_orig_index"])
    removed_mask = ~raw_df_with_idx["_orig_index"].isin(clean_orig_indices)
    df_removed = df_raw.loc[removed_mask].copy()
    
    # Add a column explaining WHY it was removed
    df_removed["Alasan Dihapus"] = "Data Kosong"
    
    # Update reasons for records that failed the outlier check
    outlier_orig_indices = set(df.loc[outlier_mask, "_orig_index"])
    df_removed.loc[df_removed.index.isin(outlier_orig_indices), "Alasan Dihapus"] = "Outlier"

    info = {
        "before": before,
        "after": len(df_clean_final),
        "removed": int(outlier_mask.sum()),
        "value_threshold": 0.0,
        "qty_threshold": 2_000_000_000.0,
    }
    return df_clean_final, info, df_removed


def apply_filters(
    df: pd.DataFrame,
    channels: list[str],
    stores: list[str],
    brands: list[str],
    subbrands: list[str],
    product_groups: list[str],
    skus: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if channels:
        filtered = filtered[filtered["Channel"].isin(channels)]
    if stores:
        filtered = filtered[filtered["Nama Store"].isin(stores)]
    if brands:
        filtered = filtered[filtered["Brand"].isin(brands)]
    if subbrands:
        filtered = filtered[filtered["Subbrand"].isin(subbrands)]
    if product_groups:
        filtered = filtered[filtered["Product Group"].isin(product_groups)]
    if skus:
        filtered = filtered[filtered["SKU"].isin(skus)]
    return filtered.copy()


def make_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("Date")
        .agg(TotalValue=("Value", "sum"), TotalQty=("Qty", "sum"))
        .reset_index()
        .sort_values("Date")
    )
    
    if daily.empty:
        return daily

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


def build_excel(future_df: pd.DataFrame, store_stats: pd.DataFrame, metrics: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        future_df.to_excel(writer, sheet_name="Prediksi", index=False)
        store_stats.to_excel(writer, sheet_name="Clustering Toko", index=False)
        metrics.to_excel(writer, sheet_name="Evaluasi Model", index=False)
    return buffer.getvalue()


def plot_cluster_scatter_plotly(store_stats: pd.DataFrame, best_k: int, explained, cluster_profile: pd.DataFrame) -> go.Figure:
    # Sort cluster profile by Total Value to assign business-friendly names
    profile_sorted = cluster_profile.sort_values("Avg_TotalValue", ascending=False).reset_index()
    cluster_names = {}
    labels = [
        "🏆 Klaster Emas (Toko Utama / Kinerja Tinggi)",
        "🥈 Klaster Perak (Toko Menengah / Kinerja Sedang)",
        "🥉 Klaster Perunggu (Toko Kecil / Potensial)"
    ]
    for idx, row in profile_sorted.iterrows():
        c_id = int(row["Cluster"])
        label = labels[idx] if idx < len(labels) else f"Klaster {c_id} (Lainnya)"
        cluster_names[c_id] = label

    store_stats_display = store_stats.copy()
    store_stats_display["Segmen Bisnis"] = store_stats_display["Cluster"].map(cluster_names)
    store_stats_display["Formatted Revenue"] = store_stats_display["TotalValue"].apply(format_rupiah_compact)
    
    # Determine size based on sales value
    size_base = store_stats_display["TotalValue"].clip(lower=0)
    max_size = max(size_base.max(), 1)
    store_stats_display["BubbleSize"] = 15 + (size_base / max_size) * 45
    
    fig = px.scatter(
        store_stats_display,
        x="PC1",
        y="PC2",
        color="Segmen Bisnis",
        size="BubbleSize",
        hover_name="Nama Store",
        hover_data={
            "PC1": False,
            "PC2": False,
            "BubbleSize": False,
            "Segmen Bisnis": True,
            "Formatted Revenue": True,
            "TxCount": True,
            "UniqueSKU": True
        },
        labels={
            "Formatted Revenue": "Total Penjualan",
            "TxCount": "Jumlah Transaksi",
            "UniqueSKU": "SKU Unik"
        },
        color_discrete_sequence=["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#EC4899", "#8B5CF6"]
    )
    
    fig.update_layout(
        xaxis_title=f"PC1 ({explained[0] * 100:.1f}%)",
        yaxis_title=f"PC2 ({explained[1] * 100:.1f}%)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    )
    return fig


def format_rupiah_compact(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"Rp {value / 1_000_000_000_000:.2f} T"
    if abs_value >= 1_000_000_000:
        return f"Rp {value / 1_000_000_000:.2f} M"
    if abs_value >= 1_000_000:
        return f"Rp {value / 1_000_000:.2f} Juta"
    if abs_value >= 1_000:
        return f"Rp {value / 1_000:.2f} Ribu"
    return f"Rp {value:,.0f}"


# Custom CSS styling for premium look & layout
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .kpi-container {
        padding: 1.25rem;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .kpi-container:hover {
        transform: translateY(-2px);
        background-color: rgba(128, 128, 128, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%); padding: 2.5rem; border-radius: 16px; color: white; margin-bottom: 2.5rem; box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.2);">
        <h1 style="color: white; margin: 0; font-size: 2.25rem; font-weight: 800;">Bogor Sales Intelligence Hub</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem;">Platform Prediksi Penjualan & Segmentasi Toko Berbasis AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Sumber Data")
    uploaded_file = st.file_uploader("Unggah File Penjualan (CSV)", type=["csv"])

# Set default modeling/clustering parameters
forecast_days = 30
q_value = 0.99
q_qty = 0.99
model_choice = "Auto Best R2"
cluster_choice = "Auto Silhouette"
manual_k = 3

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

# Clean data with default outlier settings
df, cleaning_info, df_removed = clean_data(raw_df, q_value, q_qty)

# Define Tabs
tab_overview, tab_lookup, tab_cluster, tab_model, tab_data = st.tabs(
    [
        "📊 Ringkasan Eksekutif",
        "🔍 Pencarian & Prediksi",
        "🎯 Segmentasi Toko (Clustering)",
        "🤖 Evaluasi Model AI",
        "📁 Eksplorasi Data"
    ]
)

# Render filter parameters inside the Pencarian Prediksi tab first, to get their selection values
with tab_lookup:
    st.subheader("Pencarian Prediksi Berdasarkan Parameter Terpilih")
    st.write(
        "Gunakan bagian di bawah ini untuk menentukan Store (Toko), Brand, Product (SKU), dan Tanggal. "
        "Model prediksi akan dilatih ulang secara otomatis untuk data hasil filter tersebut."
    )
    
    col_filt1, col_filt2 = st.columns(2)
    with col_filt1:
        # Store Filter
        stores = sorted(df["Nama Store"].dropna().unique().tolist())
        selected_stores = st.multiselect("Pilih Store / Toko", stores, help="Kosongkan untuk memilih semua store")
        
    with col_filt2:
        # Brand Filter (cascading from Store)
        df_brand_opt = df
        if selected_stores:
            df_brand_opt = df_brand_opt[df_brand_opt["Nama Store"].isin(selected_stores)]
        brands = sorted(df_brand_opt["Brand"].dropna().unique().tolist())
        selected_brands = st.multiselect("Pilih Brand / Merek", brands, help="Kosongkan untuk memilih semua brand")
        
        # SKU Filter (cascading from Brand)
        df_sku_opt = df_brand_opt
        if selected_brands:
            df_sku_opt = df_sku_opt[df_sku_opt["Brand"].isin(selected_brands)]
        skus = sorted(df_sku_opt["SKU"].dropna().unique().tolist())
        selected_skus = st.multiselect("Pilih Product / SKU", skus, help="Kosongkan untuk memilih semua product")

    st.write("---")
    
    # Date Picker
    min_hist_date = df["Date"].min()
    max_hist_date = df["Date"].max()
    max_forecast_date = max_hist_date + pd.Timedelta(days=forecast_days)
    
    lookup_date = st.date_input(
        "Pilih Tanggal Pencarian",
        value=max_hist_date.date(),
        min_value=min_hist_date.date(),
        max_value=max_forecast_date.date(),
        key="lookup_date_picker"
    )
    lookup_datetime = pd.to_datetime(lookup_date)

# Apply filters based on selected parameters
df_filtered = apply_filters(
    df,
    [],  # channels (removed)
    selected_stores,
    selected_brands,
    [],  # subbrands (removed)
    [],  # product_groups (removed)
    selected_skus,
)

if df_filtered.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# Aggregate daily data and train prediction models
daily = make_daily_features(df_filtered)
if len(daily) < 30:
    st.warning("Data harian terlalu sedikit untuk training model. Kurangi filter atau gunakan data lebih lengkap.")
    st.stop()

models, results, backtest = train_prediction_models(daily)
best_name, best_model, future_df = forecast_sales(
    daily, models, results, forecast_days, model_choice
)
store_stats, cluster_profile, sil_scores, best_k, explained = cluster_stores(
    df_filtered, cluster_choice, manual_k
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

# Tab 1: Executive Overview
with tab_overview:
    st.subheader("Ringkasan Kinerja Penjualan")
    
    # KPI Columns
    cols = st.columns(4)
    total_value = df_filtered["Value"].sum()
    total_qty = df_filtered["Qty"].sum()
    num_stores = df_filtered["Kode Store"].nunique()
    num_skus = df_filtered["SKU"].nunique()
    
    formatted_revenue = format_rupiah_compact(total_value)
    
    with cols[0]:
        st.markdown(f'<div class="kpi-container"><div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Total Nilai Penjualan</div><div style="font-size: 1.5rem; font-weight: 700; color: #4f46e5; margin-top: 0.25rem;">{formatted_revenue}</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="kpi-container"><div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Kuantitas Penjualan</div><div style="font-size: 1.5rem; font-weight: 700; color: #10b981; margin-top: 0.25rem;">{total_qty:,.0f} Pcs</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="kpi-container"><div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Toko Terjangkau</div><div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b; margin-top: 0.25rem;">{num_stores:,} Store</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="kpi-container"><div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Katalog Produk</div><div style="font-size: 1.5rem; font-weight: 700; color: #ef4444; margin-top: 0.25rem;">{num_skus:,} SKU</div></div>', unsafe_allow_html=True)
    
    st.write("")
    
    with st.expander("ℹ️ Klik untuk Melihat Detail Pembersihan Data & Outlier"):
        st.markdown(f"""
        * **Sumber File**: `{source_name}`
        * **Total Baris Valid (Filtered)**: `{len(df_filtered):,}`
        * **Sebelum Outlier Filter**: `{cleaning_info['before']:,} baris`
        * **Setelah Outlier Filter**: `{cleaning_info['after']:,} baris` (Pembersihan membuang `{cleaning_info['removed']:,}` baris outlier ekstrem)
        * **Aturan Outlier**: Hanya menyaring baris placeholder overflow (`Qty > 2 Miliar`)
        """)
        
    st.write("---")
    st.subheader("Tren Penjualan Aktual & Prediksi Masa Depan")
    st.write(f"Model Forecasting Terpilih: **{best_name}**")
    
    # Plotly Interactive Chart
    fig_overview = go.Figure()
    fig_overview.add_trace(go.Scatter(
        x=daily["Date"],
        y=daily["TotalValue"],
        mode="lines",
        name="Penjualan Aktual",
        line=dict(color="#4f46e5", width=2.5),
        hovertemplate="Tanggal: %{x|%d %b %Y}<br>Penjualan: Rp %{y:,.0f}<extra></extra>"
    ))
    fig_overview.add_trace(go.Scatter(
        x=future_df["Date"],
        y=future_df["Predicted"],
        mode="lines+markers",
        name="Prediksi AI",
        line=dict(color="#10b981", width=2.5, dash="dash"),
        marker=dict(size=5),
        hovertemplate="Tanggal: %{x|%d %b %Y}<br>Prediksi: Rp %{y:,.0f}<extra></extra>"
    ))
    # Connect the gap
    if len(daily) > 0:
        fig_overview.add_trace(go.Scatter(
            x=[daily["Date"].iloc[-1], future_df["Date"].iloc[0]],
            y=[daily["TotalValue"].iloc[-1], future_df["Predicted"].iloc[0]],
            mode="lines",
            showlegend=False,
            line=dict(color="#10b981", width=2.5, dash="dash"),
            hoverinfo="skip"
        ))
    
    fig_overview.update_layout(
        hovermode="x unified",
        xaxis_title="Tanggal",
        yaxis_title="Nilai Penjualan (Rp)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    )
    st.plotly_chart(fig_overview, use_container_width=True)
    
    st.write("")
    st.write("---")
    st.subheader("Unduh Laporan Executive")
    st.write("Unduh data hasil analisis segmentasi, model evaluasi, dan forecast masa depan dalam satu file Excel profesional.")
    excel_bytes = build_excel(future_df, store_stats, metrics)
    st.download_button(
        "📥 Unduh Laporan Lengkap (Excel / .xlsx)",
        excel_bytes,
        "laporan_sales_bogor.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# Tab 2: Prediction Lookup Results
with tab_lookup:
    st.write("---")
    st.subheader("Hasil Pencarian Prediksi & Historis")
    
    # Summary of currently active filters
    st.markdown(f"""
    **Segmentasi Pencarian Aktif**:
    * **Store**: {', '.join(selected_stores) if selected_stores else 'Semua Toko'}
    * **Brand**: {', '.join(selected_brands) if selected_brands else 'Semua Brand'}
    * **Product (SKU)**: {', '.join(selected_skus) if selected_skus else 'Semua SKU'}
    """)
    
    forecast_row = future_df[future_df["Date"] == lookup_datetime]
    hist_row = daily[daily["Date"] == lookup_datetime]
    backtest_row = backtest[backtest["Date"] == lookup_datetime] if not backtest.empty else pd.DataFrame()
    
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        if not forecast_row.empty:
            pred_val = forecast_row.iloc[0]["Predicted"]
            st.markdown("### Status: **Prediksi Masa Depan**")
            st.metric(
                label=f"Perkiraan Nilai Penjualan ({lookup_date.strftime('%d %b %Y')})",
                value=format_rupiah_compact(pred_val),
            )
        elif not hist_row.empty:
            actual_val = hist_row.iloc[0]["TotalValue"]
            st.markdown("### Status: **Data Historis (Aktual)**")
            if not backtest_row.empty and best_name in backtest_row.columns:
                pred_backtest = backtest_row.iloc[0][best_name]
                error = actual_val - pred_backtest
                error_pct = (error / actual_val) * 100 if actual_val != 0 else 0
                
                st.metric("Nilai Aktual Terjual", format_rupiah_compact(actual_val))
                st.metric(
                    f"Prediksi AI ({best_name})", 
                    format_rupiah_compact(pred_backtest), 
                    f"Selisih Margin: {error_pct:.2f}%", 
                    delta_color="inverse"
                )
            else:
                st.metric("Nilai Aktual Terjual", format_rupiah_compact(actual_val))
        else:
            st.warning("Tidak ada data transaksi atau prediksi pada tanggal yang dipilih.")
            
    with col_res2:
        # Full-year timeline chart with search date highlight
        st.markdown("##### Tren Penjualan & Posisi Tanggal Pencarian")
        
        fig_win = go.Figure()
        fig_win.add_trace(go.Scatter(
            x=daily["Date"],
            y=daily["TotalValue"],
            mode="lines",
            name="Aktual",
            line=dict(color="#4f46e5", width=2.5),
            hovertemplate="Tanggal: %{x|%d %b %Y}<br>Penjualan: Rp %{y:,.0f}<extra></extra>"
        ))
        
        fig_win.add_trace(go.Scatter(
            x=future_df["Date"],
            y=future_df["Predicted"],
            mode="lines+markers",
            name="Prediksi",
            line=dict(color="#10b981", width=2.5, dash="dash"),
            marker=dict(size=4),
            hovertemplate="Tanggal: %{x|%d %b %Y}<br>Prediksi: Rp %{y:,.0f}<extra></extra>"
        ))
        
        # Connect the gap
        if len(daily) > 0:
            fig_win.add_trace(go.Scatter(
                x=[daily["Date"].iloc[-1], future_df["Date"].iloc[0]],
                y=[daily["TotalValue"].iloc[-1], future_df["Predicted"].iloc[0]],
                mode="lines",
                showlegend=False,
                line=dict(color="#10b981", width=2.5, dash="dash"),
                hoverinfo="skip"
            ))
            
        # Highlight search date
        max_y = max(daily["TotalValue"].max(), future_df["Predicted"].max())
        fig_win.add_shape(
            type="line",
            x0=lookup_datetime,
            y0=0,
            x1=lookup_datetime,
            y1=max_y * 1.05 if max_y > 0 else 100,
            line=dict(color="#ef4444", width=2, dash="dot"),
        )
        
        fig_win.update_layout(
            hovermode="x unified",
            xaxis_title="Tanggal",
            yaxis_title="Nilai Penjualan (Rp)",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
        )
        st.plotly_chart(fig_win, use_container_width=True)
            
    st.write("---")
    st.subheader("Data Hasil Forecast Lengkap (30 Hari)")
    display_future = future_df.copy()
    display_future["Predicted"] = display_future["Predicted"].round(0)
    st.dataframe(display_future, use_container_width=True)
    st.download_button(
        "📥 Unduh Tabel Prediksi Saja (CSV)",
        display_future.to_csv(index=False).encode("utf-8"),
        "forecast_penjualan.csv",
        "text/csv",
        use_container_width=True
    )

# Tab 3: Store Segmentation / Clustering
with tab_cluster:
    st.subheader(f"Analisis Segmentasi Toko (K-Means Clustering, k={best_k})")
    st.markdown(
        "Model K-Means mengelompokkan toko berdasarkan 5 dimensi: **Total Nilai Penjualan**, **Kuantitas Produk**, "
        "**Jumlah Transaksi**, **Rata-Rata Nilai Belanja**, dan **Variasi SKU** yang terjual."
    )
    
    # Sort cluster profile by Total Value to assign business-friendly names
    profile_sorted = cluster_profile.sort_values("Avg_TotalValue", ascending=False).reset_index()
    cluster_names = {}
    labels = [
        "🏆 Klaster Emas (Toko Utama / Kinerja Tinggi)",
        "🥈 Klaster Perak (Toko Menengah / Kinerja Sedang)",
        "🥉 Klaster Perunggu (Toko Kecil / Potensial)"
    ]
    for idx, row in profile_sorted.iterrows():
        c_id = int(row["Cluster"])
        label = labels[idx] if idx < len(labels) else f"Klaster {c_id} (Lainnya)"
        cluster_names[c_id] = label
        
    disp_profile = cluster_profile.copy()
    disp_profile["Kategori Segmen"] = disp_profile["Cluster"].map(cluster_names)
    disp_profile["Jumlah Toko"] = disp_profile["Jumlah_Toko"]
    disp_profile["Rata-Rata Revenue"] = disp_profile["Avg_TotalValue"].apply(format_rupiah_compact)
    disp_profile["Rata-Rata Transaksi"] = disp_profile["Avg_TxCount"].round(1)
    disp_profile["Rata-Rata Nilai Order"] = disp_profile["Avg_OrderValue"].apply(format_rupiah_compact)
    disp_profile["Rata-Rata SKU Unik"] = disp_profile["Avg_SKU"].round(1)
    
    disp_cols = [
        "Kategori Segmen", "Jumlah Toko", "Rata-Rata Revenue", 
        "Rata-Rata Transaksi", "Rata-Rata Nilai Order", "Rata-Rata SKU Unik"
    ]
    st.dataframe(disp_profile[disp_cols].set_index("Kategori Segmen"), use_container_width=True)
    
    st.write("---")
    st.subheader("Peta Segmentasi Toko (Visualisasi PCA 2D)")
    st.markdown(
        "Visualisasi di bawah menggunakan reduksi dimensi PCA untuk memetakan toko berdimensi tinggi ke ruang 2D. "
        "Toko dengan performa operasional yang mirip akan mengelompok bersama."
    )
    if best_k > 1:
        fig_pca_plotly = plot_cluster_scatter_plotly(store_stats, best_k, explained, cluster_profile)
        st.plotly_chart(fig_pca_plotly, use_container_width=True)
        
    if sil_scores:
        st.write("---")
        st.subheader("Silhouette Score (Penentuan Jumlah Klaster Optimal)")
        st.markdown(
            "Skor Silhouette mengukur seberapa dekat setiap titik dalam satu klaster dengan titik di klaster tetangga. "
            "Skor tertinggi menunjukkan pembagian segmen yang paling optimal secara matematis."
        )
        
        # Plotly Silhouette Score Chart
        fig_sil = go.Figure()
        ks = list(sil_scores.keys())
        scores = list(sil_scores.values())
        best_k_val = max(sil_scores, key=sil_scores.get)
        
        fig_sil.add_trace(go.Scatter(
            x=ks,
            y=scores,
            mode="lines+markers",
            name="Score",
            line=dict(color="#f59e0b", width=3),
            marker=dict(size=8, color="#f59e0b"),
            hovertemplate="Jumlah Klaster (k): %{x}<br>Score: %{y:.4f}<extra></extra>"
        ))
        
        # Highlight best k with vertical red dashed line
        best_score = sil_scores[best_k_val]
        fig_sil.add_shape(
            type="line",
            x0=best_k_val,
            y0=min(scores) - 0.01,
            x1=best_k_val,
            y1=max(scores) + 0.01,
            line=dict(color="#ef4444", width=2, dash="dash"),
        )
        
        # Add annotation for Best k
        fig_sil.add_annotation(
            x=best_k_val,
            y=best_score,
            text=f"Best k={best_k_val}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ef4444",
            arrowsize=1,
            arrowwidth=2,
            ax=40,
            ay=-30,
            bgcolor="rgba(239, 68, 68, 0.1)",
            bordercolor="#ef4444",
            borderwidth=1,
            borderpad=4,
            font=dict(size=11, color="#ef4444")
        )
        
        # Enable auto-scaling (disable zero baseline)
        fig_sil.update_layout(
            xaxis_title="Jumlah Cluster (k)",
            yaxis_title="Score",
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickmode="linear", tick0=2, dtick=1, showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
            yaxis=dict(
                showgrid=True, 
                gridcolor="rgba(128,128,128,0.15)",
                rangemode="normal"
            )
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        
    st.write("---")
    st.subheader("Data Lengkap Klasifikasi Toko")
    disp_stats = store_stats.copy()
    disp_stats["Segmen Bisnis"] = disp_stats["Cluster"].map(cluster_names)
    disp_stats["Total Penjualan"] = disp_stats["TotalValue"].apply(format_rupiah_compact)
    disp_stats["Kuantitas Penjualan"] = disp_stats["TotalQty"]
    disp_stats["Jumlah Transaksi"] = disp_stats["TxCount"]
    disp_stats["Rata-Rata Nilai Order"] = disp_stats["AvgOrder"].apply(format_rupiah_compact)
    disp_stats["Katalog SKU Unik"] = disp_stats["UniqueSKU"]
    
    detail_cols = ["Nama Store", "Channel", "Segmen Bisnis", "Total Penjualan", "Kuantitas Penjualan", "Jumlah Transaksi", "Rata-Rata Nilai Order", "Katalog SKU Unik"]
    st.dataframe(disp_stats[detail_cols].set_index("Nama Store"), use_container_width=True)

# Tab 4: AI Model Evaluation
with tab_model:
    st.subheader(f"Laporan Performa Model AI ({best_name})")
    st.markdown(
        "Akurasi model dievaluasi menggunakan metode *Backtesting* (melatih model pada data historis "
        "dan membandingkan prediksinya dengan sisa data aktual pengujian)."
    )
    
    # Get metrics for the active model
    best_metrics = metrics[metrics["Model"] == best_name].iloc[0]
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(
            f'<div class="kpi-container">'
            f'<div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Skor Akurasi Prediksi (R2 Score)</div>'
            f'<div style="font-size: 1.5rem; font-weight: 700; color: #4f46e5; margin-top: 0.25rem;">{best_metrics["R2"]:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_m2:
        st.markdown(
            f'<div class="kpi-container">'
            f'<div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Rata-Rata Margin Kesalahan (MAE)</div>'
            f'<div style="font-size: 1.5rem; font-weight: 700; color: #10b981; margin-top: 0.25rem;">{format_rupiah_compact(best_metrics["MAE"])}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_m3:
        st.markdown(
            f'<div class="kpi-container">'
            f'<div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Standar Deviasi Kesalahan (RMSE)</div>'
            f'<div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b; margin-top: 0.25rem;">{format_rupiah_compact(best_metrics["RMSE"])}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    st.write("---")
    st.subheader("Grafik Pengujian Backtest (Aktual vs Prediksi AI)")
    st.markdown(
        "Memvisualisasikan kemampuan model terpilih (**%s**) dalam merekonstruksi data aktual selama masa uji backtest." % best_name
    )
    
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(
        x=backtest["Date"],
        y=backtest["Aktual"],
        mode="lines",
        name="Aktual",
        line=dict(color="#4f46e5", width=2.5),
        hovertemplate="Tanggal: %{x|%d %b %Y}<br>Aktual: Rp %{y:,.0f}<extra></extra>"
    ))
    fig_backtest.add_trace(go.Scatter(
        x=backtest["Date"],
        y=backtest[best_name],
        mode="lines",
        name=f"Prediksi AI ({best_name})",
        line=dict(color="#10b981", width=2.5, dash="dash"),
        hovertemplate="Tanggal: %{x|%d %b %Y}<br>Prediksi: Rp %{y:,.0f}<extra></extra>"
    ))
    fig_backtest.update_layout(
        hovermode="x unified",
        xaxis_title="Tanggal",
        yaxis_title="Nilai Penjualan (Rp)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    )
    st.plotly_chart(fig_backtest, use_container_width=True)

# Tab 5: Data Explorer
with tab_data:
    st.subheader("Analisis Pembersihan Data (Before vs After Cleaning)")
    st.markdown(
        "Alur pembersihan data ini diadaptasi dari notebook `sales_prediction_bogor.ipynb`. "
        "AI melakukan konversi tipe data, menghapus baris dengan nilai kosong (null), "
        "dan menyaring outlier ekstrem (seperti kesalahan input nilai kuantitas/harga berlebih)."
    )
    
    # Calculate raw statistics
    raw_stats_df = raw_df.copy()
    raw_stats_df["Value"] = pd.to_numeric(raw_stats_df["Value"], errors="coerce")
    raw_stats_df["Qty"] = pd.to_numeric(raw_stats_df["Qty"], errors="coerce")
    
    raw_rows = len(raw_df)
    raw_total_val = raw_stats_df["Value"].sum()
    raw_total_qty = raw_stats_df["Qty"].sum()
    
    cleaned_rows = len(df)
    cleaned_total_val = df["Value"].sum()
    cleaned_total_qty = df["Qty"].sum()
    
    comparison_data = {
        "Metrik": ["Jumlah Baris (Rows)", "Total Nilai Penjualan (Value)", "Total Kuantitas (Qty)"],
        "Sebelum Pembersihan (Raw)": [
            f"{raw_rows:,}",
            format_rupiah_compact(raw_total_val),
            f"{raw_total_qty:,.0f}"
        ],
        "Setelah Pembersihan (Cleaned)": [
            f"{cleaned_rows:,}",
            format_rupiah_compact(cleaned_total_val),
            f"{cleaned_total_qty:,.0f}"
        ],
        "Selisih / Terbuang (Outliers/Null)": [
            f"{raw_rows - cleaned_rows:,}",
            format_rupiah_compact(raw_total_val - cleaned_total_val),
            f"{raw_total_qty - cleaned_total_qty:,.0f}"
        ]
    }
    comparison_df = pd.DataFrame(comparison_data).set_index("Metrik")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Show all data that was filtered out (Data Audit Trail)
    st.markdown("---")
    st.subheader("Data yang Terbuang & Dibersihkan (Data Audit Trail)")
    st.markdown(
        "Gunakan bagian di bawah untuk meninjau seluruh data yang dikeluarkan dari database training model. "
        "Daftar ini mencakup baris dengan data kosong (null), baris gagal parsing, maupun baris outlier."
    )
    
    # Selection filter for reason
    reason_options = [
        "Semua Data Terbuang",
        "Outlier saja",
        "Data Kosong saja"
    ]
    selected_reason = st.selectbox(
        "Saring Tampilan Berdasarkan Alasan Dihapus",
        options=reason_options,
        key="removed_reason_selectbox"
    )
    
    # Filter the df_removed dataframe based on selected reason
    df_removed_disp = df_removed.copy()
    if selected_reason == "Outlier saja":
        df_removed_disp = df_removed_disp[df_removed_disp["Alasan Dihapus"] == "Outlier"]
    elif selected_reason == "Data Kosong saja":
        df_removed_disp = df_removed_disp[df_removed_disp["Alasan Dihapus"] == "Data Kosong"]
        
    st.markdown(f"Total baris terfilter sesuai kriteria: **{len(df_removed_disp):,} baris**")
    
    if not df_removed_disp.empty:
        # Display the dataframe with clean columns
        disp_cols = ["Date", "Nama Store", "SKU", "Qty", "Value", "Alasan Dihapus"]
        existing_cols = [c for c in disp_cols if c in df_removed_disp.columns]
        df_removed_table = df_removed_disp[existing_cols].copy()
        
        # Safe formatting
        if "Value" in df_removed_table.columns:
            # Coerce to numeric, handle NaN/None values, format clean rupiah
            num_val = pd.to_numeric(df_removed_table["Value"], errors="coerce").fillna(0)
            df_removed_table["Value"] = num_val.apply(format_rupiah_compact)
        if "Qty" in df_removed_table.columns:
            num_qty = pd.to_numeric(df_removed_table["Qty"], errors="coerce").fillna(0)
            df_removed_table["Qty"] = num_qty.apply(lambda q: f"{q:,.0f}")
            
        st.dataframe(df_removed_table, use_container_width=True)
    else:
        st.info("Tidak ada data yang terbuang untuk filter kriteria ini.")
            
    st.write("---")
    st.subheader("Database Transaksi Mentah (Raw)")
    st.markdown("Menampilkan tabel seluruh baris data transaksi asli sebelum melalui proses pembersihan atau penyaringan.")
    st.dataframe(raw_df, use_container_width=True)
            
    st.write("---")
    st.subheader("Database Transaksi Terfilter (Cleaned)")
    st.markdown("Menampilkan tabel data transaksi yang telah dibersihkan dan disaring berdasarkan parameter pencarian.")
    st.dataframe(df_filtered, use_container_width=True)
    
    st.write("---")
    st.subheader("Data Harian Teragregasi (Input Fitur AI)")
    st.markdown("Representasi data teragregasi per hari yang digunakan sebagai parameter input fitur waktu/lag oleh AI.")
    st.dataframe(daily, use_container_width=True)

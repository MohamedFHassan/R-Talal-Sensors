import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as plotly_go
import plotly.express as px
from scipy.signal import detrend, find_peaks
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import io
import plotly.io as pio
# --- Page Config & State ---
st.set_page_config(page_title="R Talal Sensors - Professional Sensor Pipeline", layout="wide", initial_sidebar_state="expanded")

if "app_theme" not in st.session_state: st.session_state.app_theme = "Modern Clinical"

css_clinical = """
    /* Premium Aesthetic Dashboard Overrides */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Clean App Background */
    .stApp { background-color: #f8fafc; }
    
    /* Elegant Button Styling */
    div.stButton > button:first-child, div.stButton > button:first-child * {
        background-color: #2563eb; color: white !important; border-radius: 8px;
        border: none; font-weight: 600; transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); transform: translateY(-1px);
    }
    
    /* Primary Red Buttons (like Detect Peaks) */
    div.stButton > button[kind="primary"], div.stButton > button[kind="primary"] * {
        background-color: #ef4444; color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #dc2626;
    }

    /* Tab Design */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; white-space: pre-wrap; background-color: #ffffff;
        border-radius: 10px 10px 0px 0px; padding: 10px 24px;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1); border: 1px solid #e2e8f0; border-bottom: none;
        color: #64748b; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f9ff; border-top: 4px solid #0ea5e9;
        font-weight: 700; color: #0369a1;
    }
    
    /* Sidebar Aesthetics */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important; border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    
    /* Global Text Visibility */
    .stApp, .stApp * {
        color: #1e293b;
    }
    
    /* File Uploader override for clinical */
    [data-testid="stFileUploaderDropzone"] * {
        color: #1e293b !important;
    }
    /* Info Box Aesthetics */
    div[data-testid="stDocstring"] {
        border-radius: 10px;
    }
"""

css_neon = """
    /* Neon Dashboard Overrides */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif !important; 
    }
    
    /* Core Backgrounds */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { 
        background-color: #0b0f19 !important; 
    }
    
    /* Global Typography Fixes */
    h1, h2, h3, h4, h5, h6, p, span, li, label, .stMarkdown th, .stMarkdown td {
        color: #e2e8f0 !important;
    }

    /* Elegant Button Styling */
    div.stButton > button:first-child {
        background-color: #2e1065 !important; color: #d8b4fe !important; border-radius: 6px !important;
        border: 1px solid #6b21a8 !important; font-weight: 600 !important; transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #4c1d95 !important; color: white !important; transform: translateY(-1px) !important;
        box-shadow: 0 0 10px rgba(107, 33, 168, 0.5) !important;
    }
    
    /* Primary Red Buttons (like Detect Peaks) */
    div.stButton > button[kind="primary"] {
        background-color: #be185d !important; border: 1px solid #f43f5e !important; color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #e11d48 !important; box-shadow: 0 0 12px rgba(244, 63, 94, 0.6) !important;
    }

    /* Tab Design */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent !important; }
    .stTabs [data-baseweb="tab"] {
        background-color: #17172b !important; border: 1px solid #2e2e48 !important; border-bottom: none !important;
        border-radius: 8px 8px 0px 0px !important; color: #8F8CAE !important; font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e1e38 !important; border-top: 3px solid #00E5FF !important;
        color: #00E5FF !important; font-weight: 700 !important;
        box-shadow: 0 -4px 10px rgba(0, 229, 255, 0.15) !important;
    }
    
    /* Sidebar Aesthetics */
    section[data-testid="stSidebar"], [data-testid="stSidebarNav"] {
        background-color: #111122 !important; border-right: 1px solid #2e2e48 !important;
    }
    
    /* Info Box Aesthetics */
    div[data-testid="stDocstring"], div[data-testid="stExpander"] {
        border-radius: 8px !important; background-color: #1a1a2e !important; border: 1px solid #2e2e48 !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { background-color: #1a1a2e !important; color: #e2e8f0 !important; }
    [data-testid="stExpanderDetails"] p, [data-testid="stExpanderDetails"] strong { color: #cbd5e1 !important; }
    [data-testid="stAlert"] { background-color: #1a1a2e !important; border: 1px solid #6b21a8 !important; color: #e2e8f0 !important; }
    [data-testid="stAlert"] * { color: #e2e8f0 !important; }
    
    /* Inputs & Selectbox Dropdowns */
    input, div[data-baseweb="select"] > div { background-color: #1a1a2e !important; color: white !important; border: 1px solid #2e2e48 !important; }
    
    /* Out-of-tree Popovers (Selectboxes) */
    div[data-baseweb="popover"], div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { 
        background-color: #1a1a2e !important; color: #e2e8f0 !important; 
    }
    div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { 
        background-color: #2e1065 !important; 
    }
    
    /* File Uploader Fix */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #17172b !important; border: 2px dashed #6b21a8 !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #2e1065 !important; border: 1px solid #6b21a8 !important; color: white !important;
    }
    [data-testid="stFileUploaderDropzone"] button * {
        color: white !important;
    }
"""

css_forest = """
    /* Dark Forest Eco Overrides */
    @import url('https://fonts.googleapis.com/css2?family=Jura:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Jura', sans-serif !important; 
    }
    
    /* Core Backgrounds */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { 
        background-color: #1c262b !important; 
    }
    
    /* Global Typography Fixes */
    h1, h2, h3, h4, h5, h6, p, span, li, label, .stMarkdown th, .stMarkdown td {
        color: #Eaeaea !important;
    }
    
    /* Elegant Button Styling */
    div.stButton > button:first-child {
        background-color: #273830 !important; color: #a3b899 !important; border-radius: 4px !important;
        border: 1px solid #3c5445 !important; font-weight: 600 !important; transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #314a3d !important; color: white !important; transform: translateY(-1px) !important;
    }
    
    /* Primary Red Buttons (like Detect Peaks) */
    div.stButton > button[kind="primary"] {
        background-color: #a85d3b !important; border: 1px solid #cf7953 !important; color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #c9714b !important;
    }

    /* Tab Design */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background-color: transparent !important; }
    .stTabs [data-baseweb="tab"] {
        background-color: #222d33 !important; border: 1px solid #34434a !important; border-bottom: none !important;
        border-radius: 6px 6px 0px 0px !important; color: #8F9FA6 !important; font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b3940 !important; border-top: 3px solid #8AB07D !important;
        color: #8AB07D !important; font-weight: 700 !important;
    }
    
    /* Sidebar Aesthetics */
    section[data-testid="stSidebar"], [data-testid="stSidebarNav"] {
        background-color: #172024 !important; border-right: 1px solid #34434a !important;
    }
    
    /* Info Box Aesthetics */
    div[data-testid="stDocstring"], div[data-testid="stExpander"] {
        border-radius: 6px !important; background-color: #222d33 !important; border: 1px solid #34434a !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { background-color: #222d33 !important; color: #Eaeaea !important; }
    [data-testid="stExpanderDetails"] p, [data-testid="stExpanderDetails"] strong { color: #d1d5db !important; }
    [data-testid="stAlert"] { background-color: #222d33 !important; border: 1px solid #3c5445 !important; color: #Eaeaea !important; }
    [data-testid="stAlert"] * { color: #Eaeaea !important; }

    /* Inputs & Selectbox Dropdowns */
    input, div[data-baseweb="select"] > div { background-color: #222d33 !important; color: #EAE5D9 !important; border: 1px solid #34434a !important; }
    
    /* Out-of-tree Popovers (Selectboxes) */
    div[data-baseweb="popover"], div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { 
        background-color: #222d33 !important; color: #Eaeaea !important; 
    }
    div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { 
        background-color: #273830 !important; 
    }
    
    /* File Uploader Fix */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #222d33 !important; border: 2px dashed #8AB07D !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #Eaeaea !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #273830 !important; border: 1px solid #3c5445 !important; color: #Eaeaea !important;
    }
    [data-testid="stFileUploaderDropzone"] button * {
        color: #Eaeaea !important;
    }
"""

if st.session_state.app_theme == "Neon Cyberpunk":
    active_css = css_neon
elif st.session_state.app_theme == "Dark Eco Forest":
    active_css = css_forest
else:
    active_css = css_clinical

st.markdown(f"<style>\\n{active_css}\\n</style>", unsafe_allow_html=True)
if "file_id" not in st.session_state: st.session_state.file_id = None
if "raw_data" not in st.session_state: st.session_state.raw_data = None
if "master_peaks" not in st.session_state: st.session_state.master_peaks = pd.DataFrame()
if "settings_ma" not in st.session_state: st.session_state.settings_ma = {}
if "intervals_dict" not in st.session_state: st.session_state.intervals_dict = {}
if "temp_peaks" not in st.session_state: st.session_state.temp_peaks = []
if "current_analyte" not in st.session_state: st.session_state.current_analyte = None


PLOT_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'R_Talal_Plot_Export',
        'height': 800,
        'width': 1200,
        'scale': 3
    },
    'displaylogo': False
}

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

POLYMER_MAPPING = {
    "PVA+PEO": 1, "PVA+CA": 2, "PVA+PS": 3, "PVA+PAA": 4, "PVA+PVA": 5,
    "PAA+PEO": 6, "PAA+CA": 7, "PAA+PS": 8, "PAA+PAA": 9, "PAA+PVA": 10,
    "PS+PEO": 11, "PS+CA": 12, "PS+PS": 13, "PS+PAA": 14, "PS+PVA": 15,
    "CA+PEO": 16, "CA+CA": 17, "CA+PS": 18, "CA+PAA": 19, "CA+PVA": 20,
    "PEO+PEO": 21, "PEO+CA": 22, "PEO+PS": 23, "PEO+PAA": 24, "PEO+PVA": 25,
    "PEO": 26, "CA": 27, "PS": 28, "PAA": 29, "PVA": 30
}

# --- Core Logic Functions ---
def preprocess_sensor(y, window_size=10, apply_smoothing=True):
    if not apply_smoothing or window_size < 1:
        return y.copy()
    y_smooth = pd.Series(y).rolling(window=window_size, center=True).mean()
    y_smooth.bfill(inplace=True)
    y_smooth.ffill(inplace=True)
    return y_smooth.values

def detrend_sensor(y, apply_detrend=True):
    if not apply_detrend:
        return y.copy()
    y_no_inf = pd.Series(y).replace([np.inf, -np.inf], np.nan).ffill().bfill().values
    return detrend(y_no_inf, type='linear')

def normalize_sensor(y, norm="Baseline ((x-x0)/x0)"):
    if norm == "None":
        return y.copy()
    elif norm == "Baseline ((x-x0)/x0)":
        return (y - y[0]) / np.abs(y[0]) if y[0] != 0 else y.copy()
    elif norm == "Shift (x-x0)":
        return y - y[0]
    elif norm == "StandardScaler":
        return StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()
    elif norm == "MinMaxScaler":
        return MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()
    elif norm == "RobustScaler":
        return RobustScaler().fit_transform(y.reshape(-1, 1)).flatten()
    return y.copy()

def calculate_peak_heights_from_baseline(x, y, prominence=None, width=None):
    y = np.asarray(y)
    x = np.asarray(x)
    peaks, properties = find_peaks(y, prominence=prominence, width=width)
    peak_times = x[peaks]
    peak_heights = y[peaks]
    if len(peaks) > 0:
        left_bases = properties.get("left_bases", [0]*len(peaks))
        right_bases = properties.get("right_bases", [len(y)-1]*len(peaks))
    else:
        left_bases, right_bases = [], []
    return pd.DataFrame({"Time": peak_times, "Height": peak_heights}), left_bases, right_bases

def detect_single_sensor_peaks(x, y, analyte, sensor_name, intervals, prominence_value, width_value, is_valley):
    peak_storage = []
    x, y = np.asarray(x), np.asarray(y)
    
    for interval in intervals:
        conc, start_t, end_t = interval["Conc"], interval["Start"], interval["End"]
        mask = (x >= start_t) & (x <= end_t)
        if not np.any(mask): continue
            
        selected_x, selected_y = x[mask], y[mask]
        
        if is_valley:
            peak_data, left_bases, right_bases = calculate_peak_heights_from_baseline(selected_x, -selected_y, prominence=prominence_value, width=width_value)
            peak_data["Height"] *= -1
        else:
            peak_data, left_bases, right_bases = calculate_peak_heights_from_baseline(selected_x, selected_y, prominence=prominence_value, width=width_value)
            
        if len(peak_data) > 0:
            main_peak_idx = peak_data["Height"].idxmin() if is_valley else peak_data["Height"].idxmax()
            row = peak_data.iloc[main_peak_idx]
            peak_time, peak_value = row["Time"], row["Height"]
            lb_idx, rb_idx = left_bases[main_peak_idx], right_bases[main_peak_idx]
            tL, yL = selected_x[lb_idx], selected_y[lb_idx]
            tR, yR = selected_x[rb_idx], selected_y[rb_idx]
        else:
            main_peak_idx = selected_y.argmin() if is_valley else selected_y.argmax()
            peak_time = selected_x[main_peak_idx]
            peak_value = selected_y[main_peak_idx]
            tL, yL = selected_x[0], selected_y[0]
            tR, yR = selected_x[-1], selected_y[-1]
            
        interp_baseline = yL + (yR - yL) * ((peak_time - tL) / (tR - tL)) if tR != tL else yL
        signal_value = peak_value - interp_baseline
        p_index = POLYMER_MAPPING.get(sensor_name, 'Unknown')
        
        response_time = peak_time - tL
        recovery_time = tR - peak_time
        
        peak_storage.append({
            "Analyte": analyte,
            "Sensor": sensor_name,
            "Polymer Index": p_index,
            "Conc": int(conc) if pd.notna(conc) else None,
            "Signal": signal_value,
            "Response Time (s)": response_time,
            "Recovery Time (s)": recovery_time,
            "Prominence": prominence_value,
            "Width": width_value,
            "Peak Time": peak_time,
            "Peak Value": peak_value,
            "Interp. Baseline": interp_baseline,
            "tL": tL, "tR": tR, "yL": yL, "yR": yR,
            "Interval": f"{start_t}-{end_t}",
            "Start_T": start_t, "End_T": end_t
        })
    return peak_storage

def regression_analysis_grouped(data):
    regression_results, point_contributions = [], []
    if "Polymer Index" in data.columns and not data["Polymer Index"].isna().all():
        data = data.sort_values(by="Polymer Index")
        
    for (analyte, sensor, p_idx), group in data.groupby(["Analyte", "Sensor", "Polymer Index"]):
        if len(group) < 2: continue
        X_full = sm.add_constant(group["Conc"])
        y_full = group["Signal"]
        model_full = sm.OLS(y_full, X_full).fit()
        baseline_r2 = model_full.rsquared
        
        for i, row in group.iterrows():
            X_reduced, y_reduced = X_full.drop(i), y_full.drop(i)
            model_reduced = sm.OLS(y_reduced, X_reduced).fit()
            point_contributions.append({
                'Analyte': analyte, 'Sensor': sensor, 'Polymer Index': p_idx,
                'Conc': row["Conc"], 'Signal': row["Signal"],
                'Contribution_Score': baseline_r2 - model_reduced.rsquared
            })
            
        regression_results.append({
            'Analyte': analyte, 'Sensor': sensor, 'Polymer Index': p_idx,
            'R2': baseline_r2,
            'Std_Error': np.std(model_full.resid, ddof=1) if len(model_full.resid) > 1 else np.nan,
            'Slope': model_full.params.get("Conc", np.nan),
            'Intercept': model_full.params.get('const', np.nan)
        })
    return pd.DataFrame(regression_results), pd.DataFrame(point_contributions)


# --- MAIN UI ---
st.title("R Talal Sensors - Step-By-Step Pipeline")

st.sidebar.header("🎨 Aesthetics")
theme_choice = st.sidebar.selectbox("Dashboard Theme", ["Modern Clinical", "Neon Cyberpunk", "Dark Eco Forest"], index=["Modern Clinical", "Neon Cyberpunk", "Dark Eco Forest"].index(st.session_state.app_theme))

if theme_choice != "Modern Clinical":
    pio.templates.default = "plotly_dark"
else:
    pio.templates.default = "plotly_white"

if theme_choice != st.session_state.app_theme:
    st.session_state.app_theme = theme_choice
    st.rerun()

def custom_plotly_chart(fig, **kwargs):
    theme_name = st.session_state.app_theme
    if theme_name == "Neon Cyberpunk":
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19", font_color="#e2e8f0", title_font_color="#e2e8f0")
        fig.update_xaxes(gridcolor="#1e1e38", tickfont=dict(color="#e2e8f0"), title_font=dict(color="#e2e8f0"))
        fig.update_yaxes(gridcolor="#1e1e38", tickfont=dict(color="#e2e8f0"), title_font=dict(color="#e2e8f0"))
    elif theme_name == "Dark Eco Forest":
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1c262b", plot_bgcolor="#1c262b", font_color="#d1d5db", title_font_color="#d1d5db")
        fig.update_xaxes(gridcolor="#34434a", tickfont=dict(color="#d1d5db"), title_font=dict(color="#d1d5db"))
        fig.update_yaxes(gridcolor="#34434a", tickfont=dict(color="#d1d5db"), title_font=dict(color="#d1d5db"))
    else:
        fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#1e293b", title_font_color="#1e293b")
        fig.update_xaxes(gridcolor="#e2e8f0", tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b"))
        fig.update_yaxes(gridcolor="#e2e8f0", tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b"))
    
    kwargs["theme"] = None # Force override Streamlit's light theme blocker
    return st.plotly_chart(fig, **kwargs)

st.sidebar.markdown("---")
st.sidebar.header("📁 1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        is_excel = uploaded_file.name.endswith('.xlsx')
        sheet = None
        if is_excel:
            xls = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Analyte Sheet", xls.sheet_names)
            
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}_{sheet}"
        
        if st.session_state.file_id != current_file_id:
            with st.spinner("Loading dataset..."):
                if is_excel:
                    df = pd.read_excel(uploaded_file, sheet_name=sheet, header=0)
                    st.session_state.current_analyte = sheet
                else:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.current_analyte = "Unknown"
                st.session_state.raw_data = df
                st.session_state.temp_peaks = []
                st.session_state.file_id = current_file_id
                
        if st.session_state.raw_data is not None:
            st.sidebar.success(f"Loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

if st.session_state.raw_data is not None:
    df = st.session_state.raw_data
    analyte_name = st.session_state.current_analyte
    
    # Ensure complete numeric safety for Plotly JS, mitigating European comma strings 
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    time_columns = [col for col in df.columns if "Time" in str(col) or "sec" in str(col).lower()]
    raw_sensor_columns = [col for col in df.columns if col not in time_columns]
    time_col = time_columns[0]
    
    # Prune rows where time is physically unreadable
    df = df.dropna(subset=[time_col]).copy()
    
    time_data_full = df[time_col].values - df[time_col].min()
    
    # --- TEMPORARY DIAGNOSTIC VIEWER ---
    with st.sidebar.expander("Cloud Diagnostic Tool", expanded=True):
        st.write("DF Shape:", df.shape)
        st.write("Time Col:", time_col)
        st.write("First Sensor (PVA+CA):", df["PVA+CA"].dropna().values[0] if "PVA+CA" in df and len(df["PVA+CA"].dropna())>0 else "Empty")
        st.write("NaNs in Time:", df[time_col].isna().sum())
        st.write("NaNs in PVA+CA:", df["PVA+CA"].isna().sum() if "PVA+CA" in df else "N/A")
    # -----------------------------------
    
    def format_sensor(name):
        idx = POLYMER_MAPPING.get(name, "")
        return f"[{idx}] {name}" if idx else name
        
    sensor_name_map = {format_sensor(name): name for name in raw_sensor_columns}
    formatted_sensors = list(sensor_name_map.keys())

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 2. Sensor Selection")
    sel_mode = st.sidebar.radio("Selection Mode", ["Single Sensor", "Multiple Sensors", "Polymer Index Range (1-30)"])
    
    selected_sensors = []
    if sel_mode == "Single Sensor":
        sel = st.sidebar.selectbox("Target Sensor", formatted_sensors)
        selected_sensors = [sensor_name_map[sel]]
    elif sel_mode == "Multiple Sensors":
        sels = st.sidebar.multiselect("Target Sensors", formatted_sensors, default=formatted_sensors[:1])
        selected_sensors = [sensor_name_map[s] for s in sels]
    elif sel_mode == "Polymer Index Range (1-30)":
        r1, r2 = st.sidebar.slider("Select Index Range", 1, 30, (1, 6))
        for s_disp, s_raw in sensor_name_map.items():
            idx = POLYMER_MAPPING.get(s_raw, 0)
            if r1 <= idx <= r2:
                selected_sensors.append(s_raw)
                
    if not selected_sensors:
        st.warning("Please select at least one sensor from the sidebar to continue.")
        st.stop()
        
    st.sidebar.markdown("---")
    tc = st.sidebar.number_input("Global Time Cutoff (sec)", min_value=0.0, value=0.0)
    valid_idx = time_data_full >= tc
    t_vals = time_data_full[valid_idx]
    plot_stride = max(1, len(t_vals) // 1500)  # Cloud Optimization: Limit GUI points per line to ~1500 max
    t_plot = t_vals[::plot_stride]
    
    # Pre-pipeline state dictionaries
    pipeline_raw = {s: df[s].values[valid_idx] for s in selected_sensors}
    pipeline_res = {}
    pipeline_ma = {}
    pipeline_final = {}

    st.title("R Talal Sensors - Neural Sensor Pipeline")
    st.markdown("Welcome to the advanced chemical sensor processing suite. Process raw environmental data into clean mathematical insights by following the steps left-to-right.")
    
    tab_res, tab_ma, tab_det, tab_peak, tab_reg, tab_pca = st.tabs([
        "⚙️ 1. Resistance", "〰️ 2. Moving Average", "🧹 3. Detrend & Norm", "🔎 4. Peak Finder", "📈 5. Local/Global Regression", "🧠 6. PCA Analysis"
    ])

    # ---------------- TAB 1: RESISTANCE ----------------
    with tab_res:
        st.header("Step 1: Resistance Conversion")
        with st.expander("📚 Educational Walkthrough: Why do we convert Voltage to Resistance?", expanded=True):
            st.info('''
            **The Science:** 
            The raw data collected from Data Acquisition (DAQ) hardware is typically measured in **Electrical Voltage (V)**. 
            However, when chemical analytes interact with a polymer structure, they physically alter the material's **Electrical Resistance (R)**.
            
            **The Math:** 
            We use **Ohm's Law ($R = V / I$)**. By dividing the raw voltage array by the fixed current used in your specific laboratory experiment (e.g., $0.003$ Amperes), we seamlessly transform the noisy hardware voltage into the true underlying physical resistance of the sensor.
            ''')
            
        c_r1, c_r2 = st.columns([1,3])
        with c_r1:
            div_by_003 = st.checkbox("⚙️ Apply Voltage -> Resistance (/ 0.003 A)", value=True)
            
        for s in selected_sensors:
            pipeline_res[s] = pipeline_raw[s] / 0.003 if div_by_003 else pipeline_raw[s].copy()
            
        with c_r2:
            st.markdown("**Processed Resistance (After)**")
            fig_res_after = plotly_go.Figure()
            for s in selected_sensors:
                fig_res_after.add_trace(plotly_go.Scatter(x=t_plot, y=pipeline_res[s][::plot_stride], name=f"{s}", line=dict(width=2)))
            fig_res_after.update_layout(xaxis_title="Time (s)", yaxis_title="Calculated Resistance", height=350, margin=dict(l=0, r=0, t=10, b=0))
            custom_plotly_chart(fig_res_after, use_container_width=True, theme=None, config=PLOT_CONFIG)
            
            st.markdown("**Raw Voltage (Before)**")
            fig_res_before = plotly_go.Figure()
            for s in selected_sensors:
                fig_res_before.add_trace(plotly_go.Scatter(x=t_plot, y=pipeline_raw[s][::plot_stride], name=f"{s}", line=dict(dash='dot', width=1.5)))
            fig_res_before.update_layout(xaxis_title="Time (s)", yaxis_title="Raw Voltage", height=250, margin=dict(l=0, r=0, t=10, b=0))
            custom_plotly_chart(fig_res_before, use_container_width=True, theme=None, config=PLOT_CONFIG)

    # ---------------- TAB 2: MOVING AVERAGE ----------------
    with tab_ma:
        st.header("Step 2: Moving Average Smoothing")
        with st.expander("📚 Educational Walkthrough: Why do we smooth data?", expanded=True):
            st.info('''
            **The Science:**  
            Chemical sensors are highly sensitive to microscopic environmental changes. Tiny fluctuations in lab temperature, humidity, or unshielded electrical equipment instantly manifest as **"high-frequency noise"**—thick, jagged, vertical zig-zags on your graph.
            
            **The Math:**  
            A Moving Average acts like a mathematical shock-absorber. By setting a **Window Size (e.g., 10)**, the algorithm looks at 10 consecutive data points, averages them together into a single point, and rolls forward. This forcibly irons out the random, chaotic noise spikes while fully preserving the slow, overarching shape of the chemical exposure.
            ''')
            
        c_m1, c_m2 = st.columns([1,3])
        with c_m1:
            tune_sensor = st.selectbox("Select Sensor to Configure Moving Average", selected_sensors)
            
            current_settings = st.session_state.settings_ma.get(tune_sensor, {"apply": True, "window": 10})
            apply_smooth = st.checkbox(f"Enable Smoothing for {tune_sensor}", value=current_settings["apply"])
            win_size = st.slider(f"Window Size for {tune_sensor}", 1, 500, current_settings["window"]) if apply_smooth else 1
            
            if st.button(f"Save Settings for {tune_sensor}", type="primary"):
                st.session_state.settings_ma[tune_sensor] = {"apply": apply_smooth, "window": win_size}
                st.rerun()
                
            st.markdown("---")
            if st.button("Apply Current Window to ALL Active Sensors"):
                for s in selected_sensors:
                    st.session_state.settings_ma[s] = {"apply": apply_smooth, "window": win_size}
                st.rerun()
            
        for s in selected_sensors:
            settings = st.session_state.settings_ma.get(s, {"apply": True, "window": 10})
            pipeline_ma[s] = preprocess_sensor(pipeline_res[s], window_size=settings["window"], apply_smoothing=settings["apply"])

        with c_m2:
            st.markdown("**Smoothed Signal (After)**")
            fig_ma_after = plotly_go.Figure()
            for s in selected_sensors:
                fig_ma_after.add_trace(plotly_go.Scatter(x=t_plot, y=pipeline_ma[s][::plot_stride], name=f"{s}", line=dict(width=2)))
            fig_ma_after.update_layout(xaxis_title="Time (s)", yaxis_title="Smoothed Response", height=350, margin=dict(l=0, r=0, t=10, b=0))
            custom_plotly_chart(fig_ma_after, use_container_width=True, theme=None, config=PLOT_CONFIG)

            st.markdown("**Unsmoothed Resistance (Before)**")
            fig_ma_before = plotly_go.Figure()
            for s in selected_sensors:
                fig_ma_before.add_trace(plotly_go.Scatter(x=t_plot, y=pipeline_res[s][::plot_stride], name=f"{s}", line=dict(dash='dot', width=1.5)))
            fig_ma_before.update_layout(xaxis_title="Time (s)", yaxis_title="Calculated Resistance", height=250, margin=dict(l=0, r=0, t=10, b=0))
            custom_plotly_chart(fig_ma_before, use_container_width=True, theme=None, config=PLOT_CONFIG)

    # ---------------- TAB 3: DETRENDING & NORMALIZATION ----------------
    with tab_det:
        st.header("Step 3: Detrending & Signal Normalization")
        with st.expander("📚 Educational Walkthrough: Why Detrend and Normalize?", expanded=True):
            st.info('''
            **1. Detrending (Correcting Drift):**  
            Over long experiments, polymers suffer from baseline drift. Their natural resting resistance slowly climbs up or down constantly due to aging or accumulating trace vapors. **Linear detrending** calculates the mathematical slope of the entire experiment and forcefully flattens it out so the quiet baseline is perfectly horizontal.
            
            **2. Normalization (Creating a Fair Scale):**  
            Every single sensor has a totally different resting resistance (e.g., Sensor A = 50 Ohms, Sensor B = 1,400 Ohms). If you plotted them directly, Sensor B would visually crush Sensor A. **Baseline Normalization $((R - R_0) / R_0)$** mathematically forces every sensor to start exactly at **$0.0$**, converting their massive raw resistance numbers into fair, standardized percentages.
            ''')
            
        c_d1, c_d2 = st.columns([1,3])
        with c_d1:
            apply_detrend = st.checkbox("🧹 Apply Linear Detrending", value=True)
            norm_sel = st.selectbox("⚖️ Normalization Pattern", ["None", "Baseline ((x-x0)/x0)", "Shift (x-x0)", "StandardScaler", "MinMaxScaler", "RobustScaler"], index=1)
            show_ghosts = st.checkbox("👁️ Overlay Pre-Normalized Signals", value=False, help="Normalization radically alters the Y-Axis scale. Turning this on will zoom out significantly to fit both, flattening the visual of the final signal.")
            
        for s in selected_sensors:
            y_det = detrend_sensor(pipeline_ma[s], apply_detrend=apply_detrend)
            pipeline_final[s] = normalize_sensor(y_det, norm=norm_sel)

        with c_d2:
            st.markdown("**Final Scaled Signal (After)**")
            fig_det_after = plotly_go.Figure()
            for s in selected_sensors:
                fig_det_after.add_trace(plotly_go.Scatter(x=t_plot, y=pipeline_final[s][::plot_stride], name=f"{s}", line=dict(width=2)))
            fig_det_after.update_layout(xaxis_title="Time (s)", yaxis_title="Normalized Output", height=350, margin=dict(l=0, r=0, t=10, b=0))
            custom_plotly_chart(fig_det_after, use_container_width=True, theme=None, config=PLOT_CONFIG)

            st.markdown("**Pre-Normalization Signal (Before)**")
            fig_det_before = plotly_go.Figure()
            for s in selected_sensors:
                fig_det_before.add_trace(plotly_go.Scatter(x=t_plot, y=pipeline_ma[s][::plot_stride], name=f"{s}", line=dict(dash='dot', width=1.5)))
            fig_det_before.update_layout(xaxis_title="Time (s)", yaxis_title="Smoothed Response", height=250, margin=dict(l=0, r=0, t=10, b=0))
            custom_plotly_chart(fig_det_before, use_container_width=True, theme=None, config=PLOT_CONFIG)

    # ---------------- TAB 4: PEAK FINDER ----------------
    with tab_peak:
        st.header("Step 4: Interactive Peak Finder")
        with st.expander("📚 Educational Walkthrough: Extracting Peak Metrics", expanded=True):
            st.info('''
            **The Goal:** We need to capture the exact mathematical shape of how the sensor reacted when the chemical valve was opened.
            
            **How to do it:** 
            1. Drag your mouse inside the graph below to vividly highlight the timeframe where the chemical was exposed. Click **➕ Add Region Box**.
            2. Set the exact **Target Conc (ppm)** associated with that exposure.
            3. Click the red **🔎 Detect Peaks** button. 
            
            The Deep Engine will mathematically iterate through every active sensor inside that timeframe. It will dynamically search for a peak (red dot), simulate what a "quiet" baseline beneath it should look like (dotted line), and extract the exact **Response Time** (time to reach peak maximum) and **Recovery Time** (time to fall back to equilibrium).
            ''')

        col_pf1, col_pf2 = st.columns([1, 2.5])
        
        # Interval state tracking specific to the current selection of sensors grouped
        dict_key = str(hash(tuple(sorted(selected_sensors))))
        if dict_key not in st.session_state.intervals_dict:
            st.session_state.intervals_dict[dict_key] = []
        intervals = st.session_state.intervals_dict[dict_key]
        
        with col_pf1:
            st.subheader("Peak Analytics Criteria")
            is_valley = st.checkbox("Valley Sensors (inverted peak target)?", value=False)
            prominence = st.number_input("Mathematical Prominence", value=0.04, step=0.01)
            width = st.number_input("Mathematical Peak Width", value=5, step=1)
            
            def clear_temp(): st.session_state.temp_peaks = []
            
            st.markdown("---")
            c_det, c_save = st.columns([1,1])
            with c_det:
                if st.button("🔎 Detect Peaks", type="primary", use_container_width=True, on_click=clear_temp):
                    if len(intervals) == 0:
                        st.error("Please add Concentration Blocks on the right first.")
                    else:
                        with st.spinner("Calculating Peaks via SciPy..."):
                            all_new_peaks = []
                            for s in selected_sensors:
                                p_res = detect_single_sensor_peaks(
                                    t_vals, pipeline_final[s], analyte_name, s, 
                                    intervals, prominence, width, is_valley
                                )
                                all_new_peaks.extend(p_res)
                            
                            st.session_state.temp_peaks = all_new_peaks
            
            with c_save:
                if len(st.session_state.temp_peaks) > 0:
                    if st.button("✔️ Save to Global DB", use_container_width=True):
                        new_df = pd.DataFrame(st.session_state.temp_peaks)
                        db = st.session_state.master_peaks
                        if not db.empty: # Remove overlapping old records
                            cond = ~((db['Sensor'].isin(selected_sensors)) & (db['Analyte'] == analyte_name))
                            db = db[cond]
                        st.session_state.master_peaks = pd.concat([db, new_df], ignore_index=True)
                        st.session_state.temp_peaks = []
                        st.success(f"Saved {len(new_df)} peak entries!")

        with col_pf2:
            # Interactive Graphing Engine
            fig_p = plotly_go.Figure()
            colors = ["#2CA02C", "#FF7F0E", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
            
            if len(st.session_state.temp_peaks) > 0:
                # Ghost out inactive lines outside interval blocks
                for s in selected_sensors:
                    fig_p.add_trace(plotly_go.Scatter(x=t_plot, y=pipeline_final[s][::plot_stride], mode='lines', line=dict(color='lightgray', width=1), showlegend=False))
                
                # Draw colored overlay blocks
                for i, inter in enumerate(intervals):
                    c_color = colors[i % len(colors)]
                    mask = (t_vals >= inter["Start"]) & (t_vals <= inter["End"])
                    fig_p.add_vrect(x0=inter["Start"], x1=inter["End"], fillcolor=c_color, opacity=0.05, line_width=1, line_color=c_color)
                    
                    # Highlight active line segment
                    for s in selected_sensors:
                        fig_p.add_trace(plotly_go.Scatter(x=t_vals[mask], y=pipeline_final[s][mask], mode='lines', line=dict(color=c_color, width=2), name=f"{s} {inter.get('Conc')}ppm"))

                # Draw intricate peak math structures
                for p in st.session_state.temp_peaks:
                    # Dashed Line (Base to Base)
                    fig_p.add_trace(plotly_go.Scatter(x=[p["tL"], p["tR"]], y=[p["yL"], p["yR"]], mode='lines', line=dict(color='black', dash='dash'), showlegend=False))
                    # Dotted Line (Interp Base to Peak)
                    fig_p.add_trace(plotly_go.Scatter(x=[p["Peak Time"], p["Peak Time"]], y=[p["Interp. Baseline"], p["Peak Value"]], mode='lines', line=dict(color='blue', dash='dot'), showlegend=False))
                    # Red Intercept Dot
                    fig_p.add_trace(plotly_go.Scatter(x=[p["Peak Time"]], y=[p["Peak Value"]], mode='markers', marker=dict(color='red', size=6, line=dict(color='black', width=1)), showlegend=False))
                    # Stamped Number Text
                    fig_p.add_annotation(
                        x=p["Peak Time"], y=p["Interp. Baseline"] if is_valley else p["Peak Value"],
                        text=f"<b>{p['Signal']:.2f}</b>", showarrow=False, font=dict(size=14, color="black"), yshift=15 if is_valley else 15
                    )
            
            else:
                # Standard Explore mode
                for s in selected_sensors:
                    fig_p.add_trace(plotly_go.Scatter(x=t_plot, y=pipeline_final[s][::plot_stride], mode='lines', line=dict(width=2), name=s))
                for i, inter in enumerate(intervals):
                    c_color = colors[i % len(colors)]
                    fig_p.add_vrect(x0=inter["Start"], x1=inter["End"], annotation_text=f"{inter.get('Conc')} ppm", annotation_position="top left", fillcolor=c_color, opacity=0.15, line_width=1, line_color=c_color)

            fig_p.update_layout(xaxis_title="Time (sec)", yaxis_title="Final Signal Output", height=450, margin=dict(l=0, r=0, t=10, b=0), dragmode="select")
            
            # Sub-Selection Slider bounds protection
            s_min, s_max = float(t_vals.min()), float(t_vals.max())
            if np.isnan(s_min): s_min = 0.0
            if np.isnan(s_max) or s_max <= s_min: s_max = s_min + 1.0
            s_min_val, s_max_val = s_min, s_max 
            
            event = custom_plotly_chart(fig_p, use_container_width=True, on_select="rerun", selection_mode="box", config=PLOT_CONFIG, theme=None, key=f"plot_{dict_key}")
            
            try:
                # Handle Streamlit <= 1.34 (dict-like) and >= 1.35 (object-like) events
                box_data = None
                if isinstance(event, dict): 
                    box_data = event.get("selection", {}).get("box")
                elif hasattr(event, "selection"):
                    sel = event.selection
                    box_data = sel.get("box") if isinstance(sel, dict) else getattr(sel, "box", None)
                
                if box_data and len(box_data) > 0:
                    box_item = box_data[0]
                    val = box_item.get("x") if isinstance(box_item, dict) else getattr(box_item, "x", None)
                    if val and len(val) == 2:
                        val_min, val_max = min(val[0], val[1]), max(val[0], val[1])
                        if not np.isnan(val_min) and not np.isnan(val_max) and val_min < val_max:
                            s_min_val, s_max_val = val_min, val_max
            except Exception:
                pass

            cc1, cc2, cc3 = st.columns([1.5, 3, 1.5])
            with cc1: new_conc = st.number_input("Target Conc (ppm)", value=20, step=10, key="new_conc")
            with cc2: block_slider = st.slider("Select Region Limits", min_value=s_min, max_value=s_max, value=(s_min_val, s_max_val), step=1.0, label_visibility="collapsed")
            with cc3:
                if st.button("➕ Add Region Box", use_container_width=True):
                    st.session_state.intervals_dict[dict_key].append({"Conc": new_conc, "Start": block_slider[0], "End": block_slider[1]})
                    clear_temp(); st.rerun()

            st.write("Current Mathematical Exposure Regions:")
            if len(intervals) > 0:
                edited_intervals = st.data_editor(intervals, num_rows="dynamic", use_container_width=True, key=f"table_{dict_key}")
                
                # Float tolerance checking to prevent UI recursion loops
                changed = len(edited_intervals) != len(intervals)
                if not changed:
                    for e_int, o_int in zip(edited_intervals, intervals):
                        if e_int.get("Conc") != o_int.get("Conc") or \
                           abs(float(e_int.get("Start", 0)) - float(o_int.get("Start", 0))) > 1e-3 or \
                           abs(float(e_int.get("End", 0)) - float(o_int.get("End", 0))) > 1e-3:
                            changed = True; break

                if changed:
                    st.session_state.intervals_dict[dict_key] = edited_intervals
                    clear_temp(); st.rerun()
                    
        if len(st.session_state.temp_peaks) > 0:
            st.markdown("---")
            st.subheader("Extracted Peak Analytics (Preview)")
            preview_df = pd.DataFrame(st.session_state.temp_peaks)
            preview_clean = preview_df.drop(columns=["tL", "tR", "yL", "yR"], errors='ignore')
            st.dataframe(preview_clean, use_container_width=True)
            st.download_button("📥 Export Preview to CSV", data=convert_df(preview_clean), file_name="peak_preview.csv", mime="text/csv")

            st.write("**Live Local Regression Preview:**")
            try:
                fig_preview_reg = px.scatter(preview_df, x="Conc", y="Signal", color="Sensor", trendline="ols", 
                                             title="Real-Time Sensitivity Trend Lines (Unsaved)")
                fig_preview_reg.update_layout(xaxis_title="Concentration Exposure (ppm)", yaxis_title="Mathematical Signal (Peak - Baseline)", height=400)
                custom_plotly_chart(fig_preview_reg, use_container_width=True, theme=None, config=PLOT_CONFIG)
            except Exception as e:
                st.warning(f"Live trendline preview failed (likely requires at least 2 concentration points per sensor): {e}")
    # ---------------- TAB 5: REGRESSION & DATABASE ----------------
    with tab_reg:
        st.header("Step 5: Database Regression & Trend Lines")
        with st.expander("📚 Educational Walkthrough: What is Regression?", expanded=True):
            st.info('''
            **The Science:**  
            A high-quality gas sensor should have a predictable, purely mathematical relationship between the **Target Gas Concentration (ppm)** and its **Output Signal Height**. For example: if you double the gas limits in the room, the sensor's peak signal should ideally double as well.
            
            **The Math:**  
            By plotting our extracted Peaks from Step 4 onto a Scatter grid, we can draw a **Line of Best Fit (Linear Regression)** straight through the dots. 
            - A **steep Slope** means the sensor is highly sensitive to the gas.
            - An **$R^2$ Score close to 1.0** heavily implies the sensor is perfectly stable and trustworthy. If the gas concentration is unknown, you can mathematically reverse the formula off the line to predict exactly how much gas was present.
            ''')
            
        db = st.session_state.master_peaks
        
        if db.empty:
            st.info("The peak database operates from globally confirmed peaks. Detect and Confirm peaks in the previous tab to populate!")
        else:
            db_sel = db[db["Sensor"].isin(selected_sensors)]
            
            if not db_sel.empty:
                st.subheader("Interactive Local Sensitivities (Selected Pool)")
                try: 
                    # Use Plotly Express built-in Ordinary Least Squares
                    fig_reg = px.scatter(db_sel, x="Conc", y="Signal", color="Sensor", trendline="ols", 
                                        title=f"Scatter Trend Lines ({len(selected_sensors)} Local Active Sensors)")
                    fig_reg.update_layout(xaxis_title="Concentration Exposure (ppm)", yaxis_title="Mathematical Signal (Peak - Baseline)")
                    custom_plotly_chart(fig_reg, use_container_width=True, theme=None, config=PLOT_CONFIG)
                except Exception as e:
                    st.warning(f"Trendline failed (requires at least 2 points per sensor): {e}")

            st.markdown("---")
            c_g1, c_g2 = st.columns([4, 1])
            with c_g1: st.subheader("Global Process Database")
            with c_g2: 
                if st.button("⚠️ Clear Entire Database", type="secondary"): 
                    st.session_state.master_peaks = pd.DataFrame()
                    st.rerun()
            
            st.dataframe(db.drop(columns=["tL", "tR", "yL", "yR"], errors='ignore'), use_container_width=True)
            
            if st.button("📈 Run Deep Global Regression Engine", type="primary"):
                reg_results, pt_contribs = regression_analysis_grouped(db)
                
                st.subheader("Analyte + Sensor Sensitivity Metrics")
                st.dataframe(reg_results, use_container_width=True)
                
                st.subheader("Data Export Ready")
                st.markdown("Download deeply structured regression points and coefficients locally.")
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    db.drop(columns=["tL", "tR", "yL", "yR"], errors='ignore').to_excel(writer, sheet_name='Peaks Data', index=False)
                    reg_results.to_excel(writer, sheet_name='Regression Metrics', index=False)
                    pt_contribs.to_excel(writer, sheet_name='Point Contributions', index=False)
                
                st.download_button(
                    label="Download Full Excel Output (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=f"R_Talal_Output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # ---------------- TAB 6: PCA ANALYSIS ----------------
    with tab_pca:
        st.header("Step 6: Principal Component Analysis (PCA)")
        with st.expander("📚 Educational Walkthrough: How does PCA cluster chemicals?", expanded=True):
            st.info('''
            **The Science:**  
            Sensors are notoriously indiscriminate. It is incredibly difficult to build a single sensor that only reacts to Acetone, but ignores Ethanol. To circumvent this, we use an **Array** (firing 6 or 12 distinct sensors simultaneously). This vast array generates a very unique, complex mathematical "Fingerprint" for every single gas.
            
            **The Math:**  
            Because humans cannot visualize a 12-Dimensional space (12 sensors), we run **PCA (Principal Component Analysis)**. PCA mathematically crushes down the useless overlapping noise and compresses the most highly variable parts of the 12-Sensor array matrix directly down onto a flat **2D Screen (PC1 vs PC2)**.
            
            **The Result:**  
            If your sensor array is a success, the AI will naturally draw distinct, separate "islands" (clusters). Acetone dots will cluster forcefully to the left, while Ethanol clusters perfectly to the right.
            ''')
            
        db = st.session_state.master_peaks
        if db.empty:
            st.info("The peak database operates from globally confirmed peaks. Detect and Confirm peaks in Step 4 to populate the PCA engine!")
        else:
            st.markdown("### Feature Matrix Configuration")
            st.info("Filter out any dead sensors or erratic concentrations globally to prevent them from destabilizing the cluster calculations.")
            
            avail_sensors = sorted(db["Sensor"].unique())
            avail_concs = sorted(db["Conc"].dropna().unique())
            
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                pca_sensors = st.multiselect("Target PCA Features (Sensors)", avail_sensors, default=avail_sensors, help="Sensors act as the independent dimensional vectors for PCA.")
                drop_na_toggle = st.checkbox("Strict Constraint: Drop missing data point", value=True, help="If unchecked, any missing peak signals are gracefully forced to 0.0 (non-responsive) instead of completely deleting the entire row from the matrix.")
            with c_f2:
                pca_concs = st.multiselect("Target Exposures (Concentration block)", avail_concs, default=avail_concs, help="Filters the total amount of samples (dots) plotted on the graph.")
                
            if len(pca_sensors) < 2:
                st.warning("PCA mathematically requires at least 2 sensors properties (Features) to project a map.")
            elif len(pca_concs) < 1:
                st.warning("Please select at least 1 concentration value.")
            else:
                pca_df = db[(db["Sensor"].isin(pca_sensors)) & (db["Conc"].isin(pca_concs))].copy()
                
                # Pivot data: Rows = Analyte + Concentration, Columns = Sensors, Values = Peak Signal
                pca_pivot = pca_df.pivot_table(index=["Analyte", "Conc"], columns="Sensor", values="Signal", aggfunc="mean").reset_index()
                
                # Handle Non-Response Overlaps
                pca_clean = pca_pivot.dropna() if drop_na_toggle else pca_pivot.fillna(0)
                
                if len(pca_clean) < 3 or len(pca_clean.columns) < 4:
                    st.warning("Not enough overlapping Sensor data for PCA. Try turning off 'Strict Constraint' to keep partial signal structures, or explicitly un-select the 'dead' sensors from the dropdown.")
                    st.dataframe(pca_pivot)
                else:
                    features = [col for col in pca_clean.columns if col not in ["Analyte", "Conc"]]
                    X = pca_clean[features]
                    
                    # Standardize features before applying dimensional reduction
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(X_scaled)
                    
                    pca_clean["PC1"] = components[:, 0]
                    pca_clean["PC2"] = components[:, 1]
                    
                    var_ratio = pca.explained_variance_ratio_ * 100
                    
                    fig_pca = px.scatter(
                    pca_clean, x="PC1", y="PC2", color="Analyte", text="Conc",
                    title="PCA Cluster Map (Sensors as Independent Features)",
                    labels={
                        "PC1": f"Principal Component 1 ({var_ratio[0]:.1f}%)",
                        "PC2": f"Principal Component 2 ({var_ratio[1]:.1f}%)"
                    }, 
                    height=500
                )
                
                fig_pca.update_traces(textposition='top center', marker=dict(size=10))
                custom_plotly_chart(fig_pca, use_container_width=True, theme=None, config=PLOT_CONFIG)
                
                st.markdown("---")
                c_pca1, c_pca2 = st.columns([2, 1])
                with c_pca1:
                    st.write("**Pivot Matrix ingested by PCA:**")
                    pca_df_out = pca_clean.drop(columns=["PC1", "PC2"])
                    st.dataframe(pca_df_out, use_container_width=True)
                    st.download_button("📥 Export PCA Matrix", data=convert_df(pca_df_out), file_name="pca_matrix.csv", mime="text/csv")
                with c_pca2:
                    st.write("**Explained Variance Captured:**")
                    st.write(f"- PC1 captures **{var_ratio[0]:.2f}%** of total variance")
                    st.write(f"- PC2 captures **{var_ratio[1]:.2f}%** of total variance")
                    st.write(f"- Total: **{(var_ratio[0] + var_ratio[1]):.2f}%**")
                    
                    st.write("**Component Matrix Loadings:**")
                    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
                    st.dataframe(loadings, use_container_width=True)
                    st.download_button("📥 Export Matrix Loadings", data=convert_df(loadings), file_name="pca_loadings.csv", mime="text/csv")

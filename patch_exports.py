import re

with open("app.py", "r") as f:
    code = f.read()

config_code = """
PLOT_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'Jarvis_Rania_Plot_Export',
        'height': 800,
        'width': 1200,
        'scale': 3
    },
    'displaylogo': False
}

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')
"""
if "PLOT_CONFIG =" not in code:
    code = code.replace("POLYMER_MAPPING = {", config_code + "\nPOLYMER_MAPPING = {")

# Regex to safely inject config=PLOT_CONFIG
code = re.sub(r'(st\.plotly_chart\([a-zA-Z0-9_]+, use_container_width=True)\)', r'\1, config=PLOT_CONFIG)', code)
code = re.sub(r'(st\.plotly_chart\(fig_p, use_container_width=True, on_select="rerun", selection_mode="box")(, key=.*?\))', r'\1, config=PLOT_CONFIG\2', code)

# Previews 
code = code.replace(
    'st.dataframe(preview_df.drop(columns=["tL", "tR", "yL", "yR"], errors=\'ignore\'), use_container_width=True)',
    'preview_clean = preview_df.drop(columns=["tL", "tR", "yL", "yR"], errors=\'ignore\')\n            st.dataframe(preview_clean, use_container_width=True)\n            st.download_button("📥 Export Preview to CSV", data=convert_df(preview_clean), file_name="peak_preview.csv", mime="text/csv")'
)

# PCA Loadings
code = code.replace(
    'st.dataframe(loadings, use_container_width=True)',
    'st.dataframe(loadings, use_container_width=True)\n                    st.download_button("📥 Export Matrix Loadings", data=convert_df(loadings), file_name="pca_loadings.csv", mime="text/csv")'
)

# PCA Pivot
code = code.replace(
    'st.dataframe(pca_clean.drop(columns=["PC1", "PC2"]), use_container_width=True)',
    'pca_df_out = pca_clean.drop(columns=["PC1", "PC2"])\n                    st.dataframe(pca_df_out, use_container_width=True)\n                    st.download_button("📥 Export PCA Matrix", data=convert_df(pca_df_out), file_name="pca_matrix.csv", mime="text/csv")'
)

with open("app.py", "w") as f:
    f.write(code)

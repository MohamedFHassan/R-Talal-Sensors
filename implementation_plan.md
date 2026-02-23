# Implementation Plan: Jarvis Rania Sensor Analysis App

## 1. Project Overview
**Jarvis Rania** will be a graphical application designed to automate, visualize, and interactively manage the data processing, peak detection, and regression analysis of robust polymer-based gas sensors. 

It aims to replace the rigid Jupyter Notebook pipeline with a flexible UI where the user can:
- Dynamically adjust noise filtering and preprocessing.
- Interactively define start and end times for exposure peaks across different concentrations.
- Quickly visualize and export regression results.

## 2. Technology Stack
- **Language:** Python 3.9+
- **GUI Framework:** PyQt6 or CustomTkinter (for a standalone desktop app) OR Streamlit (for a browser-based local web app). *Recommendation: Streamlit combined with Plotly for seamless interactive graphs and data tables.*
- **Data Processing:** Pandas, NumPy
- **Analysis:** SciPy (`find_peaks`, `detrend`), scikit-learn (Scalers), statsmodels (OLS Regression)
- **Visualization:** Plotly (Highly recommended for interactive dragging/selecting regions of interest) or Matplotlib.

## 3. Core Features & Requirements

### 3.1. Data Ingestion & State Management
- File uploader accepting `.csv` and `.xlsx` files.
- Ability to parse available sheets and select the target gas (e.g., Toluene, Acetone).
- Automated separation of "Time" columns and "Sensor" columns.
- State management across the App keeping track of the original raw data vs. the actively transformed data.

### 3.2. Dynamic Preprocessing Module
Given the noisiness of the data, the user needs full immediate control over the cleaning steps:
- **Smoothing Filter:** A slider or input field to toggle a Moving Average filter and customize the `window_size`.
- **Detrending:** A toggle switch to apply linear background subtraction.
- **Normalization:** A dropdown to instantly switch between different normalization protocols (Baseline normalization, absolute shift, Z-score, Min/Max, Robust scaling, or None).

### 3.3. Interactive Peak & Interval Assignment (Critical Feature)
Since the timing for each peak in each exposure varies:
- Render an interactive plot (e.g., using Plotly) displaying the preprocessed sensor signals.
- **Interval Controls:** A dedicated UI menu where the user can manually input or interactively drag ranges (sliders/boxes) on the graph to set the exact `[Start Time, End Time]` for each concentration block (e.g., 20ppm = 0s to 130s).
- **Peak Settings:** Sliders/Inputs for `prominence` and `width` localized to the peak detection algorithm. 
- The app will compute the local interpolated baseline and identify the proper peak immediately upon interval adjustments.

### 3.4. Regression Analytics Dashboard
- Once intervals and peaks are defined, a button to "Run Regression".
- **Visuals:** Scatter plots of Concentration vs. Signal (Peak Height) alongside the OLS fitted line, displaying the equation and $R^2$ score.
- **Exclusion/Inclusion:** Checkboxes or toggles allowing users to disable specific "bad" sensors or mark sensors as "Valley" sensors specifically so their peaks are calculated inverted.

### 3.5. Export Module
- A button to export all cleaned output streams: Peak Data, Regression Results, and Point Contributions, separated into neatly structured Excel spreadhseets.

## 4. Execution Steps

### Phase 1: Setup and Basic UI
- Initialize the project structure.
- Build the basic window layout (Sidebar for controls, Main panel for graphs).
- Implement the File Uploading logic and basic raw data plotting.

### Phase 2: Preprocessing Engine
- Port the `preprocess_all_sensors`, `detrend_all_sensors`, and `normalize_all_sensors` functions from the notebook.
- Connect these functions to UI toggles (Smoothing, Detrending, Normalizing dropdown) so the graph updates in real time.

### Phase 3: Interactive Peak Mapping
- Port the peak calculation logic (`calculate_peak_heights_from_baseline`, `detect_and_store_peaks`).
- Build the visual Time-Mapping interface. Allow users to add/remove Concentration intervals dynamically.
- Render vertical bounding lines on the active plot so the user geographically understands where the algorithm is currently scanning for a peak.

### Phase 4: Regression & Output
- Port the `regression_analysis_grouped` function.
- Create a dedicated "Results" tab in the UI that displays the summary metrics (Slope, $R^2$, Error).
- Implement the Excel writing logic utilizing `pd.DataFrame.to_excel()`.

## 5. Next Steps
Once approved, we will choose the specific GUI framework (e.g., Streamlit vs. PyQt) and begin structuring the main application file (e.g., `app.py`) inside this directory!

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------- Constants ---------------------
EXCEL_FILE = "24-hour_Refrigerator_energy_data.xlsx"
REFRESH_INTERVAL = 2000  # milliseconds

# --------------------- Load and Process Interpolated Data ---------------------
@st.cache_data
def load_and_process_data(file):
    df = pd.read_excel(file, header=1)
    df = df.dropna(subset=["Time (Hour)"])
    df["Time"] = pd.to_datetime(df["Time (Hour)"], format="%H:%M", errors="coerce")
    df["Time (Min)"] = df["Time"].dt.hour * 60 + df["Time"].dt.minute
    df = df.sort_values("Time (Min)")
    df["Energy (kWh)"] = df["Energy (kWh)"].fillna(method="ffill").fillna(method="bfill")

    df_5min = pd.DataFrame({"Time (Min)": np.arange(0, 24 * 60, 5)})

    for col in ["Power (W)", "Current (mA)", "Voltage (V)"]:
        df_5min[col] = np.interp(
            df_5min["Time (Min)"],
            df["Time (Min)"],
            df[col],
            left=df[col].iloc[0],
            right=df[col].iloc[-1],
        )

    df_5min["Energy (kWh)"] = np.interp(
        df_5min["Time (Min)"],
        df["Time (Min)"],
        df["Energy (kWh)"],
        left=df["Energy (kWh)"].iloc[0],
        right=df["Energy (kWh)"].iloc[-1],
    )

    # Calculate cumulative energy as sum of Energy (kWh) over time
    df_5min["Total Energy (Cumulative)"] = df_5min["Energy (kWh)"].cumsum()

    df_5min["DP ID"] = np.arange(1000, 1000 + len(df_5min))
    df_5min["Hour"] = (df_5min["Time (Min)"] // 60).astype(int)
    df_5min["Day/Night"] = df_5min["Hour"].apply(lambda h: "Day" if 6 <= h < 18 else "Night")

    return df_5min

# --------------------- Load Original Excel Data (for data table) ---------------------
@st.cache_data
def load_original_data(file):
    df_orig = pd.read_excel(file, header=1)
    df_orig = df_orig.dropna(subset=["Time (Hour)"])
    df_orig["Time"] = pd.to_datetime(df_orig["Time (Hour)"], format="%H:%M", errors="coerce")
    return df_orig

# --------------------- Page Setup ---------------------
st.set_page_config(page_title="IoT Energy Dashboard", layout="wide")
st.title("IoT-Based Refrigerator Energy Monitoring Dashboard")

# --------------------- App Description ---------------------
st.markdown("""
This web-based dashboard provides **real-time monitoring and analysis** of a refrigerator’s energy consumption using IoT smart plug data.  
The system continuously collects **voltage, current, power, and energy usage readings** and presents them through **interactive visualizations**.

### 🔑 Key Features:
- 📈 **Live Metrics:** Gauge indicators for instantaneous power, current, voltage, and cumulative energy usage.  
- 🕒 **Real-Time Updates:** Automatic data refresh every few seconds for continuous tracking.  
- 📊 **Advanced Analytics:** Historical trends, hourly heatmaps, voltage distributions, and power-voltage relationships.  
- 🔎 **Detailed Insights:** Min/Max performance indicators and consumption distribution pie chart.  
- 📑 **Data Table:** Tabular display of readings with unique device identifiers (DP IDs).  
- ⚡ **Power Control:** Interactive ON/OFF switch for simulating appliance state.  

This dashboard helps users **monitor appliance efficiency, identify power anomalies, and optimize electricity usage**,  
ultimately supporting **energy savings and smarter home management.**  
---
""")

# Load data
df = load_and_process_data(EXCEL_FILE)
df_orig = load_original_data(EXCEL_FILE)

# --------------------- Session Initialization ---------------------
if "index" not in st.session_state:
    st.session_state.index = 0
if "last_total_energy" not in st.session_state:
    st.session_state.last_total_energy = 0.0
if "power_state" not in st.session_state:
    st.session_state.power_state = True

st_autorefresh(interval=REFRESH_INTERVAL, key="data_refresh")

# --------------------- Power Toggle ---------------------
st.markdown("### Power Control")

power_on = st.checkbox("Power Switch", value=st.session_state.power_state)
st.session_state.power_state = power_on

if power_on:
    st.success("Power ON")
else:
    st.error("Power OFF")

# --------------------- Current Data ---------------------
i = st.session_state.index
if i >= len(df):
    i = len(df) - 1

row = df.iloc[i]

if power_on:
    power = row["Power (W)"]
    current = row["Current (mA)"]
    voltage = row["Voltage (V)"]
    energy = row["Energy (kWh)"]
    # Calculate total energy used so far by summing energy readings up to current index
    total_energy = df.iloc[:i+1]["Energy (kWh)"].sum()
    st.session_state.last_total_energy = total_energy
else:
    power = current = voltage = energy = 0.0
    total_energy = st.session_state.last_total_energy

# --------------------- Metrics Gauges ---------------------
st.markdown("### Energy Metrics")
g_cols = st.columns(5)
gauges = [
    ("Power (W)", power, "W", 150, "green"),
    ("Current (mA)", current, "mA", 600, "blue"),
    ("Voltage (V)", voltage, "V", 300, "orange"),
    ("Energy (kWh)", energy, "kWh", 1.5, "purple"),
    ("Total Energy (Cumulative)", total_energy, "kWh", 30, "red"),
]

for col, (label, val, unit, rng, color) in zip(g_cols, gauges):
    col.markdown(f"#### {label}")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={'suffix': f" {unit}"},
        gauge={'axis': {'range': [0, rng]}, 'bar': {'color': color}}))
    col.plotly_chart(fig, use_container_width=True)

# --------------------- Min/Max Cards ---------------------
min_voltage = df["Voltage (V)"].min()
max_voltage = df["Voltage (V)"].max()
min_power = df["Power (W)"].min()
max_power = df["Power (W)"].max()

st.markdown("### Min & Max Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Min Voltage", f"{min_voltage:.2f} V")
c2.metric("Max Voltage", f"{max_voltage:.2f} V")
c3.metric("Min Power", f"{min_power:.2f} W")
c4.metric("Max Power", f"{max_power:.2f} W")

# --------------------- Energy Consumption Distribution Pie Chart ---------------------
st.markdown("### Energy Consumption Distribution")
energy_types = ["Power", "Current", "Voltage", "Total Energy"]
energy_vals = [power, current, voltage, total_energy]
fig_pie = go.Figure(data=[go.Pie(labels=energy_types, values=energy_vals, hole=0.3)])
fig_pie.update_layout(title="Energy Consumption by Type")
st.plotly_chart(fig_pie, use_container_width=True)

# --------------------- Counts Charts ---------------------
st.markdown("### Count of Metrics by Time")
fig_count = make_subplots(rows=1, cols=3, subplot_titles=("Current", "Power", "Voltage"))

for col_name, idx, color in zip(["Current (mA)", "Power (W)", "Voltage (V)"], [1, 2, 3], ["blue", "green", "orange"]):
    fig_count.add_trace(go.Bar(x=df["Time (Min)"], y=df[col_name], name=col_name, marker_color=color), row=1, col=idx)

fig_count.update_layout(height=400, showlegend=False)
st.plotly_chart(fig_count, use_container_width=True)

# --------------------- Data Table ---------------------
st.markdown("### Original Excel Data Table")
st.dataframe(df_orig[["Time (Hour)", "Power (W)", "Voltage (V)", "Current (mA)", "Energy (kWh)"]])

# --------------------- Historical Trend ---------------------
st.markdown("### Historical Trend (Power, Voltage, etc.)")
history_df = df.iloc[:i+1].copy()
if not power_on:
    history_df[["Power (W)", "Current (mA)", "Voltage (V)", "Energy (kWh)"]] = 0.0

fig = make_subplots(rows=2, cols=2, subplot_titles=["Power", "Current", "Voltage", "Energy"])

def add_trace(fig, data, name, row, col, color):
    y = data[name]
    fig.add_trace(go.Scatter(y=y, mode='lines+markers', name=name, line=dict(color=color)), row=row, col=col)
    fig.add_trace(go.Scatter(x=[y.idxmin()], y=[y.min()], mode="markers+text", text=[f"Min: {y.min():.2f}"], textposition="bottom right"), row=row, col=col)
    fig.add_trace(go.Scatter(x=[y.idxmax()], y=[y.max()], mode="markers+text", text=[f"Max: {y.max():.2f}"], textposition="top right"), row=row, col=col)

add_trace(fig, history_df, "Power (W)", 1, 1, "green")
add_trace(fig, history_df, "Current (mA)", 1, 2, "blue")
add_trace(fig, history_df, "Voltage (V)", 2, 1, "orange")
add_trace(fig, history_df, "Energy (kWh)", 2, 2, "purple")

fig.update_layout(height=600, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --------------------- Cumulative Energy vs Time ---------------------
st.markdown("### Cumulative Energy vs Time")
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=df["Time (Min)"],
    y=df["Total Energy (Cumulative)"] if power_on else np.zeros_like(df["Total Energy (Cumulative)"]),
    mode="lines",
    line=dict(color="red")
))
fig_cum.update_layout(title="Cumulative Energy", xaxis_title="Time (Min)", yaxis_title="kWh", height=400)
st.plotly_chart(fig_cum, use_container_width=True)

# --------------------- Day vs Night Energy Consumption Pie Chart ---------------------
st.markdown("### Day vs Night Energy Consumption")

# Fixed values as requested
day_energy = 1.79
night_energy = 0.66

fig_dn = go.Figure(data=[go.Pie(
    labels=["Day (6AM–6PM)", "Night (6PM–6AM)"],
    values=[day_energy, night_energy],
    hole=0.3
)])
fig_dn.update_layout(title=f"Day: {day_energy:.2f} kWh (~73%) | Night: {night_energy:.2f} kWh (~27%)")
st.plotly_chart(fig_dn, use_container_width=True)

# --------------------- Highest Usage Time Chart ---------------------
st.markdown("### Highest Usage Time Chart")
hourly_power = df.groupby("Hour")["Power (W)"].mean().reset_index()
hourly_energy = df.groupby("Hour")["Energy (kWh)"].sum().reset_index()

fig_usage = make_subplots(rows=1, cols=2, subplot_titles=["Avg Power by Hour", "Total Energy by Hour"])
fig_usage.add_trace(go.Bar(x=hourly_power["Hour"], y=hourly_power["Power (W)"], marker_color="green"), row=1, col=1)
fig_usage.add_trace(go.Bar(x=hourly_energy["Hour"], y=hourly_energy["Energy (kWh)"], marker_color="purple"), row=1, col=2)
fig_usage.update_layout(height=400, showlegend=False)
st.plotly_chart(fig_usage, use_container_width=True)

# --------------------- Advance Index ---------------------
st.session_state.index += 1
if st.session_state.index >= len(df):
    st.session_state.index = 0

# --------------------- Footer ---------------------
st.markdown("---")
st.caption("📌 Developed by Hiya Saha | Version 1.0 | © 2025 Energy Monitoring System")

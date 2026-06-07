import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd
import plotly.graph_objects as go

# --- Core Financial Calculations ---
def get_recommendations(demand, std_dev, lead_time, service_level):
    if lead_time <= 0: return 0, 0
    z_score = norm.ppf(service_level)
    safety_stock = max(0, z_score * std_dev * math.sqrt(lead_time))
    rop = max(0, (demand * lead_time) + safety_stock)
    return rop, safety_stock

def get_financials(actual_rop, demand, lead_time, order_qty, unit_cost):
    actual_ss = max(0, actual_rop - (demand * lead_time))
    avg_working_capital = ((order_qty / 2) + actual_ss) * unit_cost
    max_working_capital = (order_qty + actual_ss) * unit_cost
    return actual_ss, avg_working_capital, max_working_capital

# --- Scenario 1: Single Warehouse (Vectorized) ---
def simulate_single_stage(demand_mean, std_dev, rop, ss, q, lead_time, days, warmup):
    total_days = warmup + days
    lt = int(lead_time)
    
    # 1. Vectorized Generation of all Random Demands
    demands = np.maximum(0, np.random.normal(demand_mean, std_dev, total_days))
    
    # 2. Pre-allocate flat NumPy Arrays (Replacing slow Python Lists)
    on_hand = np.zeros(total_days)
    pipeline = np.zeros(total_days)
    arrivals = np.zeros(total_days + lt + 1) 
    
    current_inv = rop + (q / 2)
    current_pipeline = 0
    stockout_days = 0
    
    for t in range(total_days):
        # Process Arrivals via array index
        arriving_today = arrivals[t]
        current_inv += arriving_today
        current_pipeline -= arriving_today
        
        # Fulfill Demand
        current_inv -= demands[t]
        
        if t >= warmup and current_inv < 0:
            stockout_days += 1
            
        # Check Reorder
        if current_inv + current_pipeline <= rop:
            arrivals[t + lt] += q
            current_pipeline += q
            
        # Record state
        on_hand[t] = current_inv
        pipeline[t] = current_pipeline
        
    # Slicing the NumPy arrays directly into Pandas (Instantaneous)
    df = pd.DataFrame({
        'Day': np.arange(1, days + 1),
        'Actual Daily Demand': demands[warmup:],
        'On-Hand Inventory': np.maximum(0, on_hand[warmup:]), # Floors negatives to 0
        'Pipeline Inventory': pipeline[warmup:],
        'ROP Limit': rop
    })
    
    simulated_fill_rate = 1 - (stockout_days / days)
    return df, simulated_fill_rate

# --- Scenario 2 & 3: Two-Stage (Vectorized) ---
def simulate_two_stage(sec_demand, sec_std, sec_rop, sec_q, sec_lt, main_rop, main_q, main_lt, days, warmup, strategy="installation"):
    total_days = warmup + days
    s_lt = int(sec_lt)
    m_lt = int(main_lt)
    
    # Vectorized Demand
    sec_demands = np.maximum(0, np.random.normal(sec_demand, sec_std, total_days))
    
    # Pre-allocate State Arrays
    hist_sec_inv = np.zeros(total_days)
    hist_main_inv = np.zeros(total_days)
    hist_sec_pipe = np.zeros(total_days)
    hist_main_pipe = np.zeros(total_days)
    hist_blame = np.zeros(total_days, dtype=bool)
    
    main_inv = main_rop + main_q
    sec_inv = sec_rop + sec_q
    
    # Flat array arrival timelines
    main_arrivals = np.zeros(total_days + m_lt + 1)
    sec_arrivals = np.zeros(total_days + s_lt + 1)
    
    main_pipe_qty = 0
    sec_pipe_qty = 0
    main_backlog = 0 
    sec_stockout_days = 0

    for t in range(total_days):
        # 1. Main receives
        main_inv += main_arrivals[t]
        main_pipe_qty -= main_arrivals[t]

        # 2. Main fulfills Backlogs
        while main_backlog >= sec_q and main_inv >= sec_q:
            main_inv -= sec_q
            main_backlog -= sec_q
            sec_arrivals[t + s_lt] += sec_q
            sec_pipe_qty += sec_q

        # 3. Sec receives
        sec_inv += sec_arrivals[t]
        sec_pipe_qty -= sec_arrivals[t]

        # 4. Sec Demand
        sec_inv -= sec_demands[t]
        delayed_by_main = (sec_inv < 0) and (main_backlog > 0)
        
        if t >= warmup and sec_inv < 0:
            sec_stockout_days += 1

        # 5. Sec Order
        sec_pos = sec_inv + sec_pipe_qty + main_backlog
        if sec_pos <= sec_rop:
            if main_inv >= sec_q:
                main_inv -= sec_q
                sec_arrivals[t + s_lt] += sec_q
                sec_pipe_qty += sec_q
            else:
                main_backlog += sec_q 

        # 6. Main Order
        if strategy == "installation":
            main_trigger_pos = main_inv + main_pipe_qty - main_backlog
        elif strategy == "echelon":
            main_trigger_pos = main_inv + sec_inv + main_pipe_qty + sec_pipe_qty
            
        if main_trigger_pos <= main_rop:
            main_arrivals[t + m_lt] += main_q
            main_pipe_qty += main_q

        # 7. Record State
        hist_sec_inv[t] = sec_inv
        hist_main_inv[t] = main_inv
        hist_sec_pipe[t] = sec_pipe_qty
        hist_main_pipe[t] = main_pipe_qty
        hist_blame[t] = delayed_by_main

    # Slice directly into DataFrame
    df = pd.DataFrame({
        'Day': np.arange(1, days + 1),
        'Daily Demand': sec_demands[warmup:],
        'Sec On-Hand': np.maximum(0, hist_sec_inv[warmup:]),
        'Sec Pipeline': hist_sec_pipe[warmup:],
        'Main On-Hand': np.maximum(0, hist_main_inv[warmup:]),
        'Main Pipeline': hist_main_pipe[warmup:],
        'Stockout_Blamed_On_Main': hist_blame[warmup:]
    })
        
    simulated_fill_rate = 1 - (sec_stockout_days / days)
    return df, simulated_fill_rate

# --- Plotly Helper Function ---
def render_interactive_chart(df, y_cols, colors):
    fig = go.Figure()
    
    for i, col in enumerate(y_cols):
        is_pipeline = 'Pipeline' in col
        fig.add_trace(go.Scatter(
            x=df['Day'], y=df[col], mode='lines', name=col, 
            line=dict(color=colors[i], width=2 if not is_pipeline else 3),
            line_shape='hv' if is_pipeline else 'linear',
            opacity=0.8 if is_pipeline else 1.0
        ))
        
    fig.update_layout(
        xaxis_title="Day", yaxis_title="Units", hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    
    st.plotly_chart(fig, use_container_width=True)

# --- App Configuration ---
st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")
st.title("📦 Supply Chain Scenario Architect")

# ==========================================
# GLOBAL SETTINGS
# ==========================================
st.markdown("### ⚙️ Global Simulation Settings")
col_g1, col_g2 = st.columns(2)
warmup_days = col_g1.number_input("Warm-up Period", min_value=0, value=150, step=30)
sim_days = col_g2.number_input("Display Period (Days plotted)", min_value=10, value=300, step=30)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🏢 Scenario 1: Single Central", "🏬 Scenario 2: Two-Stage (Local ROP)", "🌍 Scenario 3: Multi-Echelon"])

# --- TAB 1: SINGLE WAREHOUSE ---
with tab1:
    col1a, col1b, col1c, col1d = st.columns(4)
    s1_demand = col1a.number_input("Avg Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s1_d")
    s1_std_dev = col1b.number_input("Demand Std Dev", min_value=0.0, value=20.0, step=5.0, key="s1_std")
    s1_lead_time = col1c.number_input("Lead Time (days)", min_value=0.0, value=7.0, step=1.0, key="s1_lt")
    s1_service_level = col1d.slider("Target Fill Rate", 0.50, 0.999, 0.95, key="s1_sl")
    
    col1e, col1f, col1g = st.columns(3)
    s1_cost = col1e.number_input("Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s1_cost")
    s1_q = col1f.number_input("Order Qty (Q)", min_value=1, value=500, step=50, key="s1_q")
    
    rec_s1_rop, rec_s1_ss = get_recommendations(s1_demand, s1_std_dev, s1_lead_time, s1_service_level)
    s1_actual_rop = col1g.number_input("Actual ROP", min_value=0, value=int(rec_s1_rop), step=10, key="s1_act")
    col1g.caption(f"💡 Suggested: **{rec_s1_rop:,.0f}**")
    
    st.markdown("---")
    s1_act_ss, s1_avg_wc, s1_max_wc = get_financials(s1_actual_rop, s1_demand, s1_lead_time, s1_q, s1_cost)
    df_s1, sim_fr_1 = simulate_single_stage(s1_demand, s1_std_dev, s1_actual_rop, s1_act_ss, s1_q, s1_lead_time, sim_days, warmup_days)

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Working Capital", f"${s1_avg_wc:,.2f}")
    m2.metric("Realized Safety Stock", f"{s1_act_ss:,.0f} units")
    m3.metric("Simulated Fill Rate", f"{sim_fr_1*100:.1f}%")
    
    render_interactive_chart(df_s1, ['On-Hand Inventory', 'Pipeline Inventory', 'ROP Limit'], ["#1f77b4", "#9467bd", "#ff7f0e"])
    with st.expander("📋 View Daily Data Table"): st.dataframe(df_s1, use_container_width=True)

# --- TAB 2: TWO-STAGE (LOCAL ROP) ---
with tab2:
    s2_cost = st.number_input("Shared Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s2_cost")
    
    st.markdown("#### Secondary (Front-line)")
    c2a, c2b, c2c, c2d = st.columns(4)
    s2_sec_demand = c2a.number_input("Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s2_sec_d")
    s2_sec_std = c2b.number_input("Std Dev", min_value=0.0, value=20.0, step=5.0, key="s2_sec_std")
    s2_sec_lt = c2c.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s2_sec_lt")
    s2_sec_sl = c2d.slider("Sec Target Fill Rate", 0.50, 0.999, 0.95, key="s2_sec_sl")
    
    c2e, c2f = st.columns(2)
    s2_sec_q = c2e.number_input("Sec Order Qty", min_value=1, value=300, step=50, key="s2_sec_q")
    rec_sec_rop, _ = get_recommendations(s2_sec_demand, s2_sec_std, s2_sec_lt, s2_sec_sl)
    s2_sec_actual_rop = c2f.number_input("Sec Actual ROP", min_value=0, value=int(rec_sec_rop), step=10, key="s2_sec_act")
    c2f.caption(f"💡 Suggested: **{rec_sec_rop:,.0f}**")
    s2_sec_act_ss, s2_sec_avg_wc, _ = get_financials(s2_sec_actual_rop, s2_sec_demand, s2_sec_lt, s2_sec_q, s2_cost)
    
    st.markdown("#### Main (Hub)")
    c3a, c3b, c3c, c3d = st.columns(4)
    s2_main_demand = c3a.number_input("Agg Demand", min_value=0.0, value=s2_sec_demand, step=10.0, key="s2_main_d")
    s2_main_std = c3b.number_input("Agg Std Dev", min_value=0.0, value=s2_sec_std, step=5.0, key="s2_main_std")
    s2_main_lt = c3c.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s2_main_lt")
    s2_main_sl = c3d.slider("Main Target Fill Rate", 0.50, 0.999, 0.98, key="s2_main_sl")
    
    c3e, c3f = st.columns(2)
    s2_main_q = c3e.number_input("Main Order Qty", min_value=1, value=800, step=50, key="s2_main_q")
    rec_main_rop, _ = get_recommendations(s2_main_demand, s2_main_std, s2_main_lt, s2_main_sl)
    s2_main_actual_rop = c3f.number_input("Main Actual ROP", min_value=0, value=int(rec_main_rop), step=10, key="s2_main_act")
    c3f.caption(f"💡 Suggested: **{rec_main_rop:,.0f}**")
    _, s2_main_avg_wc, _ = get_financials(s2_main_actual_rop, s2_main_demand, s2_main_lt, s2_main_q, s2_cost)

    st.markdown("---")
    df_s2, sim_fr_2 = simulate_two_stage(s2_sec_demand, s2_sec_std, s2_sec_actual_rop, s2_sec_q, s2_sec_lt, s2_main_actual_rop, s2_main_q, s2_main_lt, sim_days, warmup_days, "installation")
    
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("System Avg WC", f"${(s2_sec_avg_wc + s2_main_avg_wc):,.2f}")
    sm2.metric("Target Fill Rate", f"{(s2_sec_sl * s2_main_sl)*100:.1f}%")
    sm3.metric("Simulated Fill Rate", f"{sim_fr_2*100:.1f}%")
    sm4.metric("Main Delay Stockouts", f"{df_s2['Stockout_Blamed_On_Main'].sum()} Days")

    render_interactive_chart(df_s2, ['Sec On-Hand', 'Sec Pipeline', 'Main On-Hand', 'Main Pipeline'], ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"])
    with st.expander("📋 View Daily Data Table"): st.dataframe(df_s2, use_container_width=True)

# --- TAB 3: ECHELON SYSTEM ---
with tab3:
    s3_cost = st.number_input("Shared Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s3_cost")
    
    st.markdown("#### Secondary (Front-line)")
    c4a, c4b, c4c, c4d = st.columns(4)
    s3_sec_demand = c4a.number_input("Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s3_sec_d")
    s3_sec_std = c4b.number_input("Std Dev", min_value=0.0, value=20.0, step=5.0, key="s3_sec_std")
    s3_sec_lt = c4c.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s3_sec_lt")
    s3_sec_sl = c4d.slider("Sec Target Fill Rate", 0.50, 0.999, 0.95, key="s3_sec_sl")
    
    c4e, c4f = st.columns(2)
    s3_sec_q = c4e.number_input("Sec Order Qty", min_value=1, value=300, step=50, key="s3_sec_q")
    rec_s3_sec_rop, _ = get_recommendations(s3_sec_demand, s3_sec_std, s3_sec_lt, s3_sec_sl)
    s3_sec_actual_rop = c4f.number_input("Sec Actual ROP", min_value=0, value=int(rec_s3_sec_rop), step=10, key="s3_sec_act")
    c4f.caption(f"💡 Suggested: **{rec_s3_sec_rop:,.0f}**")
    _, s3_sec_avg_wc, _ = get_financials(s3_sec_actual_rop, s3_sec_demand, s3_sec_lt, s3_sec_q, s3_cost)
    
    st.markdown("#### Main (Echelon Evaluator)")
    c5a, c5b, c5c, c5d = st.columns(4)
    s3_main_demand = c5a.number_input("Agg Demand", min_value=0.0, value=s3_sec_demand, step=10.0, key="s3_main_d")
    s3_main_std = c5b.number_input("Agg Std Dev", min_value=0.0, value=s3_sec_std, step=5.0, key="s3_main_std")
    s3_main_lt = c5c.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s3_main_lt")
    s3_main_sl = c5d.slider("Main Target Fill Rate", 0.50, 0.999, 0.98, key="s3_main_sl")
    
    c5e, c5f = st.columns(2)
    s3_main_q = c5e.number_input("Main Order Qty", min_value=1, value=800, step=50, key="s3_main_q")
    _, main_base_ss = get_recommendations(s3_main_demand, s3_main_std, s3_main_lt, s3_main_sl)
    rec_echelon_rop = rec_s3_sec_rop + (s3_main_demand * s3_main_lt) + main_base_ss
    s3_echelon_actual_rop = c5f.number_input("Echelon Actual ROP", min_value=0, value=int(rec_echelon_rop), step=10, key="s3_ech_act")
    c5f.caption(f"💡 Suggested: **{rec_echelon_rop:,.0f}**")
    _, s3_main_avg_wc, _ = get_financials(s3_echelon_actual_rop - rec_s3_sec_rop, s3_main_demand, s3_main_lt, s3_main_q, s3_cost)

    st.markdown("---")
    df_s3, sim_fr_3 = simulate_two_stage(s3_sec_demand, s3_sec_std, s3_sec_actual_rop, s3_sec_q, s3_sec_lt, s3_echelon_actual_rop, s3_main_q, s3_main_lt, sim_days, warmup_days, "echelon")
    
    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.metric("System Avg WC", f"${(s3_sec_avg_wc + s3_main_avg_wc):,.2f}")
    tm2.metric("Target Fill Rate", f"{(s3_sec_sl * s3_main_sl)*100:.1f}%")
    tm3.metric("Simulated Fill Rate", f"{sim_fr_3*100:.1f}%")
    tm4.metric("Main Delay Stockouts", f"{df_s3['Stockout_Blamed_On_Main'].sum()} Days")

    render_interactive_chart(df_s3, ['Sec On-Hand', 'Sec Pipeline', 'Main On-Hand', 'Main Pipeline'], ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"])
    with st.expander("📋 View Daily Data Table"): st.dataframe(df_s3, use_container_width=True)

# ==========================================
# MASTER COMPARISON TABLE
# ==========================================
st.markdown("---")
st.header("📋 Master Comparison Summary")

comparison_data = {
    "Metric": ["Target Fill Rate", "Order Qty (Q)", "Suggested ROP", "Actual Set ROP", "Avg Working Capital", "Stockout Days"],
    "S1: Central": [f"{s1_service_level*100:.1f}%", f"{s1_q:,.0f}", f"{rec_s1_rop:,.0f}", f"{s1_actual_rop:,.0f}", f"${s1_avg_wc:,.0f}", "N/A"],
    "S2: Secondary": [f"{s2_sec_sl*100:.1f}%", f"{s2_sec_q:,.0f}", f"{rec_sec_rop:,.0f}", f"{s2_sec_actual_rop:,.0f}", f"${s2_sec_avg_wc:,.0f}", "—"],
    "S2: Main": [f"{s2_main_sl*100:.1f}%", f"{s2_main_q:,.0f}", f"{rec_main_rop:,.0f}", f"{s2_main_actual_rop:,.0f}", f"${s2_main_avg_wc:,.0f}", f"{df_s2['Stockout_Blamed_On_Main'].sum()}"],
    "S3: Secondary": [f"{s3_sec_sl*100:.1f}%", f"{s3_sec_q:,.0f}", f"{rec_s3_sec_rop:,.0f}", f"{s3_sec_actual_rop:,.0f}", f"${s3_sec_avg_wc:,.0f}", "—"],
    "S3: Main (Echelon)": [f"{s3_main_sl*100:.1f}%", f"{s3_main_q:,.0f}", f"{rec_echelon_rop:,.0f}", f"{s3_echelon_actual_rop:,.0f}", f"${s3_main_avg_wc:,.0f}", f"{df_s3['Stockout_Blamed_On_Main'].sum()}"]
}
st.table(pd.DataFrame(comparison_data).set_index("Metric"))

import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go
import math
import io

# --- Helper: Excel Downloader ---
def convert_df_to_excel(df):
    output = io.BytesIO()
    # Using xlsxwriter engine to create the excel file in memory
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Daily Data')
    processed_data = output.getvalue()
    return processed_data

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

# --- Scenario 1: Single Warehouse Detailed Simulation ---
def simulate_single_stage_detailed(demand_mean, std_dev, rop, q, lead_time, days, warmup, allow_partial, track_backlogs):
    total_days = warmup + days
    demands = np.maximum(0, np.random.normal(demand_mean, std_dev, total_days))
    
    inv = rop + (q / 2)
    pipe_qty = 0
    backlog = 0
    arrivals = np.zeros(total_days + int(lead_time) + 1)
    
    rows = []
    total_dem = 0
    total_sales = 0
    stockout_days = 0
    
    for t in range(total_days):
        opening = inv
        arr = arrivals[t]
        pipe_qty -= arr
        
        if track_backlogs and backlog > 0:
            if allow_partial: fill_old = min(arr, backlog)
            else: fill_old = backlog if arr >= backlog else 0
            backlog -= fill_old
            arr_for_today = arr - fill_old
        else:
            arr_for_today = arr
            if not track_backlogs: backlog = 0
            
        avail = inv + arr_for_today
        dem = demands[t]
        
        if allow_partial:
            sales = min(avail, dem)
            shortage = dem - sales
            inv = avail - sales
            if track_backlogs: backlog += shortage
        else:
            if avail >= dem:
                sales = dem
                shortage = 0
                inv = avail - dem
            else:
                sales = 0
                shortage = dem
                inv = avail
                if track_backlogs: backlog += shortage
                
        if t >= warmup:
            total_dem += dem
            total_sales += sales
            if sales < dem: stockout_days += 1
            
        order_given = 0
        if (inv + pipe_qty - backlog) <= rop:
            order_given = q
            arrivals[t + int(lead_time)] += q
            pipe_qty += q
            
        if t >= warmup:
            rows.append([t-warmup+1, opening, arr, opening+arr, dem, sales, shortage, backlog, inv, order_given, 0, pipe_qty])
            
    cols = ["Day", "Opening Balance", "Order Received", "Available Inv", "Demand", "Sales", "Shortage", "Backlogs", "Closing Balance", "Orders Given", "Shortages from Supplier", "Pipeline Inventory"]
    vol_fr = total_sales / total_dem if total_dem > 0 else 1.0
    csl = 1 - (stockout_days / days)
    return pd.DataFrame(rows, columns=cols), vol_fr, csl

# --- Scenario 2 & 3: Two-Stage Detailed Simulation ---
def simulate_two_stage_detailed(sec_demand, sec_std, sec_rop, sec_q, sec_lt, main_rop, main_q, main_lt, days, warmup, sec_allow_partial, sec_track_backlogs, main_allow_partial, main_track_backlogs, strategy="installation"):
    total_days = warmup + days
    sec_demands = np.maximum(0, np.random.normal(sec_demand, sec_std, total_days))
    
    main_inv, sec_inv = main_rop + main_q, sec_rop + sec_q
    main_pipe_qty, sec_pipe_qty = 0, 0
    main_backlog_to_sec, sec_backlog = 0, 0
    main_arrivals = np.zeros(total_days + int(main_lt) + 1)
    sec_arrivals = np.zeros(total_days + int(sec_lt) + 1)
    
    sec_rows, main_rows = [], []
    total_dem = 0
    total_sales = 0
    stockout_days = 0
    main_delay_days = 0

    for t in range(total_days):
        ship_old = 0
        ship_now = 0
        
        # 1. MAIN WAREHOUSE (HUB)
        main_opening = main_inv
        m_arr = main_arrivals[t]
        main_inv += m_arr
        main_pipe_qty -= m_arr
        
        if main_track_backlogs and main_backlog_to_sec > 0 and main_inv > 0:
            if main_allow_partial: ship_old = min(main_backlog_to_sec, main_inv)
            else: ship_old = main_backlog_to_sec if main_inv >= main_backlog_to_sec else 0
            main_inv -= ship_old
            main_backlog_to_sec -= ship_old
            sec_arrivals[t + int(sec_lt)] += ship_old
            sec_pipe_qty += ship_old
        elif not main_track_backlogs:
            main_backlog_to_sec = 0

        # 2. SECONDARY WAREHOUSE (FRONT-LINE)
        sec_opening = sec_inv
        s_arr = sec_arrivals[t]
        sec_pipe_qty -= s_arr
        
        if sec_track_backlogs and sec_backlog > 0:
            if sec_allow_partial: fill_old = min(s_arr, sec_backlog)
            else: fill_old = sec_backlog if s_arr >= sec_backlog else 0
            sec_backlog -= fill_old
            s_arr_for_today = s_arr - fill_old
        else:
            s_arr_for_today = s_arr
            if not sec_track_backlogs: sec_backlog = 0
            
        avail = sec_inv + s_arr_for_today
        dem = sec_demands[t]
        
        if sec_allow_partial:
            sales = min(avail, dem)
            shortage = dem - sales
            sec_inv = avail - sales
            if sec_track_backlogs: sec_backlog += shortage
        else:
            if avail >= dem:
                sales = dem
                shortage = 0
                sec_inv = avail - dem
            else:
                sales = 0
                shortage = dem
                sec_inv = avail
                if sec_track_backlogs: sec_backlog += shortage
                
        delayed_by_main = (sec_inv == 0) and (main_backlog_to_sec > 0) and (sales < dem)

        if t >= warmup:
            total_dem += dem
            total_sales += sales
            if sales < dem: stockout_days += 1
            if delayed_by_main: main_delay_days += 1

        # 3. REORDER TRIGGERS
        sec_pos = sec_inv + sec_pipe_qty + main_backlog_to_sec - sec_backlog
        sec_order_given = 0
        shortage_from_supplier = 0 
        
        if sec_pos <= sec_rop:
            sec_order_given = sec_q
            if main_allow_partial: ship_now = min(main_inv, sec_q)
            else: ship_now = sec_q if main_inv >= sec_q else 0
            main_inv -= ship_now
            shortage_from_supplier = sec_q - ship_now
            if main_track_backlogs: main_backlog_to_sec += shortage_from_supplier
            sec_arrivals[t + int(sec_lt)] += ship_now
            sec_pipe_qty += ship_now

        pos = (main_inv + sec_inv + main_pipe_qty + sec_pipe_qty - sec_backlog) if strategy == "echelon" else (main_inv + main_pipe_qty - main_backlog_to_sec)
        main_order_given = 0
        
        if pos <= main_rop:
            main_order_given = main_q
            main_arrivals[t + int(main_lt)] += main_q
            main_pipe_qty += main_q
            
        if t >= warmup:
            sec_rows.append([t-warmup+1, sec_opening, s_arr, sec_opening+s_arr, dem, sales, shortage, sec_backlog, sec_inv, sec_order_given, shortage_from_supplier, sec_pipe_qty])
            main_rows.append([t-warmup+1, main_opening, m_arr, main_opening+m_arr, sec_order_given, ship_old + ship_now, shortage_from_supplier, main_backlog_to_sec, main_inv, main_order_given, 0, main_pipe_qty])

    cols = ["Day", "Opening Balance", "Order Received", "Available Inv", "Demand", "Sales", "Shortage", "Backlogs", "Closing Balance", "Orders Given", "Shortages from Supplier", "Pipeline Inventory"]
    vol_fr = total_sales / total_dem if total_dem > 0 else 1.0
    csl = 1 - (stockout_days / days)
    return pd.DataFrame(sec_rows, columns=cols), pd.DataFrame(main_rows, columns=cols), vol_fr, csl, main_delay_days

# --- Plotly Helper Function ---
def render_interactive_chart(df, y_cols):
    fig = go.Figure()
    color_map = {
        'On-Hand Inventory': '#1f77b4', 'Pipeline Inventory': '#9467bd', 'Backlogged Orders': '#d62728', 'ROP Limit': '#ff7f0e',
        'Sec On-Hand': '#1f77b4', 'Sec Pipeline': '#aec7e8', 'Main On-Hand': '#2ca02c', 'Main Pipeline': '#98df8a', 'Sec Backlogged': '#d62728'
    }
    
    for col in y_cols:
        if col not in df.columns: continue
        is_pipeline = 'Pipeline' in col
        is_backlog = 'Backlog' in col
        fig.add_trace(go.Scatter(
            x=df['Day'], y=df[col], mode='lines', name=col, 
            line=dict(color=color_map.get(col, '#333333'), width=2 if not is_pipeline else 3),
            line_shape='hv' if is_pipeline or is_backlog else 'linear', opacity=0.8 if is_pipeline else 1.0
        ))
        
    fig.update_layout(
        xaxis_title="Day", yaxis_title="Units", hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=80), plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
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
sim_days = col_g2.number_input("Display Period", min_value=10, value=300, step=30)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🏢 Scenario 1: Single Central", "🏬 Scenario 2: Two-Stage (Local ROP)", "🌍 Scenario 3: Multi-Echelon"])

# ==========================================
# TAB 1: SINGLE WAREHOUSE
# ==========================================
with tab1:
    st.markdown("#### Central Warehouse Variables")
    col1a, col1b, col1c, col1d = st.columns(4)
    s1_demand = col1a.number_input("Avg Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s1_d")
    s1_std_dev = col1b.number_input("Demand Std Dev", min_value=0.0, value=20.0, step=5.0, key="s1_std")
    s1_lead_time = col1c.number_input("Lead Time (days)", min_value=0.0, value=7.0, step=1.0, key="s1_lt")
    s1_service_level = col1d.slider("Target Fill Rate", 0.50, 0.999, 0.95, key="s1_sl")
    
    st.markdown("#### Costs & Reorder Logic")
    col1e, col1f, col1g, col1h = st.columns(4)
    s1_cost = col1e.number_input("Warehouse Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s1_cost")
    s1_pipeline_cost = col1f.number_input("Pipeline Unit Cost ($)", min_value=0.01, value=40.0, step=5.0, key="s1_pipe_cost")
    s1_q = col1g.number_input("Order Qty (Q)", min_value=1, value=500, step=50, key="s1_q")
    rec_s1_rop, _ = get_recommendations(s1_demand, s1_std_dev, s1_lead_time, s1_service_level)
    s1_actual_rop = col1h.number_input("Actual ROP", min_value=0, value=int(rec_s1_rop), step=10, key="s1_act")
    col1h.caption(f"💡 Suggested: **{rec_s1_rop:,.0f}**")
    
    st.markdown("#### Fulfillment Rules")
    s1_col1, s1_col2 = st.columns(2)
    s1_allow_partial = s1_col1.checkbox("Allow Partial", value=True, key="s1_partial", help="Ship available inventory even if it doesn't cover the full order.")
    s1_track_backlogs = s1_col2.checkbox("Track Backlogs", value=True, key="s1_backlog", help="Unmet demand goes into a backlog queue instead of being permanently lost.")
    
    st.markdown("---")
    df_s1, vol_fr_1, csl_1 = simulate_single_stage_detailed(s1_demand, s1_std_dev, s1_actual_rop, s1_q, s1_lead_time, sim_days, warmup_days, s1_allow_partial, s1_track_backlogs)

    s1_avg_oh = df_s1['Closing Balance'].mean()
    s1_avg_pipe = df_s1['Pipeline Inventory'].mean()
    s1_sim_wc = (s1_avg_oh * s1_cost) + (s1_avg_pipe * s1_pipeline_cost)
    s1_daily_wc = (df_s1['Closing Balance'] * s1_cost) + (df_s1['Pipeline Inventory'] * s1_pipeline_cost)
    s1_peak_wc = s1_daily_wc.max()
    tot_sales_1 = df_s1['Sales'].sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Simulated Avg WC", f"${s1_sim_wc:,.2f}")
    m2.metric("Peak Working Capital", f"${s1_peak_wc:,.2f}")
    m3.metric("Volume Fill Rate", f"{vol_fr_1*100:.1f}%")
    m4.metric("Cycle Service Level", f"{csl_1*100:.1f}%")
    m5.metric("Total Sales (Units)", f"{tot_sales_1:,.0f}")
    
    plot_df1 = pd.DataFrame({'Day': df_s1['Day'], 'On-Hand Inventory': df_s1['Closing Balance'], 'Pipeline Inventory': df_s1['Pipeline Inventory'], 'Backlogged Orders': df_s1['Backlogs'], 'ROP Limit': s1_actual_rop})
    render_interactive_chart(plot_df1, ['On-Hand Inventory', 'Pipeline Inventory', 'Backlogged Orders', 'ROP Limit'])
    
    st.markdown("### 💰 Average Inventory & Valuation Summary")
    val_data_1a = {
        "Asset Location / State": ["Central Warehouse (On-Hand)", "Pipeline (Supplier → Central)"],
        "Unit Cost": [f"${s1_cost:,.2f}", f"${s1_pipeline_cost:,.2f}"],
        "Average Units": [f"{s1_avg_oh:,.0f}", f"{s1_avg_pipe:,.0f}"],
        "Average Value": [f"${s1_avg_oh * s1_cost:,.2f}", f"${s1_avg_pipe * s1_pipeline_cost:,.2f}"]
    }
    val_data_1b = {
        "Ownership Entity": ["Central Facility (Total System)"],
        "Average Total Units": [f"{s1_avg_oh + s1_avg_pipe:,.0f}"],
        "Average Total Value": [f"${s1_sim_wc:,.2f}"]
    }
    tcol1_1, tcol1_2 = st.columns(2)
    with tcol1_1:
        st.markdown("**Detailed Location Valuation**")
        st.table(pd.DataFrame(val_data_1a).set_index("Asset Location / State"))
    with tcol1_2:
        st.markdown("**Ownership Valuation**")
        st.table(pd.DataFrame(val_data_1b).set_index("Ownership Entity"))

    # --- NEW: Focused Daily Tracking for Tab 1 ---
    st.markdown("### 📅 Daily Inventory Tracking")
    
    df_tab1_daily = df_s1[['Day', 'Closing Balance', 'Pipeline Inventory']].copy()
    df_tab1_daily.rename(columns={'Closing Balance': 'On-Hand Inventory (Units)'}, inplace=True)
    
    col_dl1, col_dl2 = st.columns([3, 1])
    with col_dl1:
        st.dataframe(df_tab1_daily, use_container_width=True, height=250)
    with col_dl2:
        st.write("Download this table as an Excel file:")
        st.download_button(
            label="📥 Download Excel",
            data=convert_df_to_excel(df_tab1_daily),
            file_name="central_warehouse_daily_inventory.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_tab1"
        )
    
    with st.expander("📋 View Complete Detailed Data Table"): st.dataframe(df_s1, use_container_width=True)

# ==========================================
# TAB 2: TWO-STAGE (LOCAL ROP)
# ==========================================
with tab2:
    st.markdown("#### Secondary (Front-line)")
    c2a, c2b, c2c, c2d = st.columns(4)
    s2_sec_demand = c2a.number_input("Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s2_sec_d")
    s2_sec_std = c2b.number_input("Std Dev", min_value=0.0, value=20.0, step=5.0, key="s2_sec_std")
    s2_sec_lt = c2c.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s2_sec_lt")
    s2_sec_sl = c2d.slider("Sec Target Fill Rate", 0.50, 0.999, 0.95, key="s2_sec_sl")
    
    c2e, c2f, c2g, c2h, c2i = st.columns(5)
    s2_sec_cost = c2e.number_input("Unit Cost ($)", min_value=0.01, value=60.0, step=5.0, key="s2_sec_cost")
    s2_sec_q = c2f.number_input("Order Qty", min_value=1, value=300, step=50, key="s2_sec_q")
    rec_sec_rop, _ = get_recommendations(s2_sec_demand, s2_sec_std, s2_sec_lt, s2_sec_sl)
    s2_sec_actual_rop = c2g.number_input("Actual ROP", min_value=0, value=int(rec_sec_rop), step=10, key="s2_sec_act")
    c2g.caption(f"💡 Suggested: **{rec_sec_rop:,.0f}**")
    s2_sec_allow_partial = c2h.checkbox("Allow Partial", value=True, key="s2_sec_partial", help="Secondary fulfilling customers")
    s2_sec_track_backlogs = c2i.checkbox("Track Backlogs", value=True, key="s2_sec_backlog", help="Secondary tracking customer backlogs")
    
    st.markdown("#### Main (Hub)")
    c3a, c3b, c3c, c3d = st.columns(4)
    s2_main_demand = c3a.number_input("Agg Demand", min_value=0.0, value=s2_sec_demand, step=10.0, key="s2_main_d")
    s2_main_std = c3b.number_input("Agg Std Dev", min_value=0.0, value=s2_sec_std, step=5.0, key="s2_main_std")
    s2_main_lt = c3c.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s2_main_lt")
    s2_main_sl = c3d.slider("Main Target Fill Rate", 0.50, 0.999, 0.98, key="s2_main_sl")
    
    c3e, c3f, c3g, c3h, c3i = st.columns(5)
    s2_main_cost = c3e.number_input("Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s2_main_cost")
    s2_main_q = c3f.number_input("Order Qty", min_value=1, value=800, step=50, key="s2_main_q")
    rec_main_rop, _ = get_recommendations(s2_main_demand, s2_main_std, s2_main_lt, s2_main_sl)
    s2_main_actual_rop = c3g.number_input("Actual ROP", min_value=0, value=int(rec_main_rop), step=10, key="s2_main_act")
    c3g.caption(f"💡 Suggested: **{rec_main_rop:,.0f}**")
    s2_main_allow_partial = c3h.checkbox("Allow Partial", value=True, key="s2_main_partial", help="Main fulfilling Secondary")
    s2_main_track_backlogs = c3i.checkbox("Track Backlogs", value=True, key="s2_main_backlog", help="Main tracking Secondary backlogs")

    st.markdown("---")
    df_sec_s2, df_main_s2, vol_fr_2, csl_2, delay_2 = simulate_two_stage_detailed(s2_sec_demand, s2_sec_std, s2_sec_actual_rop, s2_sec_q, s2_sec_lt, s2_main_actual_rop, s2_main_q, s2_main_lt, sim_days, warmup_days, s2_sec_allow_partial, s2_sec_track_backlogs, s2_main_allow_partial, s2_main_track_backlogs, "installation")
    
    avg_sec_oh_2 = df_sec_s2['Closing Balance'].mean()
    avg_sec_pipe_2 = df_sec_s2['Pipeline Inventory'].mean()
    avg_main_oh_2 = df_main_s2['Closing Balance'].mean()
    avg_main_pipe_2 = df_main_s2['Pipeline Inventory'].mean()
    total_sys_val_2 = (avg_sec_oh_2 * s2_sec_cost) + (avg_sec_pipe_2 * s2_sec_cost) + (avg_main_oh_2 * s2_main_cost) + (avg_main_pipe_2 * s2_main_cost)
    s2_daily_sys_val = (df_sec_s2['Closing Balance'] + df_sec_s2['Pipeline Inventory']) * s2_sec_cost + (df_main_s2['Closing Balance'] + df_main_s2['Pipeline Inventory']) * s2_main_cost
    peak_sys_val_2 = s2_daily_sys_val.max()
    tot_sales_2 = df_sec_s2['Sales'].sum()

    sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
    sm1.metric("Simulated Avg WC", f"${total_sys_val_2:,.2f}")
    sm2.metric("Peak Working Capital", f"${peak_sys_val_2:,.2f}")
    sm3.metric("Volume Fill Rate", f"{vol_fr_2*100:.1f}%")
    sm4.metric("Service Level", f"{csl_2*100:.1f}%")
    sm5.metric("Main Delays", f"{delay_2} Days")
    sm6.metric("Total Sales", f"{tot_sales_2:,.0f}")

    plot_df2 = pd.DataFrame({'Day': df_sec_s2['Day'], 'Sec On-Hand': df_sec_s2['Closing Balance'], 'Sec Pipeline': df_sec_s2['Pipeline Inventory'], 'Sec Backlogged': df_sec_s2['Backlogs'], 'Main On-Hand': df_main_s2['Closing Balance'], 'Main Pipeline': df_main_s2['Pipeline Inventory']})
    render_interactive_chart(plot_df2, ['Sec On-Hand', 'Sec Pipeline', 'Sec Backlogged', 'Main On-Hand', 'Main Pipeline'])
    
    st.markdown("### 💰 Average Inventory & Valuation Summary")
    val_data_2a = {
        "Asset Location / State": ["Secondary Warehouse (On-Hand)", "Pipeline (Main → Secondary)", "Main Warehouse (On-Hand)", "Pipeline (Supplier → Main)"],
        "Unit Cost": [f"${s2_sec_cost:,.2f}", f"${s2_sec_cost:,.2f}", f"${s2_main_cost:,.2f}", f"${s2_main_cost:,.2f}"],
        "Average Units": [f"{avg_sec_oh_2:,.0f}", f"{avg_sec_pipe_2:,.0f}", f"{avg_main_oh_2:,.0f}", f"{avg_main_pipe_2:,.0f}"],
        "Average Value": [f"${avg_sec_oh_2 * s2_sec_cost:,.2f}", f"${avg_sec_pipe_2 * s2_sec_cost:,.2f}", f"${avg_main_oh_2 * s2_main_cost:,.2f}", f"${avg_main_pipe_2 * s2_main_cost:,.2f}"]
    }
    val_data_2b = {
        "Ownership Entity": ["Secondary (Includes Main→Sec Pipeline)", "Main (Includes Sup→Main Pipeline)"],
        "Average Total Units": [f"{avg_sec_oh_2 + avg_sec_pipe_2:,.0f}", f"{avg_main_oh_2 + avg_main_pipe_2:,.0f}"],
        "Average Total Value": [f"${(avg_sec_oh_2 + avg_sec_pipe_2) * s2_sec_cost:,.2f}", f"${(avg_main_oh_2 + avg_main_pipe_2) * s2_main_cost:,.2f}"]
    }
    tcol2_1, tcol2_2 = st.columns(2)
    with tcol2_1:
        st.markdown("**Detailed Location Valuation**")
        st.table(pd.DataFrame(val_data_2a).set_index("Asset Location / State"))
    with tcol2_2:
        st.markdown("**Ownership Valuation (FOB Origin)**")
        st.table(pd.DataFrame(val_data_2b).set_index("Ownership Entity"))

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        with st.expander("📋 View Secondary Detailed Data"): st.dataframe(df_sec_s2, use_container_width=True)
    with col_t2:
        with st.expander("📋 View Main Detailed Data"): st.dataframe(df_main_s2, use_container_width=True)

# ==========================================
# TAB 3: ECHELON SYSTEM
# ==========================================
with tab3:
    st.markdown("#### Secondary (Front-line)")
    c4a, c4b, c4c, c4d = st.columns(4)
    s3_sec_demand = c4a.number_input("Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s3_sec_d")
    s3_sec_std = c4b.number_input("Std Dev", min_value=0.0, value=20.0, step=5.0, key="s3_sec_std")
    s3_sec_lt = c4c.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s3_sec_lt")
    s3_sec_sl = c4d.slider("Sec Target Fill Rate", 0.50, 0.999, 0.95, key="s3_sec_sl")
    
    c4e, c4f, c4g, c4h, c4i = st.columns(5)
    s3_sec_cost = c4e.number_input("Unit Cost ($)", min_value=0.01, value=60.0, step=5.0, key="s3_sec_cost")
    s3_sec_q = c4f.number_input("Sec Order Qty", min_value=1, value=300, step=50, key="s3_sec_q")
    rec_s3_sec_rop, _ = get_recommendations(s3_sec_demand, s3_sec_std, s3_sec_lt, s3_sec_sl)
    s3_sec_actual_rop = c4g.number_input("Sec Actual ROP", min_value=0, value=int(rec_s3_sec_rop), step=10, key="s3_sec_act")
    c4g.caption(f"💡 Suggested: **{rec_s3_sec_rop:,.0f}**")
    s3_sec_allow_partial = c4h.checkbox("Allow Partial", value=True, key="s3_sec_partial")
    s3_sec_track_backlogs = c4i.checkbox("Track Backlogs", value=True, key="s3_sec_backlog")
    
    st.markdown("#### Main (Echelon Evaluator)")
    c5a, c5b, c5c, c5d = st.columns(4)
    s3_main_demand = c5a.number_input("Agg Demand", min_value=0.0, value=s3_sec_demand, step=10.0, key="s3_main_d")
    s3_main_std = c5b.number_input("Agg Std Dev", min_value=0.0, value=s3_sec_std, step=5.0, key="s3_main_std")
    s3_main_lt = c5c.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s3_main_lt")
    s3_main_sl = c5d.slider("Main Target Fill Rate", 0.50, 0.999, 0.98, key="s3_main_sl")
    
    c5e, c5f, c5g, c5h, c5i = st.columns(5)
    s3_main_cost = c5e.number_input("Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s3_main_cost")
    s3_main_q = c5f.number_input("Main Order Qty", min_value=1, value=800, step=50, key="s3_main_q")
    _, main_base_ss = get_recommendations(s3_main_demand, s3_main_std, s3_main_lt, s3_main_sl)
    rec_echelon_rop = rec_s3_sec_rop + (s3_main_demand * s3_main_lt) + main_base_ss
    s3_echelon_actual_rop = c5g.number_input("Echelon Actual ROP", min_value=0, value=int(rec_echelon_rop), step=10, key="s3_ech_act")
    c5g.caption(f"💡 Suggested: **{rec_echelon_rop:,.0f}**")
    s3_main_allow_partial = c5h.checkbox("Allow Partial", value=True, key="s3_main_partial")
    s3_main_track_backlogs = c5i.checkbox("Track Backlogs", value=True, key="s3_main_backlog")

    st.markdown("---")
    df_sec_s3, df_main_s3, vol_fr_3, csl_3, delay_3 = simulate_two_stage_detailed(s3_sec_demand, s3_sec_std, s3_sec_actual_rop, s3_sec_q, s3_sec_lt, s3_echelon_actual_rop, s3_main_q, s3_main_lt, sim_days, warmup_days, s3_sec_allow_partial, s3_sec_track_backlogs, s3_main_allow_partial, s3_main_track_backlogs, "echelon")
    
    avg_sec_oh_3 = df_sec_s3['Closing Balance'].mean()
    avg_sec_pipe_3 = df_sec_s3['Pipeline Inventory'].mean()
    avg_main_oh_3 = df_main_s3['Closing Balance'].mean()
    avg_main_pipe_3 = df_main_s3['Pipeline Inventory'].mean()
    total_sys_val_3 = (avg_sec_oh_3 * s3_sec_cost) + (avg_sec_pipe_3 * s3_sec_cost) + (avg_main_oh_3 * s3_main_cost) + (avg_main_pipe_3 * s3_main_cost)
    s3_daily_sys_val = (df_sec_s3['Closing Balance'] + df_sec_s3['Pipeline Inventory']) * s3_sec_cost + (df_main_s3['Closing Balance'] + df_main_s3['Pipeline Inventory']) * s3_main_cost
    peak_sys_val_3 = s3_daily_sys_val.max()
    tot_sales_3 = df_sec_s3['Sales'].sum()

    tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
    tm1.metric("Simulated Avg WC", f"${total_sys_val_3:,.2f}")
    tm2.metric("Peak Working Capital", f"${peak_sys_val_3:,.2f}")
    tm3.metric("Volume Fill Rate", f"{vol_fr_3*100:.1f}%")
    tm4.metric("Service Level", f"{csl_3*100:.1f}%")
    tm5.metric("Main Delays", f"{delay_3} Days")
    tm6.metric("Total Sales", f"{tot_sales_3:,.0f}")

    plot_df3 = pd.DataFrame({'Day': df_sec_s3['Day'], 'Sec On-Hand': df_sec_s3['Closing Balance'], 'Sec Pipeline': df_sec_s3['Pipeline Inventory'], 'Sec Backlogged': df_sec_s3['Backlogs'], 'Main On-Hand': df_main_s3['Closing Balance'], 'Main Pipeline': df_main_s3['Pipeline Inventory']})
    render_interactive_chart(plot_df3, ['Sec On-Hand', 'Sec Pipeline', 'Sec Backlogged', 'Main On-Hand', 'Main Pipeline'])
    
    st.markdown("### 💰 Average Inventory & Valuation Summary")
    val_data_3a = {
        "Asset Location / State": ["Secondary Warehouse (On-Hand)", "Pipeline (Main → Secondary)", "Main Warehouse (On-Hand)", "Pipeline (Supplier → Main)"],
        "Unit Cost": [f"${s3_sec_cost:,.2f}", f"${s3_sec_cost:,.2f}", f"${s3_main_cost:,.2f}", f"${s3_main_cost:,.2f}"],
        "Average Units": [f"{avg_sec_oh_3:,.0f}", f"{avg_sec_pipe_3:,.0f}", f"{avg_main_oh_3:,.0f}", f"{avg_main_pipe_3:,.0f}"],
        "Average Value": [f"${avg_sec_oh_3 * s3_sec_cost:,.2f}", f"${avg_sec_pipe_3 * s3_sec_cost:,.2f}", f"${avg_main_oh_3 * s3_main_cost:,.2f}", f"${avg_main_pipe_3 * s3_main_cost:,.2f}"]
    }
    val_data_3b = {
        "Ownership Entity": ["Secondary (Includes Main→Sec Pipeline)", "Main (Includes Sup→Main Pipeline)"],
        "Average Total Units": [f"{avg_sec_oh_3 + avg_sec_pipe_3:,.0f}", f"{avg_main_oh_3 + avg_main_pipe_3:,.0f}"],
        "Average Total Value": [f"${(avg_sec_oh_3 + avg_sec_pipe_3) * s3_sec_cost:,.2f}", f"${(avg_main_oh_3 + avg_main_pipe_3) * s3_main_cost:,.2f}"]
    }
    tcol3_1, tcol3_2 = st.columns(2)
    with tcol3_1:
        st.markdown("**Detailed Location Valuation**")
        st.table(pd.DataFrame(val_data_3a).set_index("Asset Location / State"))
    with tcol3_2:
        st.markdown("**Ownership Valuation (FOB Origin)**")
        st.table(pd.DataFrame(val_data_3b).set_index("Ownership Entity"))
    
    # --- NEW: Focused Daily Tracking for Tab 3 ---
    st.markdown("### 📅 Daily Inventory Tracking (System-Wide)")
    
    # Combine Secondary and Main daily inventory statuses
    df_tab3_daily = pd.DataFrame({
        'Day': df_sec_s3['Day'],
        'Secondary On-Hand': df_sec_s3['Closing Balance'],
        'Secondary Pipeline': df_sec_s3['Pipeline Inventory'],
        'Main On-Hand': df_main_s3['Closing Balance'],
        'Main Pipeline': df_main_s3['Pipeline Inventory']
    })
    
    col_dl3, col_dl4 = st.columns([3, 1])
    with col_dl3:
        st.dataframe(df_tab3_daily, use_container_width=True, height=250)
    with col_dl4:
        st.write("Download this table as an Excel file:")
        st.download_button(
            label="📥 Download Excel",
            data=convert_df_to_excel(df_tab3_daily),
            file_name="multi_echelon_daily_inventory.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_tab3"
        )

    col_t3, col_t4 = st.columns(2)
    with col_t3:
        with st.expander("📋 View Secondary Detailed Data"): st.dataframe(df_sec_s3, use_container_width=True)
    with col_t4:
        with st.expander("📋 View Main Detailed Data"): st.dataframe(df_main_s3, use_container_width=True)

# ==========================================
# MASTER COMPARISON TABLE
# ==========================================
st.markdown("---")
st.header("📋 Master Comparison Summary")

comparison_data = {
    "Metric": ["Target Fill Rate", "Order Qty (Q)", "Suggested ROP", "Actual Set ROP", "Avg Working Capital", "Peak Working Capital", "Total Sales (Units)", "Stockout Days"],
    "S1: Central": [f"{s1_service_level*100:.1f}%", f"{s1_q:,.0f}", f"{rec_s1_rop:,.0f}", f"{s1_actual_rop:,.0f}", f"${s1_sim_wc:,.0f}", f"${s1_peak_wc:,.0f}", f"{tot_sales_1:,.0f}", "N/A"],
    "S2: Secondary": [f"{s2_sec_sl*100:.1f}%", f"{s2_sec_q:,.0f}", f"{rec_sec_rop:,.0f}", f"{s2_sec_actual_rop:,.0f}", f"${(avg_sec_oh_2+avg_sec_pipe_2)*s2_sec_cost:,.0f}", f"${peak_sys_val_2:,.0f} (System)", f"{tot_sales_2:,.0f}", "—"],
    "S2: Main": [f"{s2_main_sl*100:.1f}%", f"{s2_main_q:,.0f}", f"{rec_main_rop:,.0f}", f"{s2_main_actual_rop:,.0f}", f"${(avg_main_oh_2+avg_main_pipe_2)*s2_main_cost:,.0f}", "—", "—", f"{delay_2}"],
    "S3: Secondary": [f"{s3_sec_sl*100:.1f}%", f"{s3_sec_q:,.0f}", f"{rec_s3_sec_rop:,.0f}", f"{s3_sec_actual_rop:,.0f}", f"${(avg_sec_oh_3+avg_sec_pipe_3)*s3_sec_cost:,.0f}", f"${peak_sys_val_3:,.0f} (System)", f"{tot_sales_3:,.0f}", "—"],
    "S3: Main (Echelon)": [f"{s3_main_sl*100:.1f}%", f"{s3_main_q:,.0f}", f"{rec_echelon_rop:,.0f}", f"{s3_echelon_actual_rop:,.0f}", f"${(avg_main_oh_3+avg_main_pipe_3)*s3_main_cost:,.0f}", "—", "—", f"{delay_3}"]
}
st.table(pd.DataFrame(comparison_data).set_index("Metric"))

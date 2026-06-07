import streamlit as st
import numpy as np
from scipy.stats import norm
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
        
        # Fulfill old backlogs
        if track_backlogs and backlog > 0:
            fill_old = min(arr, backlog)
            backlog -= fill_old
            arr_for_today = arr - fill_old
        else:
            arr_for_today = arr
            
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
def simulate_two_stage_detailed(sec_demand, sec_std, sec_rop, sec_q, sec_lt, main_rop, main_q, main_lt, days, warmup, allow_partial, track_backlogs, strategy="installation"):
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
        # 1. MAIN WAREHOUSE (HUB)
        main_opening = main_inv
        m_arr = main_arrivals[t]
        main_inv += m_arr
        main_pipe_qty -= m_arr
        
        # Ship partial backlogs to Sec
        if main_backlog_to_sec > 0 and main_inv > 0:
            ship = min(main_backlog_to_sec, main_inv)
            main_inv -= ship
            main_backlog_to_sec -= ship
            sec_arrivals[t + int(sec_lt)] += ship
            sec_pipe_qty += ship

        # 2. SECONDARY WAREHOUSE (FRONT-LINE)
        sec_opening = sec_inv
        s_arr = sec_arrivals[t]
        sec_pipe_qty -= s_arr
        
        if track_backlogs and sec_backlog > 0:
            fill_old = min(s_arr, sec_backlog)
            sec_backlog -= fill_old
            s_arr_for_today = s_arr - fill_old
        else:
            s_arr_for_today = s_arr
            
        avail = sec_inv + s_arr_for_today
        dem = sec_demands[t]
        
        if allow_partial:
            sales = min(avail, dem)
            shortage = dem - sales
            sec_inv = avail - sales
            if track_backlogs: sec_backlog += shortage
        else:
            if avail >= dem:
                sales = dem
                shortage = 0
                sec_inv = avail - dem
            else:
                sales = 0
                shortage = dem
                sec_inv = avail
                if track_backlogs: sec_backlog += shortage
                
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
            ship_now = min(main_inv, sec_q)
            main_inv -= ship_now
            shortage_from_supplier = sec_q - ship_now
            main_backlog_to_sec += shortage_from_supplier
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
            main_rows.append([t-warmup+1, main_opening, m_arr, main_opening+m_arr, 0, 0, 0, main_backlog_to_sec, main_inv, main_order_given, 0, main_pipe_qty])

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
            line_shape='hv' if is_pipeline or is_backlog else 'linear',
            opacity=0.8 if is_pipeline else 1.0
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
col_g1, col_g2, col_g3, col_g4 = st.columns(4)
warmup_days = col_g1.number_input("Warm-up Period", min_value=0, value=150, step=30)
sim_days = col_g2.number_input("Display Period", min_value=10, value=300, step=30)
allow_partial = col_g3.checkbox("Allow Partial Fulfillment", value=True, help="Ships available inventory even if it doesn't cover the full order.")
allow_backlogs = col_g4.checkbox("Track Backlogs", value=True, help="Unmet demand goes into a backlog queue instead of being permanently lost.")
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
    s1_act_ss, s1_avg_wc, _ = get_financials(s1_actual_rop, s1_demand, s1_lead_time, s1_q, s1_cost)
    df_s1, vol_fr_1, csl_1 = simulate_single_stage_detailed(s1_demand, s1_std_dev, s1_actual_rop, s1_q, s1_lead_time, sim_days, warmup_days, allow_partial, allow_backlogs)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Working Capital", f"${s1_avg_wc:,.2f}")
    m2.metric("Realized Safety Stock", f"{s1_act_ss:,.0f} units")
    m3.metric("Volume Fill Rate (Item)", f"{vol_fr_1*100:.1f}%")
    m4.metric("Cycle Service Level", f"{csl_1*100:.1f}%")
    
    plot_df1 = pd.DataFrame({'Day': df_s1['Day'], 'On-Hand Inventory': df_s1['Closing Balance'], 'Pipeline Inventory': df_s1['Pipeline Inventory'], 'Backlogged Orders': df_s1['Backlogs'], 'ROP Limit': s1_actual_rop})
    render_interactive_chart(plot_df1, ['On-Hand Inventory', 'Pipeline Inventory', 'Backlogged Orders', 'ROP Limit'])
    
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
    _, s2_sec_avg_wc, _ = get_financials(s2_sec_actual_rop, s2_sec_demand, s2_sec_lt, s2_sec_q, s2_cost)
    
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
    df_sec_s2, df_main_s2, vol_fr_2, csl_2, delay_2 = simulate_two_stage_detailed(s2_sec_demand, s2_sec_std, s2_sec_actual_rop, s2_sec_q, s2_sec_lt, s2_main_actual_rop, s2_main_q, s2_main_lt, sim_days, warmup_days, allow_partial, allow_backlogs, "installation")
    
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("System Avg WC", f"${(s2_sec_avg_wc + s2_main_avg_wc):,.2f}")
    sm2.metric("Volume Fill Rate (Item)", f"{vol_fr_2*100:.1f}%")
    sm3.metric("Cycle Service Level", f"{csl_2*100:.1f}%")
    sm4.metric("Main Delay Stockouts", f"{delay_2} Days")

    plot_df2 = pd.DataFrame({'Day': df_sec_s2['Day'], 'Sec On-Hand': df_sec_s2['Closing Balance'], 'Sec Pipeline': df_sec_s2['Pipeline Inventory'], 'Sec Backlogged': df_sec_s2['Backlogs'], 'Main On-Hand': df_main_s2['Closing Balance'], 'Main Pipeline': df_main_s2['Pipeline Inventory']})
    render_interactive_chart(plot_df2, ['Sec On-Hand', 'Sec Pipeline', 'Sec Backlogged', 'Main On-Hand', 'Main Pipeline'])
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        with st.expander("📋 View Secondary Warehouse Data"): st.dataframe(df_sec_s2, use_container_width=True)
    with col_t2:
        with st.expander("📋 View Main Warehouse Data"): st.dataframe(df_main_s2, use_container_width=True)

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
    df_sec_s3, df_main_s3, vol_fr_3, csl_3, delay_3 = simulate_two_stage_detailed(s3_sec_demand, s3_sec_std, s3_sec_actual_rop, s3_sec_q, s3_sec_lt, s3_echelon_actual_rop, s3_main_q, s3_main_lt, sim_days, warmup_days, allow_partial, allow_backlogs, "echelon")
    
    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.metric("System Avg WC", f"${(s3_sec_avg_wc + s3_main_avg_wc):,.2f}")
    tm2.metric("Volume Fill Rate (Item)", f"{vol_fr_3*100:.1f}%")
    tm3.metric("Cycle Service Level", f"{csl_3*100:.1f}%")
    tm4.metric("Main Delay Stockouts", f"{delay_3} Days")

    plot_df3 = pd.DataFrame({'Day': df_sec_s3['Day'], 'Sec On-Hand': df_sec_s3['Closing Balance'], 'Sec Pipeline': df_sec_s3['Pipeline Inventory'], 'Sec Backlogged': df_sec_s3['Backlogs'], 'Main On-Hand': df_main_s3['Closing Balance'], 'Main Pipeline': df_main_s3['Pipeline Inventory']})
    render_interactive_chart(plot_df3, ['Sec On-Hand', 'Sec Pipeline', 'Sec Backlogged', 'Main On-Hand', 'Main Pipeline'])
    
    col_t3, col_t4 = st.columns(2)
    with col_t3:
        with st.expander("📋 View Secondary Warehouse Data"): st.dataframe(df_sec_s3, use_container_width=True)
    with col_t4:
        with st.expander("📋 View Main Warehouse Data"): st.dataframe(df_main_s3, use_container_width=True)

# ==========================================
# MASTER COMPARISON TABLE
# ==========================================
st.markdown("---")
st.header("📋 Master Comparison Summary")

comparison_data = {
    "Metric": ["Target Fill Rate", "Order Qty (Q)", "Suggested ROP", "Actual Set ROP", "Avg Working Capital", "Stockout Days"],
    "S1: Central": [f"{s1_service_level*100:.1f}%", f"{s1_q:,.0f}", f"{rec_s1_rop:,.0f}", f"{s1_actual_rop:,.0f}", f"${s1_avg_wc:,.0f}", "N/A"],
    "S2: Secondary": [f"{s2_sec_sl*100:.1f}%", f"{s2_sec_q:,.0f}", f"{rec_sec_rop:,.0f}", f"{s2_sec_actual_rop:,.0f}", f"${s2_sec_avg_wc:,.0f}", "—"],
    "S2: Main": [f"{s2_main_sl*100:.1f}%", f"{s2_main_q:,.0f}", f"{rec_main_rop:,.0f}", f"{s2_main_actual_rop:,.0f}", f"${s2_main_avg_wc:,.0f}", f"{delay_2}"],
    "S3: Secondary": [f"{s3_sec_sl*100:.1f}%", f"{s3_sec_q:,.0f}", f"{rec_s3_sec_rop:,.0f}", f"{s3_sec_actual_rop:,.0f}", f"${s3_sec_avg_wc:,.0f}", "—"],
    "S3: Main (Echelon)": [f"{s3_main_sl*100:.1f}%", f"{s3_main_q:,.0f}", f"{rec_echelon_rop:,.0f}", f"{s3_echelon_actual_rop:,.0f}", f"${s3_main_avg_wc:,.0f}", f"{delay_3}"]
}
st.table(pd.DataFrame(comparison_data).set_index("Metric"))

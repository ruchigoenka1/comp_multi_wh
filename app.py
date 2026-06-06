import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd
import plotly.graph_objects as go

# --- Core Financial Calculations ---
def get_recommendations(demand, std_dev, lead_time, service_level):
    """Calculates the mathematically recommended ROP and Safety Stock."""
    if lead_time <= 0: return 0, 0
    z_score = norm.ppf(service_level)
    safety_stock = max(0, z_score * std_dev * math.sqrt(lead_time))
    rop = max(0, (demand * lead_time) + safety_stock)
    return rop, safety_stock

def get_financials(actual_rop, demand, lead_time, order_qty, unit_cost):
    """Calculates working capital based on the USER'S chosen ROP."""
    actual_ss = max(0, actual_rop - (demand * lead_time))
    avg_working_capital = ((order_qty / 2) + actual_ss) * unit_cost
    max_working_capital = (order_qty + actual_ss) * unit_cost
    return actual_ss, avg_working_capital, max_working_capital

# --- Scenario 1: Single Warehouse ---
def simulate_single_stage(demand, rop, ss, q, lead_time, days, warmup):
    inventory = rop + (q / 2) 
    pipeline = []
    history = []
    
    for day in range(warmup + days):
        for order in pipeline:
            order['time'] -= 1
            if order['time'] <= 0: inventory += order['qty']
        pipeline = [o for o in pipeline if o['time'] > 0]
        
        inventory -= demand
        
        inv_position = inventory + sum(o['qty'] for o in pipeline)
        if inv_position <= rop:
            pipeline.append({'qty': q, 'time': lead_time})
            
        if day >= warmup:
            history.append({
                'Day': day - warmup + 1,
                'On-Hand Inventory': max(0, inventory),
                'Pipeline Inventory': sum(o['qty'] for o in pipeline),
                'ROP Limit': rop
            })
    return pd.DataFrame(history)

# --- Scenario 2 & 3: Two-Stage (Installation vs Echelon) ---
def simulate_two_stage(sec_demand, sec_rop, sec_q, sec_lt, main_rop, main_q, main_lt, days, warmup, strategy="installation"):
    main_inv = main_rop + main_q
    sec_inv = sec_rop + sec_q

    main_pipeline = []
    sec_pipeline = []
    main_backlog = 0 
    history = []

    for day in range(warmup + days):
        # 1. Main receives from Supplier
        for order in main_pipeline:
            order['time'] -= 1
            if order['time'] <= 0: main_inv += order['qty']
        main_pipeline = [o for o in main_pipeline if o['time'] > 0]

        # 2. Main fulfills Backlogs to Secondary
        while main_backlog >= sec_q and main_inv >= sec_q:
            main_inv -= sec_q
            main_backlog -= sec_q
            sec_pipeline.append({'qty': sec_q, 'time': sec_lt})

        # 3. Secondary receives from Main
        for order in sec_pipeline:
            order['time'] -= 1
            if order['time'] <= 0: sec_inv += order['qty']
        sec_pipeline = [o for o in sec_pipeline if o['time'] > 0]

        # 4. Secondary Fulfills Customer Demand
        sec_inv -= sec_demand
        delayed_by_main = True if (sec_inv < 0 and main_backlog > 0) else False

        # 5. Secondary Orders from Main (Always Local Position)
        sec_pos = sec_inv + sum(o['qty'] for o in sec_pipeline) + main_backlog
        if sec_pos <= sec_rop:
            if main_inv >= sec_q:
                main_inv -= sec_q
                sec_pipeline.append({'qty': sec_q, 'time': sec_lt})
            else:
                main_backlog += sec_q 

        # 6. Main Orders from Supplier (The Strategy Split)
        if strategy == "installation":
            # Scenario 2: Main only looks at its own stock and backlogs
            main_trigger_pos = main_inv + sum(o['qty'] for o in main_pipeline) - main_backlog
        elif strategy == "echelon":
            # Scenario 3: Main looks at the ENTIRE system (Main Inv + Sec Inv + All Pipelines)
            main_trigger_pos = main_inv + sec_inv + sum(o['qty'] for o in main_pipeline) + sum(o['qty'] for o in sec_pipeline)
            
        if main_trigger_pos <= main_rop:
            main_pipeline.append({'qty': main_q, 'time': main_lt})

        # Record Data
        if day >= warmup:
            history.append({
                'Day': day - warmup + 1,
                'Sec On-Hand': max(0, sec_inv),
                'Sec Pipeline': sum(o['qty'] for o in sec_pipeline),
                'Main On-Hand': max(0, main_inv),
                'Main Pipeline': sum(o['qty'] for o in main_pipeline),
                'Stockout_Blamed_On_Main': delayed_by_main
            })
            
    return pd.DataFrame(history)

# --- Plotly Helper Function ---
def render_interactive_chart(df, y_cols, colors):
    """Renders a Plotly chart with interactive legends and step-lines for pipelines."""
    fig = go.Figure()
    
    for i, col in enumerate(y_cols):
        is_pipeline = 'Pipeline' in col
        
        fig.add_trace(go.Scatter(
            x=df['Day'], 
            y=df[col], 
            mode='lines', 
            name=col, 
            line=dict(color=colors[i], width=2 if not is_pipeline else 3),
            line_shape='hv' if is_pipeline else 'linear', # 'hv' creates the step effect
            opacity=0.8 if is_pipeline else 1.0
        ))
        
    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Units",
        legend_title="Metrics (Click to Hide/Show)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)' # Makes the chart background clean
    )
    
    # Grid lines disabled for clean interpretation
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
st.markdown("Adjust these sliders to define the simulation horizon. The **Warm-up Period** runs invisibly to stabilize the supply chain.")
col_g1, col_g2 = st.columns(2)
warmup_days = col_g1.number_input("Warm-up Period (Days hidden to stabilize)", min_value=0, value=150, step=30)
sim_days = col_g2.number_input("Display Period (Days plotted)", min_value=10, value=90, step=30)
st.markdown("---")

# ==========================================
# TAB LAYOUT
# ==========================================
tab1, tab2, tab3 = st.tabs(["🏢 Scenario 1: Single Central", "🏬 Scenario 2: Two-Stage (Local ROP)", "🌍 Scenario 3: Multi-Echelon"])

# --- TAB 1: SINGLE WAREHOUSE ---
with tab1:
    st.markdown("### Single Central Warehouse")
    col1a, col1b, col1c, col1d = st.columns(4)
    s1_demand = col1a.number_input("Avg Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s1_d")
    s1_std_dev = col1b.number_input("Demand Std Dev", min_value=0.0, value=20.0, step=5.0, key="s1_std")
    s1_lead_time = col1c.number_input("Lead Time (days)", min_value=0.0, value=7.0, step=1.0, key="s1_lt")
    s1_service_level = col1d.slider("Fill Rate Goal", 0.50, 0.999, 0.95, key="s1_sl")
    
    col1e, col1f, col1g = st.columns(3)
    s1_cost = col1e.number_input("Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s1_cost")
    s1_q = col1f.number_input("Order Qty (Q)", min_value=1, value=500, step=50, key="s1_q")
    
    rec_s1_rop, rec_s1_ss = get_recommendations(s1_demand, s1_std_dev, s1_lead_time, s1_service_level)
    
    # Input with suggested ROP below
    s1_actual_rop = col1g.number_input("Actual ROP", min_value=0, value=int(rec_s1_rop), step=10, key="s1_act")
    col1g.caption(f"💡 Suggested: **{rec_s1_rop:,.0f}**")
    
    st.markdown("---")
    s1_act_ss, s1_avg_wc, s1_max_wc = get_financials(s1_actual_rop, s1_demand, s1_lead_time, s1_q, s1_cost)
    m1, m2 = st.columns(2)
    m1.metric("Avg Working Capital", f"${s1_avg_wc:,.2f}")
    m2.metric("Realized Safety Stock", f"{s1_act_ss:,.0f} units")
    
    df_s1 = simulate_single_stage(s1_demand, s1_actual_rop, s1_act_ss, s1_q, s1_lead_time, sim_days, warmup_days)
    
    st.markdown(f"#### 📈 {sim_days}-Day Simulation")
    render_interactive_chart(df_s1, ['On-Hand Inventory', 'Pipeline Inventory', 'ROP Limit'], ["#1f77b4", "#9467bd", "#ff7f0e"])
    
    with st.expander("📋 View Daily Data Table"):
        st.dataframe(df_s1, use_container_width=True)

# --- TAB 2: TWO-STAGE (LOCAL ROP) ---
with tab2:
    st.markdown("### Linked Two-Stage System (Installation Strategy)")
    st.caption("The Main Warehouse only triggers replenishments based on its local inventory levels.")
    
    s2_cost = st.number_input("Shared Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s2_cost")
    
    st.markdown("#### Secondary (Front-line)")
    c2a, c2b, c2c, c2d = st.columns(4)
    s2_sec_demand = c2a.number_input("Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s2_sec_d")
    s2_sec_std = c2b.number_input("Std Dev", min_value=0.0, value=20.0, step=5.0, key="s2_sec_std")
    s2_sec_lt = c2c.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s2_sec_lt")
    s2_sec_sl = c2d.slider("Sec Fill Rate", 0.50, 0.999, 0.95, key="s2_sec_sl")
    
    c2e, c2f = st.columns(2)
    s2_sec_q = c2e.number_input("Sec Order Qty", min_value=1, value=300, step=50, key="s2_sec_q")
    rec_sec_rop, _ = get_recommendations(s2_sec_demand, s2_sec_std, s2_sec_lt, s2_sec_sl)
    
    s2_sec_actual_rop = c2f.number_input("Sec Actual ROP", min_value=0, value=int(rec_sec_rop), step=10, key="s2_sec_act")
    c2f.caption(f"💡 Suggested: **{rec_sec_rop:,.0f}**")
    
    s2_sec_act_ss, s2_sec_avg_wc, s2_sec_max_wc = get_financials(s2_sec_actual_rop, s2_sec_demand, s2_sec_lt, s2_sec_q, s2_cost)
    
    st.markdown("#### Main (Hub)")
    c3a, c3b, c3c, c3d = st.columns(4)
    s2_main_demand = c3a.number_input("Agg Demand", min_value=0.0, value=100.0, step=10.0, key="s2_main_d")
    s2_main_std = c3b.number_input("Agg Std Dev", min_value=0.0, value=25.0, step=5.0, key="s2_main_std")
    s2_main_lt = c3c.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s2_main_lt")
    s2_main_sl = c3d.slider("Main Fill Rate", 0.50, 0.999, 0.98, key="s2_main_sl")
    
    c3e, c3f = st.columns(2)
    s2_main_q = c3e.number_input("Main Order Qty", min_value=1, value=800, step=50, key="s2_main_q")
    rec_main_rop, _ = get_recommendations(s2_main_demand, s2_main_std, s2_main_lt, s2_main_sl)
    
    s2_main_actual_rop = c3f.number_input("Main Actual ROP", min_value=0, value=int(rec_main_rop), step=10, key="s2_main_act")
    c3f.caption(f"💡 Suggested: **{rec_main_rop:,.0f}**")
    
    s2_main_act_ss, s2_main_avg_wc, s2_main_max_wc = get_financials(s2_main_actual_rop, s2_main_demand, s2_main_lt, s2_main_q, s2_cost)

    st.markdown("---")
    df_s2 = simulate_two_stage(s2_sec_demand, s2_sec_actual_rop, s2_sec_q, s2_sec_lt, s2_main_actual_rop, s2_main_q, s2_main_lt, sim_days, warmup_days, "installation")
    
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("System Avg WC", f"${(s2_sec_avg_wc + s2_main_avg_wc):,.2f}")
    sm2.metric("Effective Fill Rate", f"{(s2_sec_sl * s2_main_sl)*100:.1f}%")
    sm3.metric("Main Delay Stockouts", f"{df_s2['Stockout_Blamed_On_Main'].sum()} Days")

    st.markdown(f"#### 📈 {sim_days}-Day Simulation")
    render_interactive_chart(df_s2, ['Sec On-Hand', 'Sec Pipeline', 'Main On-Hand', 'Main Pipeline'], ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"])
    
    with st.expander("📋 View Daily Data Table"):
        st.dataframe(df_s2, use_container_width=True)

# --- TAB 3: ECHELON SYSTEM ---
with tab3:
    st.markdown("### Multi-Echelon System (Echelon Strategy)")
    st.caption("The Main Warehouse calculates its ROP based on the *total system inventory* (Main + Secondary + Pipelines).")
    
    s3_cost = st.number_input("Shared Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s3_cost")
    
    st.markdown("#### Secondary (Front-line)")
    c4a, c4b, c4c, c4d = st.columns(4)
    s3_sec_demand = c4a.number_input("Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s3_sec_d")
    s3_sec_std = c4b.number_input("Std Dev", min_value=0.0, value=20.0, step=5.0, key="s3_sec_std")
    s3_sec_lt = c4c.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s3_sec_lt")
    s3_sec_sl = c4d.slider("Sec Fill Rate", 0.50, 0.999, 0.95, key="s3_sec_sl")
    
    c4e, c4f = st.columns(2)
    s3_sec_q = c4e.number_input("Sec Order Qty", min_value=1, value=300, step=50, key="s3_sec_q")
    rec_s3_sec_rop, _ = get_recommendations(s3_sec_demand, s3_sec_std, s3_sec_lt, s3_sec_sl)
    
    s3_sec_actual_rop = c4f.number_input("Sec Actual ROP", min_value=0, value=int(rec_s3_sec_rop), step=10, key="s3_sec_act")
    c4f.caption(f"💡 Suggested: **{rec_s3_sec_rop:,.0f}**")
    
    s3_sec_act_ss, s3_sec_avg_wc, s3_sec_max_wc = get_financials(s3_sec_actual_rop, s3_sec_demand, s3_sec_lt, s3_sec_q, s3_cost)
    
    st.markdown("#### Main (Echelon Evaluator)")
    c5a, c5b, c5c, c5d = st.columns(4)
    s3_main_demand = c5a.number_input("Agg Demand", min_value=0.0, value=100.0, step=10.0, key="s3_main_d")
    s3_main_std = c5b.number_input("Agg Std Dev", min_value=0.0, value=25.0, step=5.0, key="s3_main_std")
    s3_main_lt = c5c.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s3_main_lt")
    s3_main_sl = c5d.slider("Main Fill Rate", 0.50, 0.999, 0.98, key="s3_main_sl")
    
    c5e, c5f = st.columns(2)
    s3_main_q = c5e.number_input("Main Order Qty", min_value=1, value=800, step=50, key="s3_main_q")
    
    # ECHELON ROP RECOMMENDATION: Sec ROP + Main Lead Time Demand + Main Safety Stock
    _, main_base_ss = get_recommendations(s3_main_demand, s3_main_std, s3_main_lt, s3_main_sl)
    rec_echelon_rop = rec_s3_sec_rop + (s3_main_demand * s3_main_lt) + main_base_ss
    
    s3_echelon_actual_rop = c5f.number_input("Echelon Actual ROP", min_value=0, value=int(rec_echelon_rop), step=10, key="s3_ech_act")
    c5f.caption(f"💡 Suggested: **{rec_echelon_rop:,.0f}**")
    
    s3_main_act_ss, s3_main_avg_wc, s3_main_max_wc = get_financials(s3_echelon_actual_rop - rec_s3_sec_rop, s3_main_demand, s3_main_lt, s3_main_q, s3_cost)

    st.markdown("---")
    df_s3 = simulate_two_stage(s3_sec_demand, s3_sec_actual_rop, s3_sec_q, s3_sec_lt, s3_echelon_actual_rop, s3_main_q, s3_main_lt, sim_days, warmup_days, "echelon")
    
    tm1, tm2, tm3 = st.columns(3)
    tm1.metric("System Avg WC", f"${(s3_sec_avg_wc + s3_main_avg_wc):,.2f}")
    tm2.metric("Effective Fill Rate", f"{(s3_sec_sl * s3_main_sl)*100:.1f}%")
    tm3.metric("Main Delay Stockouts", f"{df_s3['Stockout_Blamed_On_Main'].sum()} Days")

    st.markdown(f"#### 📈 {sim_days}-Day Simulation")
    render_interactive_chart(df_s3, ['Sec On-Hand', 'Sec Pipeline', 'Main On-Hand', 'Main Pipeline'], ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"])
    
    with st.expander("📋 View Daily Data Table"):
        st.dataframe(df_s3, use_container_width=True)

# ==========================================
# MASTER COMPARISON TABLE
# ==========================================
st.markdown("---")
st.header("📋 Master Comparison Summary")

comparison_data = {
    "Metric": ["Target Fill Rate", "Order Quantity (Q)", "Recommended ROP", "Actual Set ROP", "Avg Working Capital", "Stockout Days (Due to Main)"],
    "S1: Central": [f"{s1_service_level*100:.1f}%", f"{s1_q:,.0f}", f"{rec_s1_rop:,.0f}", f"{s1_actual_rop:,.0f}", f"${s1_avg_wc:,.0f}", "N/A"],
    "S2: Secondary": [f"{s2_sec_sl*100:.1f}%", f"{s2_sec_q:,.0f}", f"{rec_sec_rop:,.0f}", f"{s2_sec_actual_rop:,.0f}", f"${s2_sec_avg_wc:,.0f}", "—"],
    "S2: Main": [f"{s2_main_sl*100:.1f}%", f"{s2_main_q:,.0f}", f"{rec_main_rop:,.0f}", f"{s2_main_actual_rop:,.0f}", f"${s2_main_avg_wc:,.0f}", f"{df_s2['Stockout_Blamed_On_Main'].sum()}"],
    "S3: Secondary": [f"{s3_sec_sl*100:.1f}%", f"{s3_sec_q:,.0f}", f"{rec_s3_sec_rop:,.0f}", f"{s3_sec_actual_rop:,.0f}", f"${s3_sec_avg_wc:,.0f}", "—"],
    "S3: Main (Echelon)": [f"{s3_main_sl*100:.1f}%", f"{s3_main_q:,.0f}", f"{rec_echelon_rop:,.0f}", f"{s3_echelon_actual_rop:,.0f}", f"${s3_main_avg_wc:,.0f}", f"{df_s3['Stockout_Blamed_On_Main'].sum()}"]
}
st.table(pd.DataFrame(comparison_data).set_index("Metric"))

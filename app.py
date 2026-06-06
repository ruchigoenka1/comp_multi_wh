import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd

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

# --- Scenario 1: Single Warehouse Simulation ---
def simulate_single_stage(demand, rop, ss, q, lead_time, days=90, warmup=150):
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
            pipeline_qty = sum(o['qty'] for o in pipeline)
            history.append({
                'Day': day - warmup + 1,
                'On-Hand Inventory': max(0, inventory),
                'Pipeline Inventory': pipeline_qty,
                'Backorder Qty': abs(inventory) if inventory < 0 else 0,
                'ROP': rop,
                'SS': ss
            })
    return pd.DataFrame(history)

# --- Scenario 2: Linked Two-Stage Simulation ---
def simulate_two_stage(sec_demand, sec_rop, sec_q, sec_lt, main_rop, main_q, main_lt, days=90, warmup=150):
    main_inv = main_rop + main_q
    sec_inv = sec_rop + sec_q

    main_pipeline = []
    sec_pipeline = []
    main_backlog = 0 
    history = []

    for day in range(warmup + days):
        # 1. Process Supplier -> Main Arrivals
        for order in main_pipeline:
            order['time'] -= 1
            if order['time'] <= 0: main_inv += order['qty']
        main_pipeline = [o for o in main_pipeline if o['time'] > 0]

        # Fulfill Backlogs from Main to Secondary
        while main_backlog >= sec_q and main_inv >= sec_q:
            main_inv -= sec_q
            main_backlog -= sec_q
            sec_pipeline.append({'qty': sec_q, 'time': sec_lt})

        # 2. Process Main -> Secondary Arrivals
        for order in sec_pipeline:
            order['time'] -= 1
            if order['time'] <= 0: sec_inv += order['qty']
        sec_pipeline = [o for o in sec_pipeline if o['time'] > 0]

        # 3. Fulfill Daily Customer Demand at Secondary
        sec_inv -= sec_demand
        delayed_by_main = True if (sec_inv < 0 and main_backlog > 0) else False

        # 4. Check Reorder Trigger at Secondary
        sec_pos = sec_inv + sum(o['qty'] for o in sec_pipeline) + main_backlog
        if sec_pos <= sec_rop:
            if main_inv >= sec_q:
                main_inv -= sec_q
                sec_pipeline.append({'qty': sec_q, 'time': sec_lt})
            else:
                main_backlog += sec_q 

        # 5. Check Reorder Trigger at Main
        main_pos = main_inv + sum(o['qty'] for o in main_pipeline) - main_backlog
        if main_pos <= main_rop:
            main_pipeline.append({'qty': main_q, 'time': main_lt})

        # 6. Record Data
        if day >= warmup:
            sec_pipeline_qty = sum(o['qty'] for o in sec_pipeline)
            main_pipeline_qty = sum(o['qty'] for o in main_pipeline)
            
            history.append({
                'Day': day - warmup + 1,
                'Sec_On_Hand': max(0, sec_inv),
                'Sec_Pipeline': sec_pipeline_qty,
                'Main_On_Hand': max(0, main_inv),
                'Main_Pipeline': main_pipeline_qty,
                'Sec_Backordered': abs(sec_inv) if sec_inv < 0 else 0,
                'Main_Backlog_to_Sec': main_backlog,
                'Stockout_Blamed_On_Main': delayed_by_main,
                'Sec_ROP': sec_rop,
                'Main_ROP': main_rop
            })
            
    return pd.DataFrame(history)

# --- App Configuration ---
st.set_page_config(page_title="Supply Chain Multi-Scenario Optimizer", layout="wide")
st.title("📦 Supply Chain Financial & Inventory Metrics")
st.markdown("Use the collapsible panels below to configure and run simulations for different warehouse structures.")

# ==========================================
# SCENARIO 1: SINGLE WAREHOUSE
# ==========================================
with st.expander("🏢 Scenario 1: Single Central Warehouse", expanded=True):
    st.markdown("### Inventory & Financial Parameters")
    
    col1a, col1b, col1c, col1d = st.columns(4)
    s1_demand = col1a.number_input("Avg Demand (units/day)", min_value=0.0, value=100.0, step=10.0, key="s1_d")
    s1_std_dev = col1b.number_input("Demand Std Dev", min_value=0.0, value=20.0, step=5.0, key="s1_std")
    s1_lead_time = col1c.number_input("Lead Time (days)", min_value=0.0, value=7.0, step=1.0, key="s1_lt")
    s1_service_level = col1d.slider("Target Fill Rate (S1)", min_value=0.50, max_value=0.999, value=0.95, step=0.01, key="s1_sl")
    
    col1e, col1f, col1g = st.columns(3)
    s1_cost = col1e.number_input("Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s1_cost")
    s1_q = col1f.number_input("Order Qty (Q)", min_value=1, value=500, step=50, key="s1_q")
    
    rec_s1_rop, rec_s1_ss = get_recommendations(s1_demand, s1_std_dev, s1_lead_time, s1_service_level)
    s1_actual_rop = col1g.number_input("Actual ROP (S1)", min_value=0, value=int(rec_s1_rop), step=10, key="s1_act_rop", help=f"Recommended: {rec_s1_rop:.0f}")
    
    st.markdown("---")
    st.markdown("### 🎯 Scenario 1 Metrics & Simulation")
    s1_act_ss, s1_avg_wc, s1_max_wc = get_financials(s1_actual_rop, s1_demand, s1_lead_time, s1_q, s1_cost)
    
    m1, m2 = st.columns(2)
    m1.metric("Avg Working Capital", f"${s1_avg_wc:,.2f}")
    m2.metric("Realized Safety Stock", f"{s1_act_ss:,.0f} units")
    
    df_s1 = simulate_single_stage(s1_demand, s1_actual_rop, s1_act_ss, s1_q, s1_lead_time)
    
    st.markdown("#### 📈 90-Day Simulation (Inventory vs Pipeline)")
    # Added Pipeline Inventory to the chart with a distinct purple color
    st.line_chart(df_s1.set_index('Day')[['On-Hand Inventory', 'Pipeline Inventory', 'ROP']], color=["#1f77b4", "#9467bd", "#ff7f0e"], height=350)
    
    with st.expander("📋 View Daily Data Table for Scenario 1"):
        st.dataframe(df_s1, use_container_width=True)

# ==========================================
# SCENARIO 2: TWO-STAGE WAREHOUSE
# ==========================================
with st.expander("🏬 Scenario 2: Linked Two-Stage System", expanded=False):
    st.markdown("### Shared Financials")
    s2_cost = st.number_input("Shared Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s2_cost")
    st.markdown("---")
    
    # --- Secondary ---
    st.markdown("#### 1. Secondary Warehouse (Customer-Facing)")
    col2a, col2b, col2c, col2d = st.columns(4)
    s2_sec_demand = col2a.number_input("Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s2_sec_d")
    s2_sec_std = col2b.number_input("Std Dev", min_value=0.0, value=20.0, step=5.0, key="s2_sec_std")
    s2_sec_lt = col2c.number_input("Transit from Main", min_value=0.0, value=3.0, step=1.0, key="s2_sec_lt")
    s2_sec_sl = col2d.slider("Target Fill Rate (Sec)", min_value=0.50, max_value=0.999, value=0.95, step=0.01, key="s2_sec_sl")
    
    col2e, col2f = st.columns(2)
    s2_sec_q = col2e.number_input("Sec Order Qty", min_value=1, value=300, step=50, key="s2_sec_q")
    rec_sec_rop, rec_sec_ss = get_recommendations(s2_sec_demand, s2_sec_std, s2_sec_lt, s2_sec_sl)
    s2_sec_actual_rop = col2f.number_input("Actual Sec ROP", min_value=0, value=int(rec_sec_rop), step=10, key="s2_sec_act_rop", help=f"Recommended: {rec_sec_rop:.0f}")
    
    s2_sec_act_ss, s2_sec_avg_wc, s2_sec_max_wc = get_financials(s2_sec_actual_rop, s2_sec_demand, s2_sec_lt, s2_sec_q, s2_cost)
    
    st.markdown("---")
    
    # --- Main ---
    st.markdown("#### 2. Main Warehouse (Supplier-Facing)")
    col3a, col3b, col3c, col3d = st.columns(4)
    s2_main_demand = col3a.number_input("Aggregated Demand", min_value=0.0, value=100.0, step=10.0, key="s2_main_d")
    s2_main_std = col3b.number_input("Aggregated Std Dev", min_value=0.0, value=25.0, step=5.0, key="s2_main_std")
    s2_main_lt = col3c.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s2_main_lt")
    s2_main_sl = col3d.slider("Target Fill Rate (Main)", min_value=0.50, max_value=0.999, value=0.98, step=0.01, key="s2_main_sl")
    
    col3e, col3f = st.columns(2)
    s2_main_q = col3e.number_input("Main Order Qty", min_value=1, value=800, step=50, key="s2_main_q")
    rec_main_rop, rec_main_ss = get_recommendations(s2_main_demand, s2_main_std, s2_main_lt, s2_main_sl)
    s2_main_actual_rop = col3f.number_input("Actual Main ROP", min_value=0, value=int(rec_main_rop), step=10, key="s2_main_act_rop", help=f"Recommended: {rec_main_rop:.0f}")
    
    s2_main_act_ss, s2_main_avg_wc, s2_main_max_wc = get_financials(s2_main_actual_rop, s2_main_demand, s2_main_lt, s2_main_q, s2_cost)

    st.markdown("---")
    st.markdown("### 🎯 Scenario 2 Metrics & Simulation")
    
    df_s2 = simulate_two_stage(s2_sec_demand, s2_sec_actual_rop, s2_sec_q, s2_sec_lt,
                               s2_main_actual_rop, s2_main_q, s2_main_lt)
    
    main_blame_days = df_s2['Stockout_Blamed_On_Main'].sum()

    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("System Avg WC", f"${(s2_sec_avg_wc + s2_main_avg_wc):,.2f}")
    sm2.metric("Effective System Fill Rate", f"{(s2_sec_sl * s2_main_sl)*100:.1f}%")
    sm3.metric("Stockouts Caused by Main Whse Delay", f"{main_blame_days} Days")

    st.markdown("#### 📈 90-Day System Simulation (Inventory vs Pipeline)")
    # Grouped On-Hand vs Pipeline. Dark Blue/Green for On-Hand, Light Blue/Green for Pipeline
    st.line_chart(df_s2.set_index('Day')[['Sec_On_Hand', 'Sec_Pipeline', 'Main_On_Hand', 'Main_Pipeline']], 
                  color=["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"], height=350)
    
    with st.expander("📋 View Daily Data Table for Scenario 2"):
        st.dataframe(df_s2, use_container_width=True)

# ==========================================
# MASTER COMPARISON TABLE
# ==========================================
st.markdown("---")
st.header("📋 Cross-Scenario Parameter & Metric Comparison")

comparison_data = {
    "Parameter / Metric": [
        "**Target Fill Rate**",
        "Average Demand (units/day)",
        "Standard Deviation",
        "Lead Time (days)",
        "Order Quantity (Q)",
        "Unit Cost ($)",
        "Recommended ROP",
        "**Actual ROP**",
        "Realized Safety Stock",
        "**Average Working Capital**",
        "**Max Working Capital**"
    ],
    "Scenario 1: Single Central": [
        f"{s1_service_level*100:.1f}%",
        f"{s1_demand:,.0f}",
        f"{s1_std_dev:,.0f}",
        f"{s1_lead_time:,.0f}",
        f"{s1_q:,.0f}",
        f"${s1_cost:,.2f}",
        f"{rec_s1_rop:,.0f}",
        f"{s1_actual_rop:,.0f}",
        f"{s1_act_ss:,.0f}",
        f"${s1_avg_wc:,.2f}",
        f"${s1_max_wc:,.2f}"
    ],
    "Scenario 2: Secondary (Front-line)": [
        f"{s2_sec_sl*100:.1f}%",
        f"{s2_sec_demand:,.0f}",
        f"{s2_sec_std:,.0f}",
        f"{s2_sec_lt:,.0f}",
        f"{s2_sec_q:,.0f}",
        f"${s2_cost:,.2f}",
        f"{rec_sec_rop:,.0f}",
        f"{s2_sec_actual_rop:,.0f}",
        f"{s2_sec_act_ss:,.0f}",
        f"${s2_sec_avg_wc:,.2f}",
        f"${s2_sec_max_wc:,.2f}"
    ],
    "Scenario 2: Main (Hub)": [
        f"{s2_main_sl*100:.1f}%",
        f"{s2_main_demand:,.0f}",
        f"{s2_main_std:,.0f}",
        f"{s2_main_lt:,.0f}",
        f"{s2_main_q:,.0f}",
        f"${s2_cost:,.2f}",
        f"{rec_main_rop:,.0f}",
        f"{s2_main_actual_rop:,.0f}",
        f"{s2_main_act_ss:,.0f}",
        f"${s2_main_avg_wc:,.2f}",
        f"${s2_main_max_wc:,.2f}"
    ]
}

st.table(pd.DataFrame(comparison_data).set_index("Parameter / Metric"))

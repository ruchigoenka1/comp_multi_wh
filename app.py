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
            history.append({
                'Day': day - warmup + 1,
                'On-Hand Inventory': max(0, inventory),
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
    main_backlog = 0 # Orders from Sec waiting because Main is out of stock
    history = []

    for day in range(warmup + days):
        # 1. Process Supplier -> Main Arrivals
        for order in main_pipeline:
            order['time'] -= 1
            if order['time'] <= 0: main_inv += order['qty']
        main_pipeline = [o for o in main_pipeline if o['time'] > 0]

        # Fulfill Backlogs from Main to Secondary (if stock is now available)
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
        
        # Track if Secondary is stocked out explicitly because Main hasn't shipped
        delayed_by_main = True if (sec_inv < 0 and main_backlog > 0) else False

        # 4. Check Reorder Trigger at Secondary
        sec_pos = sec_inv + sum(o['qty'] for o in sec_pipeline) + main_backlog
        if sec_pos <= sec_rop:
            if main_inv >= sec_q:
                main_inv -= sec_q
                sec_pipeline.append({'qty': sec_q, 'time': sec_lt})
            else:
                main_backlog += sec_q # Main doesn't have stock, order is backlogged

        # 5. Check Reorder Trigger at Main
        main_pos = main_inv + sum(o['qty'] for o in main_pipeline) - main_backlog
        if main_pos <= main_rop:
            main_pipeline.append({'qty': main_q, 'time': main_lt})

        # 6. Record Data
        if day >= warmup:
            history.append({
                'Day': day - warmup + 1,
                'Sec_On_Hand': max(0, sec_inv),
                'Main_On_Hand': max(0, main_inv),
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

col1, col2 = st.columns(2)

# ==========================================
# SCENARIO 1: SINGLE WAREHOUSE
# ==========================================
with col1:
    st.header("🏢 Scenario 1: Single Central")
    st.markdown("---")
    
    s1_demand = st.number_input("Avg Demand (units/day)", min_value=0.0, value=100.0, step=10.0, key="s1_d")
    s1_std_dev = st.number_input("Demand Std Dev", min_value=0.0, value=20.0, step=5.0, key="s1_std")
    s1_lead_time = st.number_input("Lead Time (days)", min_value=0.0, value=7.0, step=1.0, key="s1_lt")
    s1_service_level = st.slider("Target Fill Rate (S1)", min_value=0.50, max_value=0.999, value=0.95, step=0.01, key="s1_sl")
    s1_cost = st.number_input("Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s1_cost")
    s1_q = st.number_input("Order Qty (Q)", min_value=1, value=500, step=50, key="s1_q")
    
    rec_s1_rop, rec_s1_ss = get_recommendations(s1_demand, s1_std_dev, s1_lead_time, s1_service_level)
    s1_actual_rop = st.number_input("Actual ROP (S1)", min_value=0, value=int(rec_s1_rop), step=10, key="s1_act_rop", help=f"Recommended: {rec_s1_rop:.0f}")
    
    s1_act_ss, s1_avg_wc, s1_max_wc = get_financials(s1_actual_rop, s1_demand, s1_lead_time, s1_q, s1_cost)
    
    st.markdown("#### Scenario 1 Matrices")
    m1, m2 = st.columns(2)
    m1.metric("Avg Working Capital", f"${s1_avg_wc:,.2f}")
    m2.metric("Realized Safety Stock", f"{s1_act_ss:,.0f} units")
    
    df_s1 = simulate_single_stage(s1_demand, s1_actual_rop, s1_act_ss, s1_q, s1_lead_time)
    
    st.markdown("#### 📈 90-Day Simulation")
    st.line_chart(df_s1.set_index('Day')[['On-Hand Inventory', 'ROP']], color=["#1f77b4", "#ff7f0e"])
    
    with st.expander("📋 View Scenario 1 Daily Data Table"):
        st.dataframe(df_s1, use_container_width=True)

# ==========================================
# SCENARIO 2: TWO-STAGE WAREHOUSE
# ==========================================
with col2:
    st.header("🏬 Scenario 2: Two-Stage System")
    st.markdown("---")
    
    s2_cost = st.number_input("Shared Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s2_cost")
    
    # --- Secondary ---
    st.subheader("1. Secondary Whse (Customer-Facing)")
    sec_dem, sec_std = st.columns(2)
    s2_sec_demand = sec_dem.number_input("Demand/Day", min_value=0.0, value=100.0, step=10.0, key="s2_sec_d")
    s2_sec_std = sec_std.number_input("Std Dev", min_value=0.0, value=20.0, step=5.0, key="s2_sec_std")
    
    sec_lt, sec_q = st.columns(2)
    s2_sec_lt = sec_lt.number_input("Transit from Main", min_value=0.0, value=3.0, step=1.0, key="s2_sec_lt")
    s2_sec_q = sec_q.number_input("Sec Order Qty", min_value=1, value=300, step=50, key="s2_sec_q")
    s2_sec_sl = st.slider("Target Fill Rate (Sec)", min_value=0.50, max_value=0.999, value=0.95, step=0.01, key="s2_sec_sl")
    
    rec_sec_rop, rec_sec_ss = get_recommendations(s2_sec_demand, s2_sec_std, s2_sec_lt, s2_sec_sl)
    s2_sec_actual_rop = st.number_input("Actual Secondary ROP", min_value=0, value=int(rec_sec_rop), step=10, key="s2_sec_act_rop", help=f"Recommended: {rec_sec_rop:.0f}")
    s2_sec_act_ss, s2_sec_avg_wc, s2_sec_max_wc = get_financials(s2_sec_actual_rop, s2_sec_demand, s2_sec_lt, s2_sec_q, s2_cost)
    
    st.markdown("---")
    
    # --- Main ---
    st.subheader("2. Main Whse (Supplier-Facing)")
    s2_main_demand = st.number_input("Aggregated Demand", min_value=0.0, value=100.0, step=10.0, key="s2_main_d")
    s2_main_std = st.number_input("Aggregated Std Dev", min_value=0.0, value=25.0, step=5.0, key="s2_main_std")
    
    main_lt, main_q = st.columns(2)
    s2_main_lt = main_lt.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s2_main_lt")
    s2_main_q = main_q.number_input("Main Order Qty", min_value=1, value=800, step=50, key="s2_main_q")
    s2_main_sl = st.slider("Target Fill Rate (Main)", min_value=0.50, max_value=0.999, value=0.98, step=0.01, key="s2_main_sl")
    
    rec_main_rop, rec_main_ss = get_recommendations(s2_main_demand, s2_main_std, s2_main_lt, s2_main_sl)
    s2_main_actual_rop = st.number_input("Actual Main ROP", min_value=0, value=int(rec_main_rop), step=10, key="s2_main_act_rop", help=f"Recommended: {rec_main_rop:.0f}")
    s2_main_act_ss, s2_main_avg_wc, s2_main_max_wc = get_financials(s2_main_actual_rop, s2_main_demand, s2_main_lt, s2_main_q, s2_cost)

    # Simulation Execution
    df_s2 = simulate_two_stage(s2_sec_demand, s2_sec_actual_rop, s2_sec_q, s2_sec_lt,
                               s2_main_actual_rop, s2_main_q, s2_main_lt)
    
    main_blame_days = df_s2['Stockout_Blamed_On_Main'].sum()

    st.markdown("### 🎯 Scenario 2 Matrices")
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("System Avg WC", f"${(s2_sec_avg_wc + s2_main_avg_wc):,.2f}")
    sm2.metric("Effective Fill Rate", f"{(s2_sec_sl * s2_main_sl)*100:.1f}%")
    sm3.metric("Stockout Days due to Main Delay", f"{main_blame_days} Days")

    st.markdown("#### 📈 90-Day System Simulation")
    st.line_chart(df_s2.set_index('Day')[['Sec_On_Hand', 'Main_On_Hand']], color=["#1f77b4", "#2ca02c"])
    
    with st.expander("📋 View Scenario 2 Daily Linked Data Table"):
        st.dataframe(df_s2, use_container_width=True)

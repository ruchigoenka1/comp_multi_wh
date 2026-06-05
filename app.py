import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd

# --- Core Calculation Functions ---
def get_recommendations(demand, std_dev, lead_time, service_level):
    """Calculates the mathematically recommended ROP and Safety Stock."""
    if lead_time <= 0:
        return 0, 0
    z_score = norm.ppf(service_level)
    safety_stock = max(0, z_score * std_dev * math.sqrt(lead_time))
    rop = max(0, (demand * lead_time) + safety_stock)
    return rop, safety_stock

def get_financials(actual_rop, demand, lead_time, order_qty, unit_cost):
    """Calculates working capital based on the USER'S chosen ROP."""
    # Realized safety stock is whatever buffer remains after lead time demand
    actual_ss = max(0, actual_rop - (demand * lead_time))
    avg_working_capital = ((order_qty / 2) + actual_ss) * unit_cost
    max_working_capital = (order_qty + actual_ss) * unit_cost
    return actual_ss, avg_working_capital, max_working_capital

def simulate_inventory_cycle(demand, rop, ss, q, lead_time, days=90, warmup_days=150):
    """Simulates daily inventory, skipping a warmup period to show stabilized data."""
    inventory = rop + (q / 2) 
    pipeline = []
    
    # 1. WARM-UP PHASE
    for _ in range(warmup_days):
        inventory -= demand
        for order in pipeline:
            order['arrival_in'] -= 1
            if order['arrival_in'] <= 0:
                inventory += order['qty']
        pipeline = [o for o in pipeline if o['arrival_in'] > 0]
        
        inv_position = inventory + sum(o['qty'] for o in pipeline)
        if inv_position <= rop:
            pipeline.append({'qty': q, 'arrival_in': lead_time})
            
    # 2. RECORDING PHASE
    inv_history = []
    for _ in range(days):
        inv_history.append(max(0, inventory)) 
        inventory -= demand
        
        for order in pipeline:
            order['arrival_in'] -= 1
            if order['arrival_in'] <= 0:
                inventory += order['qty']
        pipeline = [o for o in pipeline if o['arrival_in'] > 0]
        
        inv_position = inventory + sum(o['qty'] for o in pipeline)
        if inv_position <= rop:
            pipeline.append({'qty': q, 'arrival_in': lead_time})
            
    df = pd.DataFrame({
        'Day': range(days),
        'On-Hand Inventory': inv_history,
        'Actual Reorder Point (ROP)': [rop] * days,
        'Realized Safety Stock (SS)': [ss] * days
    })
    return df.set_index('Day')

# --- App Configuration ---
st.set_page_config(page_title="Supply Chain Multi-Scenario Optimizer", layout="wide")
st.title("📦 Supply Chain Financial & Inventory Metrics")

col1, col2 = st.columns(2)

# ==========================================
# SCENARIO 1: SINGLE WAREHOUSE
# ==========================================
with col1:
    st.header("🏢 Scenario 1: Single Central Warehouse")
    st.markdown("---")
    
    s1_demand = st.number_input("Average Demand (units/day)", min_value=0.0, value=100.0, step=10.0, key="s1_d")
    s1_std_dev = st.number_input("Standard Dev. of Demand", min_value=0.0, value=20.0, step=5.0, key="s1_std")
    s1_lead_time = st.number_input("Lead Time from Supplier (days)", min_value=0.0, value=7.0, step=1.0, key="s1_lt")
    s1_service_level = st.slider("Target Fill Rate", min_value=0.50, max_value=0.999, value=0.95, step=0.01, key="s1_sl", format="%.2f")
    
    st.markdown("##### Financial Parameters")
    s1_cost = st.number_input("Unit Cost ($/unit)", min_value=0.01, value=50.0, step=5.0, key="s1_cost")
    s1_q = st.number_input("Order Quantity (Q)", min_value=1, value=500, step=50, key="s1_q")
    
    # ROP DECISION BLOCK
    st.markdown("### 🎯 Set Reorder Point")
    rec_s1_rop, rec_s1_ss = get_recommendations(s1_demand, s1_std_dev, s1_lead_time, s1_service_level)
    st.caption(f"💡 *Mathematically Recommended ROP: {rec_s1_rop:,.0f} units*")
    
    # THE RESTORED INPUT BOX
    s1_actual_rop = st.number_input("Actual ROP (Scenario 1)", min_value=0, value=int(rec_s1_rop), step=10, key="s1_actual_rop")
    s1_actual_ss, s1_avg_wc, s1_max_wc = get_financials(s1_actual_rop, s1_demand, s1_lead_time, s1_q, s1_cost)
    
    st.markdown("#### Scenario 1 Matrices")
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Working Capital", f"${s1_avg_wc:,.2f}")
    m2.metric("Max Working Capital", f"${s1_max_wc:,.2f}")
    m3.metric("Realized Safety Stock", f"{s1_actual_ss:,.0f} units")
    
    st.markdown("#### 📈 90-Day Inventory Projection")
    df_s1 = simulate_inventory_cycle(s1_demand, s1_actual_rop, s1_actual_ss, s1_q, s1_lead_time)
    st.line_chart(df_s1, color=["#1f77b4", "#ff7f0e", "#d62728"])

# ==========================================
# SCENARIO 2: TWO-STAGE WAREHOUSE
# ==========================================
with col2:
    st.header("🏬 Scenario 2: Two-Stage System")
    st.markdown("---")
    
    st.markdown("##### Financial Parameters (Shared)")
    s2_cost = st.number_input("Unit Cost ($/unit)", min_value=0.01, value=50.0, step=5.0, key="s2_cost")
    st.markdown("---")
    
    # --- Secondary Warehouse ---
    st.subheader("1. Secondary Warehouse (Front-line)")
    s2_sec_demand = st.number_input("Average Daily Demand", min_value=0.0, value=100.0, step=10.0, key="s2_sec_d")
    s2_sec_std = st.number_input("Std Dev of Demand", min_value=0.0, value=20.0, step=5.0, key="s2_sec_std")
    s2_sec_lt = st.number_input("Lead Time from Main Whse", min_value=0.0, value=3.0, step=1.0, key="s2_sec_lt")
    s2_sec_sl = st.slider("Secondary Fill Rate", min_value=0.50, max_value=0.999, value=0.95, step=0.01, key="s2_sec_sl", format="%.2f")
    s2_sec_q = st.number_input("Secondary Order Qty", min_value=1, value=300, step=50, key="s2_sec_q")
    
    rec_sec_rop, rec_sec_ss = get_recommendations(s2_sec_demand, s2_sec_std, s2_sec_lt, s2_sec_sl)
    st.caption(f"💡 *Mathematically Recommended ROP: {rec_sec_rop:,.0f} units*")
    s2_sec_actual_rop = st.number_input("Actual Secondary ROP", min_value=0, value=int(rec_sec_rop), step=10, key="s2_sec_actual_rop")
    
    s2_sec_actual_ss, s2_sec_avg_wc, s2_sec_max_wc = get_financials(s2_sec_actual_rop, s2_sec_demand, s2_sec_lt, s2_sec_q, s2_cost)
    
    with st.expander("📊 View Secondary Warehouse Projection"):
        df_s2_sec = simulate_inventory_cycle(s2_sec_demand, s2_sec_actual_rop, s2_sec_actual_ss, s2_sec_q, s2_sec_lt)
        st.line_chart(df_s2_sec, color=["#1f77b4", "#ff7f0e", "#d62728"])
        
    st.markdown("---")
    
    # --- Main Warehouse ---
    st.subheader("2. Main Warehouse (Hub)")
    s2_main_demand = st.number_input("Aggregated Daily Demand", min_value=0.0, value=100.0, step=10.0, key="s2_main_d")
    s2_main_std = st.number_input("Aggregated Std Dev", min_value=0.0, value=25.0, step=5.0, key="s2_main_std")
    s2_main_lt = st.number_input("Lead Time from Supplier", min_value=0.0, value=10.0, step=1.0, key="s2_main_lt")
    s2_main_sl = st.slider("Main Target Fill Rate", min_value=0.50, max_value=0.999, value=0.98, step=0.01, key="s2_main_sl", format="%.2f")
    s2_main_q = st.number_input("Main Order Qty", min_value=1, value=800, step=50, key="s2_main_q")
    
    rec_main_rop, rec_main_ss = get_recommendations(s2_main_demand, s2_main_std, s2_main_lt, s2_main_sl)
    st.caption(f"💡 *Mathematically Recommended ROP: {rec_main_rop:,.0f} units*")
    s2_main_actual_rop = st.number_input("Actual Main ROP", min_value=0, value=int(rec_main_rop), step=10, key="s2_main_actual_rop")
    
    s2_main_actual_ss, s2_main_avg_wc, s2_main_max_wc = get_financials(s2_main_actual_rop, s2_main_demand, s2_main_lt, s2_main_q, s2_cost)

    with st.expander("📊 View Main Warehouse Projection"):
        df_s2_main = simulate_inventory_cycle(s2_main_demand, s2_main_actual_rop, s2_main_actual_ss, s2_main_q, s2_main_lt)
        st.line_chart(df_s2_main, color=["#1f77b4", "#ff7f0e", "#d62728"])

    # --- Combined Metrics ---
    st.markdown("### 🎯 Scenario 2 Matrices")
    
    st.markdown("**Individual Breakdown:**")
    st.table({
        "Metric": ["Avg Working Capital", "Max Working Capital", "Fill Rate Goal", "Realized Safety Stock"],
        "Secondary Warehouse": [f"${s2_sec_avg_wc:,.2f}", f"${s2_sec_max_wc:,.2f}", f"{s2_sec_sl*100:.1f}%", f"{s2_sec_actual_ss:,.0f} units"],
        "Main Warehouse": [f"${s2_main_avg_wc:,.2f}", f"${s2_main_max_wc:,.2f}", f"{s2_main_sl*100:.1f}%", f"{s2_main_actual_ss:,.0f} units"]
    })
    
    st.markdown("**Combined System Totals:**")
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Total Avg WC", f"${(s2_sec_avg_wc + s2_main_avg_wc):,.2f}")
    sm2.metric("Total Max WC", f"${(s2_sec_max_wc + s2_main_max_wc):,.2f}")
    sm3.metric("Effective Fill Rate", f"{(s2_sec_sl * s2_main_sl)*100:.1f}%")

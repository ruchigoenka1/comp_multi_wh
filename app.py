import streamlit as st
import numpy as np
from scipy.stats import norm
import math
import pandas as pd

# --- Core Calculation Functions ---
def calculate_metrics(demand, std_dev, lead_time, service_level, unit_cost, order_qty):
    if lead_time <= 0:
        return 0, 0, 0, 0
        
    z_score = norm.ppf(service_level)
    safety_stock = max(0, z_score * std_dev * math.sqrt(lead_time))
    rop = max(0, (demand * lead_time) + safety_stock)
    
    avg_inventory = (order_qty / 2) + safety_stock
    max_inventory = order_qty + safety_stock
    
    avg_working_capital = avg_inventory * unit_cost
    max_working_capital = max_inventory * unit_cost
    
    return rop, safety_stock, avg_working_capital, max_working_capital

def simulate_inventory_cycle(demand, rop, ss, q, lead_time, days=90):
    """Simulates daily inventory levels to create a sawtooth chart."""
    inventory = q + ss  # Start with a full warehouse
    inv_history = []
    pipeline = [] # Tracks orders placed but not yet arrived
    
    for day in range(days):
        inv_history.append(inventory)
        inventory -= demand # Fulfill daily demand
        
        # Process arriving orders
        for order in pipeline:
            order['arrival_in'] -= 1
            if order['arrival_in'] <= 0:
                inventory += order['qty']
        pipeline = [o for o in pipeline if o['arrival_in'] > 0]
        
        # Check if we need to reorder (based on Inventory Position)
        inv_position = inventory + sum(o['qty'] for o in pipeline)
        if inv_position <= rop:
            pipeline.append({'qty': q, 'arrival_in': lead_time})
            
    # Create DataFrame for Streamlit charting
    df = pd.DataFrame({
        'Day': range(days),
        'On-Hand Inventory': inv_history,
        'Reorder Point (ROP)': [rop] * days,
        'Safety Stock (SS)': [ss] * days
    })
    return df.set_index('Day')

# --- App Configuration ---
st.set_page_config(page_title="Supply Chain Multi-Scenario Optimizer", layout="wide")
st.title("📦 Supply Chain Financial & Inventory Metrics Dashboard")
st.markdown("Evaluate working capital efficiency and visualize inventory levels across single and two-stage warehouse structures.")

# --- Layout Columns ---
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
    
    s1_rop, s1_ss, s1_avg_wc, s1_max_wc = calculate_metrics(s1_demand, s1_std_dev, s1_lead_time, s1_service_level, s1_cost, s1_q)
    
    st.markdown("### 🎯 Scenario 1 Metrics")
    st.info(f"**Recommended Reorder Point (ROP):** {s1_rop:,.0f} units *(Safety Stock: {s1_ss:,.0f} units)*")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Working Capital", f"${s1_avg_wc:,.2f}")
    m2.metric("Max Working Capital", f"${s1_max_wc:,.2f}")
    m3.metric("Target Fill Rate", f"{s1_service_level*100:.1f}%")
    
    # Chart
    st.markdown("#### 📈 90-Day Inventory Projection")
    df_s1 = simulate_inventory_cycle(s1_demand, s1_rop, s1_ss, s1_q, s1_lead_time)
    st.line_chart(df_s1, color=["#1f77b4", "#ff7f0e", "#d62728"]) # Blue Inventory, Orange ROP, Red SS

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
    
    s2_sec_rop, s2_sec_ss, s2_sec_avg_wc, s2_sec_max_wc = calculate_metrics(s2_sec_demand, s2_sec_std, s2_sec_lt, s2_sec_sl, s2_cost, s2_sec_q)
    
    # Secondary Chart
    with st.expander("📊 View Secondary Warehouse Projection"):
        st.caption(f"**ROP:** {s2_sec_rop:,.0f} | **SS:** {s2_sec_ss:,.0f}")
        df_s2_sec = simulate_inventory_cycle(s2_sec_demand, s2_sec_rop, s2_sec_ss, s2_sec_q, s2_sec_lt)
        st.line_chart(df_s2_sec, color=["#1f77b4", "#ff7f0e", "#d62728"])
        
    st.markdown("---")
    
    # --- Main Warehouse ---
    st.

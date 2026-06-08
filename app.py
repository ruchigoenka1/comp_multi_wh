import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import io
from collections import deque, defaultdict

# --- Helper: Excel Downloader ---
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Daily Data')
    return output.getvalue()

# --- Core Financial Calculations ---
def get_recommendations(demand, std_dev, lead_time, service_level):
    if lead_time <= 0: return 0, 0
    z_score = norm.ppf(service_level)
    safety_stock = max(0, z_score * std_dev * math.sqrt(lead_time))
    rop = max(0, (demand * lead_time) + safety_stock)
    return rop, safety_stock

def get_periodic_recommendations(demand, std_dev, lead_time, review_period, service_level):
    if lead_time < 0 or review_period <= 0: return 0, 0
    z_score = norm.ppf(service_level)
    risk_period = lead_time + review_period
    safety_stock = max(0, z_score * std_dev * math.sqrt(risk_period))
    target_level = max(0, (demand * risk_period) + safety_stock)
    return target_level, safety_stock

# --- Scenario 1: Single Warehouse Detailed Simulation ---
def simulate_single_stage_detailed(demands, rop, q, lead_time, warmup, allow_partial, track_backlogs):
    total_days = len(demands)
    
    opening_bal = np.zeros(total_days); order_recv = np.zeros(total_days)
    avail_inv = np.zeros(total_days); sales_arr = np.zeros(total_days)
    shortage_arr = np.zeros(total_days); backlogs_arr = np.zeros(total_days)
    closing_bal = np.zeros(total_days); orders_given = np.zeros(total_days)
    pipeline_arr = np.zeros(total_days)
    
    inv = rop * 1.25  # Initialize safely above trigger
    pipe_qty = 0; backlog = 0
    arrivals = np.zeros(total_days + int(lead_time) + 1)
    
    on_hand = deque([{'qty': inv, 'order_t': -999, 'arrive_t': -999}])
    pipe_events = defaultdict(list)
    age_records = []
    daily_inventory_age = []
    
    total_dem = 0; total_sales = 0; stockout_days = 0
    
    for t in range(total_days):
        opening = inv
        arr = arrivals[t]
        pipe_qty -= arr
        
        if t in pipe_events:
            for batch in pipe_events[t]: on_hand.append(batch)
        
        fill_old = 0
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
        
        sales = 0; shortage = 0
        if allow_partial:
            sales = min(avail, dem); shortage = dem - sales; inv = avail - sales
            if track_backlogs: backlog += shortage
        else:
            if avail >= dem:
                sales = dem; shortage = 0; inv = avail - dem
            else:
                sales = 0; shortage = dem; inv = avail
                if track_backlogs: backlog += shortage
                
        units_to_ship = fill_old + sales
        while units_to_ship > 0 and on_hand:
            batch = on_hand.popleft()
            if batch['qty'] <= units_to_ship:
                if batch['order_t'] >= 0: 
                    age_records.append({'Qty': batch['qty'], 'Pipeline Time': batch['arrive_t'] - batch['order_t'], 'Warehouse Time': t - batch['arrive_t']})
                units_to_ship -= batch['qty']
            else:
                if batch['order_t'] >= 0:
                    age_records.append({'Qty': units_to_ship, 'Pipeline Time': batch['arrive_t'] - batch['order_t'], 'Warehouse Time': t - batch['arrive_t']})
                on_hand.appendleft({'qty': batch['qty'] - units_to_ship, 'order_t': batch['order_t'], 'arrive_t': batch['arrive_t']})
                units_to_ship = 0
                
        if t >= warmup:
            total_dem += dem; total_sales += sales
            if sales < dem: stockout_days += 1
            
            day_val = t - warmup + 1
            for b in on_hand:
                if b['order_t'] >= 0:
                    daily_inventory_age.append({'Day': day_val, 'Location': 'Central On-Hand', 'Age': t - b['order_t'], 'Qty': b['qty']})
            for arr_day, batches in pipe_events.items():
                if arr_day > t: 
                    for b in batches:
                        if 0 <= b['order_t'] <= t:
                            daily_inventory_age.append({'Day': day_val, 'Location': 'Supplier Pipeline', 'Age': t - b['order_t'], 'Qty': b['qty']})
            
        order_given = 0
        if (inv + pipe_qty - backlog) <= rop:
            order_given = q
            arrivals[t + int(lead_time)] += q; pipe_qty += q
            pipe_events[t + int(lead_time)].append({'qty': q, 'order_t': t, 'arrive_t': t + int(lead_time)})
            
        opening_bal[t] = opening; order_recv[t] = arr; avail_inv[t] = opening + arr
        sales_arr[t] = sales; shortage_arr[t] = shortage; backlogs_arr[t] = backlog
        closing_bal[t] = inv; orders_given[t] = order_given; pipeline_arr[t] = pipe_qty

    df = pd.DataFrame({
        "Day": np.arange(1, total_days - warmup + 1), "Opening Balance": opening_bal[warmup:], "Order Received": order_recv[warmup:],
        "Available Inv": avail_inv[warmup:], "Demand": demands[warmup:], "Sales": sales_arr[warmup:], "Shortage": shortage_arr[warmup:], 
        "Backlogs": backlogs_arr[warmup:], "Closing Balance": closing_bal[warmup:], "Orders Given": orders_given[warmup:], "Pipeline Inventory": pipeline_arr[warmup:]
    })
    
    vol_fr = total_sales / total_dem if total_dem > 0 else 1.0
    csl = 1 - (stockout_days / (total_days - warmup))
    return df, vol_fr, csl, pd.DataFrame(age_records), pd.DataFrame(daily_inventory_age), total_sales, total_dem, stockout_days

# --- Scenario 2, 3 & 4: Multi-Stage Detailed Simulation ---
def simulate_two_stage_detailed(
    demands, sec_rop, sec_q, sec_lt, main_rop, main_q, main_lt, warmup, 
    sec_allow_partial, sec_track_backlogs, main_allow_partial, main_track_backlogs, 
    strategy="installation", sec_policy="continuous", sec_r=1, sec_s=0, main_policy="continuous", main_r=1, main_s=0
):
    total_days = len(demands)
    
    s_opening = np.zeros(total_days); s_recv = np.zeros(total_days); s_avail = np.zeros(total_days)
    s_sales = np.zeros(total_days); s_short = np.zeros(total_days); s_back = np.zeros(total_days)
    s_close = np.zeros(total_days); s_order = np.zeros(total_days); s_pipe = np.zeros(total_days); s_sup_short = np.zeros(total_days)
    
    m_opening = np.zeros(total_days); m_recv = np.zeros(total_days); m_avail = np.zeros(total_days)
    m_sales = np.zeros(total_days); m_short = np.zeros(total_days); m_back = np.zeros(total_days)
    m_close = np.zeros(total_days); m_order = np.zeros(total_days); m_pipe = np.zeros(total_days)

    main_inv = (main_rop * 1.25) if main_policy == "continuous" else (main_s * 1.25)
    sec_inv = (sec_rop * 1.25) if sec_policy == "continuous" else (sec_s * 1.25)
    
    main_pipe_qty, sec_pipe_qty = 0, 0
    main_backlog_to_sec, sec_backlog = 0, 0
    main_arrivals = np.zeros(total_days + int(main_lt) + 1)
    sec_arrivals = np.zeros(total_days + int(sec_lt) + 1)
    
    main_on_hand = deque([{'qty': main_inv, 'order_t': -999, 'main_arr_t': -999}])
    sec_on_hand = deque([{'qty': sec_inv, 'order_t': -999, 'main_arr_t': -999, 'main_ship_t': -999, 'sec_arr_t': -999}])
    
    main_pipe_events = defaultdict(list)
    sec_pipe_events = defaultdict(list)
    age_records = []
    daily_inventory_age = []
    
    total_dem = 0; total_sales = 0; stockout_days = 0; main_delay_days = 0

    for t in range(total_days):
        ship_old = 0; ship_now = 0
        
        m_opening[t] = main_inv
        m_arr = main_arrivals[t]
        main_inv += m_arr; main_pipe_qty -= m_arr
        
        if t in main_pipe_events:
            for b in main_pipe_events[t]: main_on_hand.append(b)
        
        if main_track_backlogs and main_backlog_to_sec > 0 and main_inv > 0:
            if main_allow_partial: ship_old = min(main_backlog_to_sec, main_inv)
            else: ship_old = main_backlog_to_sec if main_inv >= main_backlog_to_sec else 0
            main_inv -= ship_old; main_backlog_to_sec -= ship_old
            sec_arrivals[t + int(sec_lt)] += ship_old; sec_pipe_qty += ship_old
            
            u_to_ship = ship_old
            while u_to_ship > 0 and main_on_hand:
                b = main_on_hand.popleft()
                if b['qty'] <= u_to_ship:
                    sec_pipe_events[t + int(sec_lt)].append({'qty': b['qty'], 'order_t': b['order_t'], 'main_arr_t': b['main_arr_t'], 'main_ship_t': t, 'sec_arr_t': t + int(sec_lt)})
                    u_to_ship -= b['qty']
                else:
                    sec_pipe_events[t + int(sec_lt)].append({'qty': u_to_ship, 'order_t': b['order_t'], 'main_arr_t': b['main_arr_t'], 'main_ship_t': t, 'sec_arr_t': t + int(sec_lt)})
                    main_on_hand.appendleft({'qty': b['qty'] - u_to_ship, 'order_t': b['order_t'], 'main_arr_t': b['main_arr_t']})
                    u_to_ship = 0
        elif not main_track_backlogs:
            main_backlog_to_sec = 0

        s_opening[t] = sec_inv
        s_arr = sec_arrivals[t]
        sec_pipe_qty -= s_arr
        
        if t in sec_pipe_events:
            for b in sec_pipe_events[t]: sec_on_hand.append(b)
        
        fill_old = 0
        if sec_track_backlogs and sec_backlog > 0:
            if sec_allow_partial: fill_old = min(s_arr, sec_backlog)
            else: fill_old = sec_backlog if s_arr >= sec_backlog else 0
            sec_backlog -= fill_old; s_arr_for_today = s_arr - fill_old
        else:
            s_arr_for_today = s_arr
            if not sec_track_backlogs: sec_backlog = 0
            
        avail = sec_inv + s_arr_for_today
        dem = demands[t]
        
        sales = 0; shortage = 0
        if sec_allow_partial:
            sales = min(avail, dem); shortage = dem - sales; sec_inv = avail - sales
            if sec_track_backlogs: sec_backlog += shortage
        else:
            if avail >= dem:
                sales = dem; shortage = 0; sec_inv = avail - dem
            else:
                sales = 0; shortage = dem; sec_inv = avail
                if sec_track_backlogs: sec_backlog += shortage
                
        units_sold = fill_old + sales
        while units_sold > 0 and sec_on_hand:
            b = sec_on_hand.popleft()
            if b['qty'] <= units_sold:
                if b['order_t'] >= 0:
                    age_records.append({'Qty': b['qty'], 'Main Pipeline Time': b['main_arr_t'] - b['order_t'], 'Main Warehouse Time': b['main_ship_t'] - b['main_arr_t'], 'Sec Pipeline Time': b['sec_arr_t'] - b['main_ship_t'], 'Sec Warehouse Time': t - b['sec_arr_t']})
                units_sold -= b['qty']
            else:
                if b['order_t'] >= 0:
                    age_records.append({'Qty': units_sold, 'Main Pipeline Time': b['main_arr_t'] - b['order_t'], 'Main Warehouse Time': b['main_ship_t'] - b['main_arr_t'], 'Sec Pipeline Time': b['sec_arr_t'] - b['main_ship_t'], 'Sec Warehouse Time': t - b['sec_arr_t']})
                sec_on_hand.appendleft({'qty': b['qty'] - units_sold, 'order_t': b['order_t'], 'main_arr_t': b['main_arr_t'], 'main_ship_t': b['main_ship_t'], 'sec_arr_t': b['sec_arr_t']})
                units_sold = 0

        delayed_by_main = (sec_inv == 0) and (main_backlog_to_sec > 0) and (sales < dem)

        if t >= warmup:
            total_dem += dem; total_sales += sales
            if sales < dem: stockout_days += 1
            if delayed_by_main: main_delay_days += 1
            
            day_val = t - warmup + 1
            for b in main_on_hand:
                if b['order_t'] >= 0: daily_inventory_age.append({'Day': day_val, 'Location': 'Main On-Hand', 'Age': t - b['order_t'], 'Qty': b['qty']})
            for arr_day, batches in main_pipe_events.items():
                if arr_day > t:
                    for b in batches:
                        if 0 <= b['order_t'] <= t: daily_inventory_age.append({'Day': day_val, 'Location': 'Main Pipeline', 'Age': t - b['order_t'], 'Qty': b['qty']})
            for b in sec_on_hand:
                if b['order_t'] >= 0: daily_inventory_age.append({'Day': day_val, 'Location': 'Sec On-Hand', 'Age': t - b['order_t'], 'Qty': b['qty']})
            for arr_day, batches in sec_pipe_events.items():
                if arr_day > t:
                    for b in batches:
                        if 0 <= b['order_t'] <= t: daily_inventory_age.append({'Day': day_val, 'Location': 'Sec Pipeline', 'Age': t - b['order_t'], 'Qty': b['qty']})

        sec_pos = sec_inv + sec_pipe_qty + main_backlog_to_sec - sec_backlog
        sec_order_given = 0; shortage_from_supplier = 0 
        sec_trigger_action = False
        
        if sec_policy == "continuous":
            if sec_pos <= sec_rop:
                sec_trigger_action = True
                sec_order_qty_eval = sec_q
        elif sec_policy == "periodic":
            if t % sec_r == 0:
                if sec_pos < sec_s:
                    sec_trigger_action = True
                    sec_order_qty_eval = sec_s - sec_pos
        
        if sec_trigger_action:
            sec_order_given = sec_order_qty_eval
            if main_allow_partial: ship_now = min(main_inv, sec_order_qty_eval)
            else: ship_now = sec_order_qty_eval if main_inv >= sec_order_qty_eval else 0
            main_inv -= ship_now; shortage_from_supplier = sec_order_qty_eval - ship_now
            if main_track_backlogs: main_backlog_to_sec += shortage_from_supplier
            sec_arrivals[t + int(sec_lt)] += ship_now; sec_pipe_qty += ship_now
            
            u_to_ship = ship_now
            while u_to_ship > 0 and main_on_hand:
                b = main_on_hand.popleft()
                if b['qty'] <= u_to_ship:
                    sec_pipe_events[t + int(sec_lt)].append({'qty': b['qty'], 'order_t': b['order_t'], 'main_arr_t': b['main_arr_t'], 'main_ship_t': t, 'sec_arr_t': t + int(sec_lt)})
                    u_to_ship -= b['qty']
                else:
                    sec_pipe_events[t + int(sec_lt)].append({'qty': u_to_ship, 'order_t': b['order_t'], 'main_arr_t': b['main_arr_t'], 'main_ship_t': t, 'sec_arr_t': t + int(sec_lt)})
                    main_on_hand.appendleft({'qty': b['qty'] - u_to_ship, 'order_t': b['order_t'], 'main_arr_t': b['main_arr_t']})
                    u_to_ship = 0

        pos = (main_inv + sec_inv + main_pipe_qty + sec_pipe_qty - sec_backlog) if strategy == "echelon" else (main_inv + main_pipe_qty - main_backlog_to_sec)
        main_order_given = 0
        main_trigger_action = False
        
        if main_policy == "continuous":
            if pos <= main_rop:
                main_trigger_action = True
                main_order_qty_eval = main_q
        elif main_policy == "periodic":
            if t % main_r == 0:
                if pos < main_s:
                    main_trigger_action = True
                    main_order_qty_eval = main_s - pos
        
        if main_trigger_action:
            main_order_given = main_order_qty_eval
            main_arrivals[t + int(main_lt)] += main_order_qty_eval; main_pipe_qty += main_order_qty_eval
            main_pipe_events[t + int(main_lt)].append({'qty': main_order_qty_eval, 'order_t': t, 'main_arr_t': t + int(main_lt)})
            
        s_recv[t] = s_arr; s_avail[t] = s_opening[t] + s_arr; s_sales[t] = sales; s_short[t] = shortage
        s_back[t] = sec_backlog; s_close[t] = sec_inv; s_order[t] = sec_order_given; s_sup_short[t] = shortage_from_supplier; s_pipe[t] = sec_pipe_qty
        
        m_recv[t] = m_arr; m_avail[t] = m_opening[t] + m_arr; m_sales[t] = ship_old + ship_now; m_short[t] = shortage_from_supplier
        m_back[t] = main_backlog_to_sec; m_close[t] = main_inv; m_order[t] = main_order_given; m_pipe[t] = main_pipe_qty

    df_sec = pd.DataFrame({
        "Day": np.arange(1, total_days - warmup + 1), "Opening Balance": s_opening[warmup:], "Order Received": s_recv[warmup:],
        "Available Inv": s_avail[warmup:], "Demand": demands[warmup:], "Sales": s_sales[warmup:], "Shortage": s_short[warmup:], 
        "Backlogs": s_back[warmup:], "Closing Balance": s_close[warmup:], "Orders Given": s_order[warmup:], 
        "Shortages from Supplier": s_sup_short[warmup:], "Pipeline Inventory": s_pipe[warmup:]
    })
    
    df_main = pd.DataFrame({
        "Day": np.arange(1, total_days - warmup + 1), "Opening Balance": m_opening[warmup:], "Order Received": m_recv[warmup:],
        "Available Inv": m_avail[warmup:], "Demand": s_order[warmup:], "Sales": m_sales[warmup:], "Shortage": m_short[warmup:], 
        "Backlogs": m_back[warmup:], "Closing Balance": m_close[warmup:], "Orders Given": m_order[warmup:], 
        "Pipeline Inventory": m_pipe[warmup:]
    })
    
    vol_fr = total_sales / total_dem if total_dem > 0 else 1.0
    csl = 1 - (stockout_days / (total_days - warmup))
    return df_sec, df_main, vol_fr, csl, main_delay_days, pd.DataFrame(age_records), pd.DataFrame(daily_inventory_age), total_sales, total_dem, stockout_days

# --- Plotly Helper Functions ---
def render_interactive_chart(df, y_cols):
    fig = go.Figure()
    color_map = {'On-Hand Inventory': '#1f77b4', 'Pipeline Inventory': '#9467bd', 'Backlogged Orders': '#d62728', 'ROP Limit': '#ff7f0e', 'Sec On-Hand': '#1f77b4', 'Sec Pipeline': '#aec7e8', 'Main On-Hand': '#2ca02c', 'Main Pipeline': '#98df8a', 'Sec Backlogged': '#d62728'}
    for col in y_cols:
        if col not in df.columns: continue
        is_pipeline = 'Pipeline' in col; is_backlog = 'Backlog' in col
        fig.add_trace(go.Scatter(x=df['Day'], y=df[col], mode='lines', name=col, line=dict(color=color_map.get(col, '#333333'), width=2 if not is_pipeline else 3), line_shape='hv' if is_pipeline or is_backlog else 'linear', opacity=0.8 if is_pipeline else 1.0))
    fig.update_layout(xaxis_title="Day", yaxis_title="Units", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=80), plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    st.plotly_chart(fig, use_container_width=True)

def render_age_histogram(df_age, cols):
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, col in enumerate(cols):
        if col not in df_age.columns: continue
        fig.add_trace(go.Histogram(
            x=df_age[col], y=df_age['Qty'], histfunc='sum', name=col, 
            marker_color=colors[i % len(colors)], opacity=0.75, xbins=dict(size=1)
        ))
    fig.update_layout(barmode='overlay', xaxis_title="Days Spent", yaxis_title="Units (Qty)", margin=dict(l=0, r=0, t=30, b=80), plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    st.plotly_chart(fig, use_container_width=True)

def render_daily_age_profile(df_snapshots, selected_day):
    day_df = df_snapshots[df_snapshots['Day'] == selected_day].copy()
    if day_df.empty:
        st.info("No age data available for this day. (Units may belong to initial warmup stock with unknown origins).")
        return
    overall_df = day_df.copy()
    overall_df['Location'] = 'Overall System'
    combined_df = pd.concat([day_df, overall_df])
    grouped = combined_df.groupby(['Location', 'Age'])['Qty'].sum().reset_index()
    fig = px.bar(grouped, x="Location", y="Qty", color="Age", color_continuous_scale='RdYlBu_r', title=f"Inventory Age Profile on Day {selected_day} (System Age)", labels={"Qty": "Total Units", "Age": "System Age (Days since ordered)"})
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', barmode='stack', margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

def render_aging_buckets_chart(df_snapshots, location_filter):
    if df_snapshots.empty: return
    df = df_snapshots.copy()
    if location_filter != "Overall System":
        df = df[df['Location'].str.contains(location_filter)]
    if df.empty:
        st.info("No tracked inventory for this location during the period.")
        return

    bins = [-1, 30, 60, 90, float('inf')]
    labels = ['0-30', '31-60', '61-90', '90+']
    df['Age Bucket'] = pd.cut(df['Age'], bins=bins, labels=labels)
    grouped = df.groupby(['Day', 'Age Bucket'])['Qty'].sum().reset_index()
    
    color_map = {'0-30': '#82CAFA', '31-60': '#0066CC', '61-90': '#FF9999', '90+': '#FF0000'}
    fig = px.bar(grouped, x="Day", y="Qty", color="Age Bucket", color_discrete_map=color_map, category_orders={"Age Bucket": ['0-30', '31-60', '61-90', '90+']})
                 
    fig.update_layout(
        barmode='stack', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Day", yaxis_title="Units (Qty)",
        legend_title="Age (Days)", margin=dict(l=0, r=0, t=30, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), bargap=0
    )
    fig.update_traces(marker_line_width=0)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    st.plotly_chart(fig, use_container_width=True)

def render_working_capital_chart_single(df, cost, pipe_cost, include_pipeline):
    df_wc = pd.DataFrame({'Day': df['Day']})
    df_wc['On-Hand'] = df['Closing Balance'] * cost
    if include_pipeline:
        df_wc['Pipeline'] = df['Pipeline Inventory'] * pipe_cost
        df_wc['Overall System'] = df_wc['On-Hand'] + df_wc['Pipeline']
    else:
        df_wc['Overall System'] = df_wc['On-Hand']

    fig = px.line(df_wc, x='Day', y=[c for c in df_wc.columns if c != 'Day'], title="Dynamic Working Capital ($)", labels={'value': 'USD ($)', 'variable': 'Location'})
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=80), legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    st.plotly_chart(fig, use_container_width=True)

def render_working_capital_chart_multi(df_sec, df_main, s_cost, m_cost, s_pipe_cost, m_pipe_cost, include_pipeline):
    df_wc = pd.DataFrame({'Day': df_sec['Day']})
    df_wc['Secondary On-Hand'] = df_sec['Closing Balance'] * s_cost
    df_wc['Main On-Hand'] = df_main['Closing Balance'] * m_cost
    if include_pipeline:
        df_wc['Secondary Pipeline'] = df_sec['Pipeline Inventory'] * s_pipe_cost
        df_wc['Main Pipeline'] = df_main['Pipeline Inventory'] * m_pipe_cost
        df_wc['Overall System'] = df_wc['Secondary On-Hand'] + df_wc['Main On-Hand'] + df_wc['Secondary Pipeline'] + df_wc['Main Pipeline']
    else:
        df_wc['Overall System'] = df_wc['Secondary On-Hand'] + df_wc['Main On-Hand']

    fig = px.line(df_wc, x='Day', y=[c for c in df_wc.columns if c != 'Day'], title="Dynamic Working Capital ($)", labels={'value': 'USD ($)', 'variable': 'Location'})
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=80), legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)', zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    st.plotly_chart(fig, use_container_width=True)

def render_avg_age_flux_chart(df_snapshots, location_filter, df_flows):
    if df_snapshots.empty or df_flows.empty: return
    df = df_snapshots.copy()
    
    if location_filter != "Overall System":
        df = df[df['Location'].str.contains(location_filter)]
        
    if df.empty:
        st.info("No tracked inventory for this location during the period.")
        return

    # Calculate weighted average age
    df['Age_x_Qty'] = df['Age'] * df['Qty']
    daily_avg = df.groupby('Day').agg({'Age_x_Qty': 'sum', 'Qty': 'sum'}).reset_index()
    daily_avg['Avg Age'] = daily_avg['Age_x_Qty'] / daily_avg['Qty']
    
    # Merge with flow data
    merged = pd.merge(daily_avg, df_flows, on='Day', how='left').fillna(0)

    # Dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Unit Bars (Receipts and Dispatches)
    fig.add_trace(go.Bar(x=merged['Day'], y=merged['Order Received'], name="Receipts (In)", marker_color='#2ca02c', opacity=0.6), secondary_y=False)
    fig.add_trace(go.Bar(x=merged['Day'], y=merged['Sales'], name="Dispatches (Out)", marker_color='#d62728', opacity=0.6), secondary_y=False)
    
    # Age Line
    fig.add_trace(go.Scatter(x=merged['Day'], y=merged['Avg Age'], name="Average Age (Days)", mode='lines', line=dict(color='#1f77b4', width=3)), secondary_y=True)
    
    fig.update_layout(
        title=f"Inventory Age Flux: {location_filter}",
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='group',
        margin=dict(l=0, r=0, t=40, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    
    fig.update_xaxes(showgrid=False, title_text="Day", zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    fig.update_yaxes(title_text="Units (Qty)", showgrid=False, secondary_y=False, zeroline=True, zerolinecolor='rgba(200,200,200,0.5)')
    fig.update_yaxes(title_text="Average Age (Days)", showgrid=True, gridcolor='rgba(200,200,200,0.2)', secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# --- App Configuration & State ---
st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")
st.title("📦 Supply Chain Scenario Architect")

# ==========================================
# GLOBAL SETTINGS & DEMAND GENERATION
# ==========================================
st.markdown("### ⚙️ Global Environmental Settings")
col_g1, col_g2, col_g3, col_g4 = st.columns(4)
warmup_days = col_g1.number_input("Warm-up Period", min_value=0, value=150, step=30)
sim_days = col_g2.number_input("Display Period", min_value=10, value=300, step=30)
global_dem = col_g3.number_input("Global Target Demand/Day", min_value=0.0, value=100.0, step=10.0)
global_std = col_g4.number_input("Global Demand Std Dev", min_value=0.0, value=20.0, step=5.0)

total_sim_days = int(warmup_days + sim_days)

if 'demand_array' not in st.session_state or 'sim_params' not in st.session_state or st.session_state.sim_params != (total_sim_days, global_dem, global_std):
    st.session_state.demand_array = np.maximum(0, np.random.normal(global_dem, global_std, total_sim_days)).astype(int)
    st.session_state.sim_params = (total_sim_days, global_dem, global_std)

if st.button("🔄 Generate New Demand Profile"):
    st.session_state.demand_array = np.maximum(0, np.random.normal(global_dem, global_std, total_sim_days)).astype(int)

st.caption("Note: All scenarios run against the exact same pre-generated daily demand to ensure fair comparisons.")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🏢 S1: Single Central", "🏬 S2: Two-Stage (Local ROP)", "🌍 S3: Multi-Echelon", "🔄 S4: Policy Diagnostics (s,Q vs R,S)"])

# ==========================================
# TAB 1: SINGLE WAREHOUSE
# ==========================================
with tab1:
    st.markdown("#### Central Warehouse Variables")
    col1c, col1d = st.columns(2)
    s1_lead_time = col1c.number_input("Lead Time (days)", min_value=0.0, value=7.0, step=1.0, key="s1_lt")
    s1_service_level = col1d.slider("Target Fill Rate", 0.50, 0.999, 0.95, key="s1_sl")
    
    st.markdown("#### Costs & Reorder Logic")
    col1e, col1f, col1g, col1h = st.columns(4)
    s1_cost = col1e.number_input("Warehouse Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s1_cost")
    s1_pipeline_cost = col1f.number_input("Pipeline Unit Cost ($)", min_value=0.01, value=40.0, step=5.0, key="s1_pipe_cost")
    s1_q = col1g.number_input("Order Qty (Q)", min_value=1, value=500, step=50, key="s1_q")
    rec_s1_rop, _ = get_recommendations(global_dem, global_std, s1_lead_time, s1_service_level)
    s1_actual_rop = col1h.number_input("Actual ROP", min_value=0, value=int(rec_s1_rop), step=10, key="s1_act")
    col1h.caption(f"💡 Suggested: **{rec_s1_rop:,.0f}**")
    
    st.markdown("#### Fulfillment Rules")
    s1_col1, s1_col2 = st.columns(2)
    s1_allow_partial = s1_col1.checkbox("Allow Partial", value=True, key="s1_partial")
    s1_track_backlogs = s1_col2.checkbox("Track Backlogs", value=True, key="s1_backlog")
    
    st.markdown("---")
    df_s1, vol_fr_1, csl_1, age_s1, d_age_s1, t_sales_1, t_dem_1, stockout_d_1 = simulate_single_stage_detailed(st.session_state.demand_array, s1_actual_rop, s1_q, s1_lead_time, warmup_days, s1_allow_partial, s1_track_backlogs)

    s1_avg_oh = df_s1['Closing Balance'].mean(); s1_avg_pipe = df_s1['Pipeline Inventory'].mean()
    s1_sim_wc = (s1_avg_oh * s1_cost) + (s1_avg_pipe * s1_pipeline_cost)
    s1_daily_wc = (df_s1['Closing Balance'] * s1_cost) + (df_s1['Pipeline Inventory'] * s1_pipeline_cost)
    s1_peak_wc = s1_daily_wc.max()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Simulated Avg WC", f"${s1_sim_wc:,.2f}"); m2.metric("Peak Working Capital", f"${s1_peak_wc:,.2f}")
    m3.metric("Volume Fill Rate", f"{vol_fr_1*100:.1f}%", f"{t_sales_1:,.0f} / {t_dem_1:,.0f} Phys.", delta_color="off")
    m4.metric("Service Level", f"{csl_1*100:.1f}%", f"{sim_days - stockout_d_1:,.0f} / {sim_days:,.0f} Days", delta_color="off")
    m5.metric("Total Sales", f"{t_sales_1:,.0f}")
    
    plot_df1 = pd.DataFrame({'Day': df_s1['Day'], 'On-Hand Inventory': df_s1['Closing Balance'], 'Pipeline Inventory': df_s1['Pipeline Inventory'], 'Backlogged Orders': df_s1['Backlogs'], 'ROP Limit': s1_actual_rop})
    render_interactive_chart(plot_df1, ['On-Hand Inventory', 'Pipeline Inventory', 'Backlogged Orders', 'ROP Limit'])

    st.markdown("### 📈 Working Capital Analysis")
    show_pipe_1 = st.checkbox("Include Pipeline Inventory in WC Calculation", value=True, key="wc_t1")
    render_working_capital_chart_single(df_s1, s1_cost, s1_pipeline_cost, show_pipe_1)
    
    st.markdown("### 📊 Daily Age Profile & Aging Buckets")
    tab1_sub1, tab1_sub2, tab1_sub3 = st.tabs(["Longitudinal Aging Buckets", "Single Day Thermal Profile", "Average Age Flux"])
    with tab1_sub1:
        bucket_view_1 = st.selectbox("Select View", ["Overall System", "Central On-Hand", "Supplier Pipeline"], key="b_s1")
        render_aging_buckets_chart(d_age_s1, bucket_view_1)
    with tab1_sub2:
        selected_day_1 = st.slider("Select Day to View Age Distribution", min_value=1, max_value=int(sim_days), value=int(sim_days), key="day_s1")
        render_daily_age_profile(d_age_s1, selected_day_1)
    with tab1_sub3:
        flux_view_1 = st.selectbox("Select View for Age Flux", ["Overall System", "Central On-Hand"], key="flux_s1")
        df_flows_1 = df_s1[['Day', 'Order Received', 'Sales']]
        render_avg_age_flux_chart(d_age_s1, flux_view_1, df_flows_1)

    st.markdown("### ⏳ Age of Inventory at Sale (FIFO Analytics)")
    if not age_s1.empty:
        age_s1['Total Time'] = age_s1['Pipeline Time'] + age_s1['Warehouse Time']
        a1, a2, a3 = st.columns(3)
        a1.metric("Avg Pipeline Time", f"{(age_s1['Pipeline Time'] * age_s1['Qty']).sum() / age_s1['Qty'].sum():.1f} Days")
        a2.metric("Avg Warehouse Time", f"{(age_s1['Warehouse Time'] * age_s1['Qty']).sum() / age_s1['Qty'].sum():.1f} Days")
        a3.metric("Avg Total Age at Sale", f"{(age_s1['Total Time'] * age_s1['Qty']).sum() / age_s1['Qty'].sum():.1f} Days")
        render_age_histogram(age_s1, ['Pipeline Time', 'Warehouse Time', 'Total Time'])

# ==========================================
# TAB 2: TWO-STAGE (LOCAL ROP)
# ==========================================
with tab2:
    st.markdown("#### Secondary (Front-line)")
    c2c, c2d = st.columns(2)
    s2_sec_lt = c2c.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s2_sec_lt")
    s2_sec_sl = c2d.slider("Sec Target Fill Rate", 0.50, 0.999, 0.95, key="s2_sec_sl")
    
    c2e, c2f, c2g, c2h, c2i = st.columns(5)
    s2_sec_cost = c2e.number_input("Unit Cost ($)", min_value=0.01, value=60.0, step=5.0, key="s2_sec_cost")
    s2_sec_q = c2f.number_input("Order Qty", min_value=1, value=300, step=50, key="s2_sec_q")
    rec_sec_rop, _ = get_recommendations(global_dem, global_std, s2_sec_lt, s2_sec_sl)
    s2_sec_actual_rop = c2g.number_input("Actual ROP", min_value=0, value=int(rec_sec_rop), step=10, key="s2_sec_act")
    c2g.caption(f"💡 Suggested: **{rec_sec_rop:,.0f}**")
    s2_sec_allow_partial = c2h.checkbox("Allow Partial", value=True, key="s2_sec_partial")
    s2_sec_track_backlogs = c2i.checkbox("Track Backlogs", value=True, key="s2_sec_backlog")
    
    st.markdown("#### Main (Hub)")
    c3a, c3b, c3c, c3d = st.columns(4)
    s2_main_demand = c3a.number_input("Agg Demand", min_value=0.0, value=global_dem, step=10.0, key="s2_main_d")
    s2_main_std = c3b.number_input("Agg Std Dev", min_value=0.0, value=global_std, step=5.0, key="s2_main_std")
    s2_main_lt = c3c.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s2_main_lt")
    s2_main_sl = c3d.slider("Main Target Fill Rate", 0.50, 0.999, 0.98, key="s2_main_sl")
    
    c3e, c3f, c3g, c3h, c3i = st.columns(5)
    s2_main_cost = c3e.number_input("Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s2_main_cost")
    s2_main_q = c3f.number_input("Order Qty", min_value=1, value=800, step=50, key="s2_main_q")
    rec_main_rop, _ = get_recommendations(s2_main_demand, s2_main_std, s2_main_lt, s2_main_sl)
    s2_main_actual_rop = c3g.number_input("Actual ROP", min_value=0, value=int(rec_main_rop), step=10, key="s2_main_act")
    c3g.caption(f"💡 Suggested: **{rec_main_rop:,.0f}**")
    s2_main_allow_partial = c3h.checkbox("Allow Partial", value=True, key="s2_main_partial")
    s2_main_track_backlogs = c3i.checkbox("Track Backlogs", value=True, key="s2_main_backlog")

    st.markdown("---")
    df_sec_s2, df_main_s2, vol_fr_2, csl_2, delay_2, age_s2, d_age_s2, t_sales_2, t_dem_2, stockout_d_2 = simulate_two_stage_detailed(st.session_state.demand_array, s2_sec_actual_rop, s2_sec_q, s2_sec_lt, s2_main_actual_rop, s2_main_q, s2_main_lt, warmup_days, s2_sec_allow_partial, s2_sec_track_backlogs, s2_main_allow_partial, s2_main_track_backlogs, "installation")
    
    avg_sec_oh_2 = df_sec_s2['Closing Balance'].mean(); avg_sec_pipe_2 = df_sec_s2['Pipeline Inventory'].mean()
    avg_main_oh_2 = df_main_s2['Closing Balance'].mean(); avg_main_pipe_2 = df_main_s2['Pipeline Inventory'].mean()
    total_sys_val_2 = (avg_sec_oh_2 * s2_sec_cost) + (avg_sec_pipe_2 * s2_sec_cost) + (avg_main_oh_2 * s2_main_cost) + (avg_main_pipe_2 * s2_main_cost)
    s2_daily_sys_val = (df_sec_s2['Closing Balance'] + df_sec_s2['Pipeline Inventory']) * s2_sec_cost + (df_main_s2['Closing Balance'] + df_main_s2['Pipeline Inventory']) * s2_main_cost
    peak_sys_val_2 = s2_daily_sys_val.max()

    sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
    sm1.metric("Simulated Avg WC", f"${total_sys_val_2:,.2f}"); sm2.metric("Peak Working Capital", f"${peak_sys_val_2:,.2f}")
    sm3.metric("Volume Fill Rate", f"{vol_fr_2*100:.1f}%", f"{t_sales_2:,.0f} / {t_dem_2:,.0f} Phys.", delta_color="off")
    sm4.metric("Service Level", f"{csl_2*100:.1f}%", f"{sim_days - stockout_d_2:,.0f} / {sim_days:,.0f} Days", delta_color="off")
    sm5.metric("Main Delays", f"{delay_2} Days"); sm6.metric("Total Sales", f"{t_sales_2:,.0f}")

    plot_df2 = pd.DataFrame({'Day': df_sec_s2['Day'], 'Sec On-Hand': df_sec_s2['Closing Balance'], 'Sec Pipeline': df_sec_s2['Pipeline Inventory'], 'Sec Backlogged': df_sec_s2['Backlogs'], 'Main On-Hand': df_main_s2['Closing Balance'], 'Main Pipeline': df_main_s2['Pipeline Inventory']})
    render_interactive_chart(plot_df2, ['Sec On-Hand', 'Sec Pipeline', 'Sec Backlogged', 'Main On-Hand', 'Main Pipeline'])

    st.markdown("### 📈 Working Capital Analysis")
    show_pipe_2 = st.checkbox("Include Pipeline Inventory in WC Calculation", value=True, key="wc_t2")
    render_working_capital_chart_multi(df_sec_s2, df_main_s2, s2_sec_cost, s2_main_cost, s2_sec_cost, s2_main_cost, show_pipe_2)
    
    st.markdown("### 📊 Daily Age Profile & Aging Buckets")
    tab2_sub1, tab2_sub2, tab2_sub3 = st.tabs(["Longitudinal Aging Buckets", "Single Day Thermal Profile", "Average Age Flux"])
    with tab2_sub1:
        bucket_view_2 = st.selectbox("Select View", ["Overall System", "Main On-Hand", "Sec On-Hand"], key="b_s2")
        render_aging_buckets_chart(d_age_s2, bucket_view_2)
    with tab2_sub2:
        selected_day_2 = st.slider("Select Day to View Age Distribution", min_value=1, max_value=int(sim_days), value=int(sim_days), key="day_s2")
        render_daily_age_profile(d_age_s2, selected_day_2)
    with tab2_sub3:
        flux_view_2 = st.selectbox("Select View for Age Flux", ["Overall System", "Main On-Hand", "Sec On-Hand"], key="flux_s2")
        if flux_view_2 == "Overall System":
            df_flows_2 = pd.DataFrame({'Day': df_sec_s2['Day'], 'Order Received': df_main_s2['Order Received'], 'Sales': df_sec_s2['Sales']})
        elif flux_view_2 == "Main On-Hand":
            df_flows_2 = df_main_s2[['Day', 'Order Received', 'Sales']]
        else:
            df_flows_2 = df_sec_s2[['Day', 'Order Received', 'Sales']]
        render_avg_age_flux_chart(d_age_s2, flux_view_2, df_flows_2)

    st.markdown("### ⏳ Age of Inventory at Sale (FIFO Analytics)")
    if not age_s2.empty:
        age_s2['Total Time'] = age_s2['Main Pipeline Time'] + age_s2['Main Warehouse Time'] + age_s2['Sec Pipeline Time'] + age_s2['Sec Warehouse Time']
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Avg Main Pipe", f"{(age_s2['Main Pipeline Time'] * age_s2['Qty']).sum() / age_s2['Qty'].sum():.1f} d")
        a2.metric("Avg Main WH", f"{(age_s2['Main Warehouse Time'] * age_s2['Qty']).sum() / age_s2['Qty'].sum():.1f} d")
        a3.metric("Avg Sec Pipe", f"{(age_s2['Sec Pipeline Time'] * age_s2['Qty']).sum() / age_s2['Qty'].sum():.1f} d")
        a4.metric("Avg Sec WH", f"{(age_s2['Sec Warehouse Time'] * age_s2['Qty']).sum() / age_s2['Qty'].sum():.1f} d")
        a5.metric("Total Age at Sale", f"{(age_s2['Total Time'] * age_s2['Qty']).sum() / age_s2['Qty'].sum():.1f} d")
        render_age_histogram(age_s2, ['Main Warehouse Time', 'Sec Warehouse Time', 'Total Time'])

# ==========================================
# TAB 3: ECHELON SYSTEM
# ==========================================
with tab3:
    st.markdown("#### Secondary (Front-line)")
    c4c, c4d = st.columns(2)
    s3_sec_lt = c4c.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s3_sec_lt")
    s3_sec_sl = c4d.slider("Sec Target Fill Rate", 0.50, 0.999, 0.95, key="s3_sec_sl")
    
    c4e, c4f, c4g, c4h, c4i = st.columns(5)
    s3_sec_cost = c4e.number_input("Unit Cost ($)", min_value=0.01, value=60.0, step=5.0, key="s3_sec_cost")
    s3_sec_q = c4f.number_input("Sec Order Qty", min_value=1, value=300, step=50, key="s3_sec_q")
    rec_s3_sec_rop, _ = get_recommendations(global_dem, global_std, s3_sec_lt, s3_sec_sl)
    s3_sec_actual_rop = c4g.number_input("Sec Actual ROP", min_value=0, value=int(rec_s3_sec_rop), step=10, key="s3_sec_act")
    c4g.caption(f"💡 Suggested: **{rec_s3_sec_rop:,.0f}**")
    s3_sec_allow_partial = c4h.checkbox("Allow Partial", value=True, key="s3_sec_partial")
    s3_sec_track_backlogs = c4i.checkbox("Track Backlogs", value=True, key="s3_sec_backlog")
    
    st.markdown("#### Main (Echelon Evaluator)")
    c5a, c5b, c5c, c5d = st.columns(4)
    s3_main_demand = c5a.number_input("Agg Demand", min_value=0.0, value=global_dem, step=10.0, key="s3_main_d")
    s3_main_std = c5b.number_input("Agg Std Dev", min_value=0.0, value=global_std, step=5.0, key="s3_main_std")
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
    df_sec_s3, df_main_s3, vol_fr_3, csl_3, delay_3, age_s3, d_age_s3, t_sales_3, t_dem_3, stockout_d_3 = simulate_two_stage_detailed(st.session_state.demand_array, s3_sec_actual_rop, s3_sec_q, s3_sec_lt, s3_echelon_actual_rop, s3_main_q, s3_main_lt, warmup_days, s3_sec_allow_partial, s3_sec_track_backlogs, s3_main_allow_partial, s3_main_track_backlogs, "echelon")
    
    avg_sec_oh_3 = df_sec_s3['Closing Balance'].mean(); avg_sec_pipe_3 = df_sec_s3['Pipeline Inventory'].mean()
    avg_main_oh_3 = df_main_s3['Closing Balance'].mean(); avg_main_pipe_3 = df_main_s3['Pipeline Inventory'].mean()
    total_sys_val_3 = (avg_sec_oh_3 * s3_sec_cost) + (avg_sec_pipe_3 * s3_sec_cost) + (avg_main_oh_3 * s3_main_cost) + (avg_main_pipe_3 * s3_main_cost)
    s3_daily_sys_val = (df_sec_s3['Closing Balance'] + df_sec_s3['Pipeline Inventory']) * s3_sec_cost + (df_main_s3['Closing Balance'] + df_main_s3['Pipeline Inventory']) * s3_main_cost
    peak_sys_val_3 = s3_daily_sys_val.max()

    tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
    tm1.metric("Simulated Avg WC", f"${total_sys_val_3:,.2f}"); tm2.metric("Peak Working Capital", f"${peak_sys_val_3:,.2f}")
    tm3.metric("Volume Fill Rate", f"{vol_fr_3*100:.1f}%", f"{t_sales_3:,.0f} / {t_dem_3:,.0f} Phys.", delta_color="off")
    tm4.metric("Service Level", f"{csl_3*100:.1f}%", f"{sim_days - stockout_d_3:,.0f} / {sim_days:,.0f} Days", delta_color="off")
    tm5.metric("Main Delays", f"{delay_3} Days"); tm6.metric("Total Sales", f"{t_sales_3:,.0f}")

    plot_df3 = pd.DataFrame({'Day': df_sec_s3['Day'], 'Sec On-Hand': df_sec_s3['Closing Balance'], 'Sec Pipeline': df_sec_s3['Pipeline Inventory'], 'Sec Backlogged': df_sec_s3['Backlogs'], 'Main On-Hand': df_main_s3['Closing Balance'], 'Main Pipeline': df_main_s3['Pipeline Inventory']})
    render_interactive_chart(plot_df3, ['Sec On-Hand', 'Sec Pipeline', 'Sec Backlogged', 'Main On-Hand', 'Main Pipeline'])

    st.markdown("### 📈 Working Capital Analysis")
    show_pipe_3 = st.checkbox("Include Pipeline Inventory in WC Calculation", value=True, key="wc_t3")
    render_working_capital_chart_multi(df_sec_s3, df_main_s3, s3_sec_cost, s3_main_cost, s3_sec_cost, s3_main_cost, show_pipe_3)
    
    st.markdown("### 📊 Daily Age Profile & Aging Buckets")
    tab3_sub1, tab3_sub2, tab3_sub3 = st.tabs(["Longitudinal Aging Buckets", "Single Day Thermal Profile", "Average Age Flux"])
    with tab3_sub1:
        bucket_view_3 = st.selectbox("Select View", ["Overall System", "Main On-Hand", "Sec On-Hand"], key="b_s3")
        render_aging_buckets_chart(d_age_s3, bucket_view_3)
    with tab3_sub2:
        selected_day_3 = st.slider("Select Day to View Age Distribution", min_value=1, max_value=int(sim_days), value=int(sim_days), key="day_s3")
        render_daily_age_profile(d_age_s3, selected_day_3)
    with tab3_sub3:
        flux_view_3 = st.selectbox("Select View for Age Flux", ["Overall System", "Main On-Hand", "Sec On-Hand"], key="flux_s3")
        if flux_view_3 == "Overall System":
            df_flows_3 = pd.DataFrame({'Day': df_sec_s3['Day'], 'Order Received': df_main_s3['Order Received'], 'Sales': df_sec_s3['Sales']})
        elif flux_view_3 == "Main On-Hand":
            df_flows_3 = df_main_s3[['Day', 'Order Received', 'Sales']]
        else:
            df_flows_3 = df_sec_s3[['Day', 'Order Received', 'Sales']]
        render_avg_age_flux_chart(d_age_s3, flux_view_3, df_flows_3)

    st.markdown("### ⏳ Age of Inventory at Sale (FIFO Analytics)")
    if not age_s3.empty:
        age_s3['Total Time'] = age_s3['Main Pipeline Time'] + age_s3['Main Warehouse Time'] + age_s3['Sec Pipeline Time'] + age_s3['Sec Warehouse Time']
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Avg Main Pipe", f"{(age_s3['Main Pipeline Time'] * age_s3['Qty']).sum() / age_s3['Qty'].sum():.1f} d")
        a2.metric("Avg Main WH", f"{(age_s3['Main Warehouse Time'] * age_s3['Qty']).sum() / age_s3['Qty'].sum():.1f} d")
        a3.metric("Avg Sec Pipe", f"{(age_s3['Sec Pipeline Time'] * age_s3['Qty']).sum() / age_s3['Qty'].sum():.1f} d")
        a4.metric("Avg Sec WH", f"{(age_s3['Sec Warehouse Time'] * age_s3['Qty']).sum() / age_s3['Qty'].sum():.1f} d")
        a5.metric("Total Age at Sale", f"{(age_s3['Total Time'] * age_s3['Qty']).sum() / age_s3['Qty'].sum():.1f} d")
        render_age_histogram(age_s3, ['Main Warehouse Time', 'Sec Warehouse Time', 'Total Time'])

# ==========================================
# TAB 4: POLICY DIAGNOSTICS (CONTINUOUS VS PERIODIC)
# ==========================================
with tab4:
    st.markdown("#### 🏪 Secondary (Front-line) Policy")
    col4_1, col4_2 = st.columns([1, 3])
    s4_sec_policy = col4_1.radio("Secondary Policy", ["Continuous (s, Q)", "Periodic (R, S)"], key="s4_sec_pol")
    
    with col4_2:
        c6a, c6b, c6c = st.columns(3)
        s4_sec_lt = c6a.number_input("Transit LT", min_value=0.0, value=3.0, step=1.0, key="s4_sec_lt")
        s4_sec_sl = c6b.slider("Sec Target Fill Rate", 0.50, 0.999, 0.95, key="s4_sec_sl")
        s4_sec_cost = c6c.number_input("Unit Cost ($)", min_value=0.01, value=60.0, step=5.0, key="s4_sec_cost")
        
        c6d, c6e, c6f = st.columns(3)
        if s4_sec_policy == "Continuous (s, Q)":
            s4_sec_q = c6d.number_input("Order Qty (Q)", min_value=1, value=300, step=50, key="s4_sec_q")
            rec_s4_sec_rop, _ = get_recommendations(global_dem, global_std, s4_sec_lt, s4_sec_sl)
            s4_sec_actual_rop = c6e.number_input("Actual ROP (s)", min_value=0, value=int(rec_s4_sec_rop), step=10, key="s4_sec_rop")
            c6e.caption(f"💡 Suggested ROP: **{rec_s4_sec_rop:,.0f}**")
            s4_sec_r, s4_sec_s = 1, 0
        else:
            s4_sec_r = c6d.number_input("Review Period (R)", min_value=1, value=7, step=1, key="s4_sec_r")
            rec_s4_sec_s, _ = get_periodic_recommendations(global_dem, global_std, s4_sec_lt, s4_sec_r, s4_sec_sl)
            s4_sec_s = c6e.number_input("Target Level (S)", min_value=0, value=int(rec_s4_sec_s), step=10, key="s4_sec_s_tgt")
            c6e.caption(f"💡 Suggested Target: **{rec_s4_sec_s:,.0f}**")
            s4_sec_q, s4_sec_actual_rop = 0, 0
            
        s4_sec_allow_partial = c6f.checkbox("Allow Partial", value=True, key="s4_sec_partial")
        s4_sec_track_backlogs = c6f.checkbox("Track Backlogs", value=True, key="s4_sec_backlog")

    st.markdown("#### 🏭 Main (Echelon Evaluator) Policy")
    col4_3, col4_4 = st.columns([1, 3])
    s4_main_policy = col4_3.radio("Main Policy", ["Continuous (s, Q)", "Periodic (R, S)"], key="s4_main_pol")
    
    with col4_4:
        c7a, c7b, c7c = st.columns(3)
        s4_main_demand = c7a.number_input("Agg Demand", min_value=0.0, value=global_dem, step=10.0, key="s4_main_d")
        s4_main_lt = c7b.number_input("Supplier LT", min_value=0.0, value=10.0, step=1.0, key="s4_main_lt")
        s4_main_sl = c7c.slider("Main Target Fill Rate", 0.50, 0.999, 0.98, key="s4_main_sl")
        
        c7d, c7e, c7f = st.columns(3)
        s4_main_cost = c7d.number_input("Unit Cost ($)", min_value=0.01, value=50.0, step=5.0, key="s4_main_cost")
        
        if s4_main_policy == "Continuous (s, Q)":
            s4_main_q = c7e.number_input("Main Order Qty (Q)", min_value=1, value=800, step=50, key="s4_main_q")
            _, main_base_ss = get_recommendations(s4_main_demand, global_std, s4_main_lt, s4_main_sl)
            rec_echelon_rop = rec_s4_sec_rop + (s4_main_demand * s4_main_lt) + main_base_ss if s4_sec_policy == "Continuous (s, Q)" else rec_s4_sec_s + (s4_main_demand * s4_main_lt) + main_base_ss
            s4_echelon_actual_rop = c7f.number_input("Echelon Actual ROP", min_value=0, value=int(rec_echelon_rop), step=10, key="s4_ech_act")
            c7f.caption(f"💡 Suggested: **{rec_echelon_rop:,.0f}**")
            s4_main_r, s4_main_s = 1, 0
        else:
            s4_main_r = c7e.number_input("Main Review Period (R)", min_value=1, value=14, step=1, key="s4_main_r")
            rec_main_s, _ = get_periodic_recommendations(s4_main_demand, global_std, s4_main_lt, s4_main_r, s4_main_sl)
            sec_trigger = rec_s4_sec_rop if s4_sec_policy == "Continuous (s, Q)" else rec_s4_sec_s
            rec_echelon_s = sec_trigger + rec_main_s
            s4_main_s = c7f.number_input("Echelon Target Level (S)", min_value=0, value=int(rec_echelon_s), step=10, key="s4_main_s_tgt")
            c7f.caption(f"💡 Suggested: **{rec_echelon_s:,.0f}**")
            s4_main_q, s4_echelon_actual_rop = 0, 0
            
        c7g, c7h = st.columns(2)
        s4_main_allow_partial = c7g.checkbox("Allow Partial", value=True, key="s4_main_partial")
        s4_main_track_backlogs = c7h.checkbox("Track Backlogs", value=True, key="s4_main_backlog")

    st.markdown("---")
    
    sec_pol_str = "continuous" if s4_sec_policy == "Continuous (s, Q)" else "periodic"
    main_pol_str = "continuous" if s4_main_policy == "Continuous (s, Q)" else "periodic"
    
    df_sec_s4, df_main_s4, vol_fr_4, csl_4, delay_4, age_s4, d_age_s4, t_sales_4, t_dem_4, stockout_d_4 = simulate_two_stage_detailed(
        st.session_state.demand_array, s4_sec_actual_rop, s4_sec_q, s4_sec_lt, s4_echelon_actual_rop, s4_main_q, s4_main_lt, 
        warmup_days, s4_sec_allow_partial, s4_sec_track_backlogs, s4_main_allow_partial, s4_main_track_backlogs, 
        "echelon", sec_pol_str, s4_sec_r, s4_sec_s, main_pol_str, s4_main_r, s4_main_s
    )
    
    avg_sec_oh_4 = df_sec_s4['Closing Balance'].mean(); avg_sec_pipe_4 = df_sec_s4['Pipeline Inventory'].mean()
    avg_main_oh_4 = df_main_s4['Closing Balance'].mean(); avg_main_pipe_4 = df_main_s4['Pipeline Inventory'].mean()
    total_sys_val_4 = (avg_sec_oh_4 * s4_sec_cost) + (avg_sec_pipe_4 * s4_sec_cost) + (avg_main_oh_4 * s4_main_cost) + (avg_main_pipe_4 * s4_main_cost)
    s4_daily_sys_val = (df_sec_s4['Closing Balance'] + df_sec_s4['Pipeline Inventory']) * s4_sec_cost + (df_main_s4['Closing Balance'] + df_main_s4['Pipeline Inventory']) * s4_main_cost
    peak_sys_val_4 = s4_daily_sys_val.max()

    tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
    tm1.metric("Simulated Avg WC", f"${total_sys_val_4:,.2f}"); tm2.metric("Peak Working Capital", f"${peak_sys_val_4:,.2f}")
    tm3.metric("Volume Fill Rate", f"{vol_fr_4*100:.1f}%", f"{t_sales_4:,.0f} / {t_dem_4:,.0f} Phys.", delta_color="off")
    tm4.metric("Service Level", f"{csl_4*100:.1f}%", f"{sim_days - stockout_d_4:,.0f} / {sim_days:,.0f} Days", delta_color="off")
    tm5.metric("Main Delays", f"{delay_4} Days"); tm6.metric("Total Sales", f"{t_sales_4:,.0f}")

    plot_df4 = pd.DataFrame({'Day': df_sec_s4['Day'], 'Sec On-Hand': df_sec_s4['Closing Balance'], 'Sec Pipeline': df_sec_s4['Pipeline Inventory'], 'Sec Backlogged': df_sec_s4['Backlogs'], 'Main On-Hand': df_main_s4['Closing Balance'], 'Main Pipeline': df_main_s4['Pipeline Inventory']})
    fig4 = go.Figure()
    color_map = {'Sec On-Hand': '#1f77b4', 'Sec Pipeline': '#aec7e8', 'Main On-Hand': '#0066CC', 'Main Pipeline': '#82CAFA', 'Sec Backlogged': '#d62728'}
    for col in ['Sec On-Hand', 'Sec Pipeline', 'Sec Backlogged', 'Main On-Hand', 'Main Pipeline']:
        is_pipeline = 'Pipeline' in col; is_backlog = 'Backlog' in col
        fig4.add_trace(go.Scatter(x=plot_df4['Day'], y=plot_df4[col], mode='lines', name=col, line=dict(color=color_map.get(col, '#333333'), width=2 if not is_pipeline else 3), line_shape='hv' if is_pipeline or is_backlog else 'linear', opacity=0.8 if is_pipeline else 1.0))
    fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Day", yaxis_title="Units", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=80), legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📈 Working Capital Analysis")
    show_pipe_4 = st.checkbox("Include Pipeline Inventory in WC Calculation", value=True, key="wc_t4")
    render_working_capital_chart_multi(df_sec_s4, df_main_s4, s4_sec_cost, s4_main_cost, s4_sec_cost, s4_main_cost, show_pipe_4)
    
    st.markdown("### 📊 Daily Age Profile & Aging Buckets")
    tab4_sub1, tab4_sub2, tab4_sub3 = st.tabs(["Longitudinal Aging Buckets", "Single Day Thermal Profile", "Average Age Flux"])
    with tab4_sub1:
        bucket_view_4 = st.selectbox("Select View", ["Overall System", "Main On-Hand", "Sec On-Hand"], key="b_s4")
        render_aging_buckets_chart(d_age_s4, bucket_view_4)
    with tab4_sub2:
        selected_day_4 = st.slider("Select Day to View Age Distribution", min_value=1, max_value=int(sim_days), value=int(sim_days), key="day_s4")
        render_daily_age_profile(d_age_s4, selected_day_4)
    with tab4_sub3:
        flux_view_4 = st.selectbox("Select View for Age Flux", ["Overall System", "Main On-Hand", "Sec On-Hand"], key="flux_s4")
        if flux_view_4 == "Overall System":
            df_flows_4 = pd.DataFrame({'Day': df_sec_s4['Day'], 'Order Received': df_main_s4['Order Received'], 'Sales': df_sec_s4['Sales']})
        elif flux_view_4 == "Main On-Hand":
            df_flows_4 = df_main_s4[['Day', 'Order Received', 'Sales']]
        else:
            df_flows_4 = df_sec_s4[['Day', 'Order Received', 'Sales']]
        render_avg_age_flux_chart(d_age_s4, flux_view_4, df_flows_4)

    st.markdown("### ⏳ Age of Inventory at Sale (FIFO Analytics)")
    if not age_s4.empty:
        age_s4['Total Time'] = age_s4['Main Pipeline Time'] + age_s4['Main Warehouse Time'] + age_s4['Sec Pipeline Time'] + age_s4['Sec Warehouse Time']
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Avg Main Pipe", f"{(age_s4['Main Pipeline Time'] * age_s4['Qty']).sum() / age_s4['Qty'].sum():.1f} d")
        a2.metric("Avg Main WH", f"{(age_s4['Main Warehouse Time'] * age_s4['Qty']).sum() / age_s4['Qty'].sum():.1f} d")
        a3.metric("Avg Sec Pipe", f"{(age_s4['Sec Pipeline Time'] * age_s4['Qty']).sum() / age_s4['Qty'].sum():.1f} d")
        a4.metric("Avg Sec WH", f"{(age_s4['Sec Warehouse Time'] * age_s4['Qty']).sum() / age_s4['Qty'].sum():.1f} d")
        a5.metric("Total Age at Sale", f"{(age_s4['Total Time'] * age_s4['Qty']).sum() / age_s4['Qty'].sum():.1f} d")
        render_age_histogram(age_s4, ['Main Warehouse Time', 'Sec Warehouse Time', 'Total Time'])

# ==========================================
# MASTER COMPARISON TABLE
# ==========================================
st.markdown("---")
st.header("📋 Master Comparison Summary")
comparison_data = {
    "Metric": ["Target Fill Rate", "Order Qty (Q)", "Suggested ROP", "Actual Set ROP", "Avg Working Capital", "Peak Working Capital", "Total Sales (Units)", "Stockout Days"],
    "S1: Central": [f"{s1_service_level*100:.1f}%", f"{s1_q:,.0f}", f"{rec_s1_rop:,.0f}", f"{s1_actual_rop:,.0f}", f"${s1_sim_wc:,.0f}", f"${s1_peak_wc:,.0f}", f"{t_sales_1:,.0f}", "N/A"],
    "S2: Secondary": [f"{s2_sec_sl*100:.1f}%", f"{s2_sec_q:,.0f}", f"{rec_sec_rop:,.0f}", f"{s2_sec_actual_rop:,.0f}", f"${(avg_sec_oh_2+avg_sec_pipe_2)*s2_sec_cost:,.0f}", f"${peak_sys_val_2:,.0f} (System)", f"{t_sales_2:,.0f}", "—"],
    "S2: Main": [f"{s2_main_sl*100:.1f}%", f"{s2_main_q:,.0f}", f"{rec_main_rop:,.0f}", f"{s2_main_actual_rop:,.0f}", f"${(avg_main_oh_2+avg_main_pipe_2)*s2_main_cost:,.0f}", "—", "—", f"{delay_2}"],
    "S3: Secondary": [f"{s3_sec_sl*100:.1f}%", f"{s3_sec_q:,.0f}", f"{rec_s3_sec_rop:,.0f}", f"{s3_sec_actual_rop:,.0f}", f"${(avg_sec_oh_3+avg_sec_pipe_3)*s3_sec_cost:,.0f}", f"${peak_sys_val_3:,.0f} (System)", f"{t_sales_3:,.0f}", "—"],
    "S3: Main (Echelon)": [f"{s3_main_sl*100:.1f}%", f"{s3_main_q:,.0f}", f"{rec_echelon_rop:,.0f}", f"{s3_echelon_actual_rop:,.0f}", f"${(avg_main_oh_3+avg_main_pipe_3)*s3_main_cost:,.0f}", "—", "—", f"{delay_3}"],
    "S4: Sec. Policy Test": [f"{s4_sec_sl*100:.1f}%", f"{s4_sec_q:,.0f}", f"{rec_s4_sec_rop:,.0f}", f"{s4_sec_actual_rop:,.0f}", f"${(avg_sec_oh_4+avg_sec_pipe_4)*s4_sec_cost:,.0f}", f"${peak_sys_val_4:,.0f} (System)", f"{t_sales_4:,.0f}", "—"],
    "S4: Main Policy Test": [f"{s4_main_sl*100:.1f}%", f"{s4_main_q:,.0f}", f"{rec_echelon_rop if s4_main_policy == 'Continuous (s, Q)' else rec_echelon_s:,.0f}", f"{s4_echelon_actual_rop if s4_main_policy == 'Continuous (s, Q)' else s4_main_s:,.0f}", f"${(avg_main_oh_4+avg_main_pipe_4)*s4_main_cost:,.0f}", "—", "—", f"{delay_4}"]
}
st.table(pd.DataFrame(comparison_data).set_index("Metric"))

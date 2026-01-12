import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
from snowflake.snowpark.window import Window

# --- Snowflake Connection ---
def get_session():
    if "snowpark_session" not in st.session_state:
        connection_params = {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "password": st.secrets["snowflake"]["password"],
            "warehouse": st.secrets["snowflake"]["warehouse"],
            "database": st.secrets["snowflake"]["database"],
            "schema": st.secrets["snowflake"]["schema"],
            "role": st.secrets["snowflake"]["role"]
        }
        st.session_state.snowpark_session = Session.builder.configs(connection_params).create()
    return st.session_state.snowpark_session

session = get_session()

def get_call_stats(selected_date):
    """Calculates Total Calls and Billable Calls from the Call Stats table."""
    c = session.table("RAW.DIC.CALL_STATS")
    r = session.table("DBGA_TEST_ANALYTICS.DBGA_TEST_ORG_DATA.DIM_ACTIVE_AGENT_ROSTER_FLAT")
    o = session.table("DBGA_TEST_ANALYTICS.DBGA_TEST_BERMUDA.OLYMPUS")
    
    # Join call stats with roster and olympus
    return c.join(r, c["USER_ID"] == r["ICD_ID"])\
            .join(o, c["CALL_UID"] == o["CALL_UID"], "left")\
            .filter(F.to_date(c["TIMESTAMP"]) == str(selected_date))\
            .select(
                r["NAME"].alias("NAME"), 
                r["DEPARTMENT"].alias("DEPARTMENT"), 
                r["MANAGER"].alias("MANAGER"), 
                c["CALL_UID"].alias("CALL_UID"),
                o["ANSWERED_FLAG"],
                o["BILLABLE_FLAG"]
            )\
            .group_by("NAME", "DEPARTMENT", "MANAGER")\
            .agg(
                F.count("CALL_UID").alias("TOTAL_CALLS"),
                F.sum(
                    F.when(
                        (F.col("ANSWERED_FLAG") == "Y") & (F.col("BILLABLE_FLAG") == "Y"), 
                        1
                    ).otherwise(0)
                ).alias("BILLABLE_CALLS")
            )

def get_queue_behavior_data(selected_date):
    """Calculates Time/Session Metrics from Status Stats table."""
    s = session.table("RAW.DIC.CALL_IN_USER_STATUS_STATS")
    r = session.table("DBGA_TEST_ANALYTICS.DBGA_TEST_ORG_DATA.DIM_ACTIVE_AGENT_ROSTER_FLAT")
    
    s_user_id = F.call_builtin("TRY_TO_NUMBER", s["USER_ID"])
    r_icd_id = F.call_builtin("TRY_TO_NUMBER", r["ICD_ID"])

    # Explicitly selecting s["STATUS"] to distinguish from r["STATUS"]
    base_data = s.join(r, s_user_id == r_icd_id).filter(
        (s["USER_ID"].is_not_null()) & 
        (s["USER_ID"] != F.lit("0")) & 
        (F.to_date(s["TIMESTAMP"]) == str(selected_date))
    ).select(
        s["USER_ID"], 
        s["STATUS"].alias("STATUS"),
        s["WAIT_TIME"], 
        s["TIME_SPENT_IN_QUEUE"], 
        s["TIMESTAMP"],
        r["NAME"], 
        r["DEPARTMENT"], 
        r["MANAGER"]
    )

    window_session = Window.partition_by("USER_ID").order_by("TIMESTAMP").rows_between(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW)
    window_metrics = Window.partition_by("USER_ID", "QUEUE_SESSION_ID").order_by("TIMESTAMP").rows_between(Window.UNBOUNDED_PRECEDING, Window.UNBOUNDED_FOLLOWING)
    
    # Create session and metrics
    df = base_data.with_column("QUEUE_SESSION_ID", F.sum(F.when(F.col("STATUS") == 1, 1).otherwise(0)).over(window_session))
    
    df = df.with_column("FIRST_CALL_WAIT_TIME", 
                        F.first_value(F.when(F.col("STATUS") == 3, F.col("WAIT_TIME")).otherwise(None), ignore_nulls=True).over(window_metrics))
    
    df = df.with_column("SESSION_TOTAL_TIME", 
                        F.first_value(F.when(F.col("STATUS").is_null(), F.col("TIME_SPENT_IN_QUEUE")).otherwise(None), ignore_nulls=True).over(window_metrics))
    
    return df.with_column("CALL_DURATION_MINUTES", F.col("SESSION_TOTAL_TIME") - F.col("FIRST_CALL_WAIT_TIME"))

def get_sales_data(selected_date):
    """Calculates Sales count and total premium per agent from Policies table."""
    org = session.table("DBGA_TEST_ANALYTICS.DBGA_TEST_ORG_DATA.DIM_ACTIVE_AGENT_ROSTER_FLAT")
    p = session.table("RAW.CRM_DSB.POLICIES")
    
    return org.join(p, p["USER_ID"] == org["CRM_ID"], "left")\
            .filter(
                (F.to_date(p["SALE_MADE_DATE"]) == str(selected_date)) &
                (p["ANNUAL_PREMIUM"].is_not_null())
            )\
            .group_by(
                org["NAME"].alias("NAME"),
                org["DEPARTMENT"].alias("DEPARTMENT"),
                org["MANAGER"].alias("MANAGER")
            )\
            .agg(
                F.count("*").alias("SALES"),
                F.sum(p["ANNUAL_PREMIUM"]).alias("TOTAL_PREMIUM")
            )

# --- UI Setup ---
st.set_page_config(layout="wide")
st.title("📊 Agent Performance Dashboard")

selected_date = st.date_input("Select Date", value=None)

if selected_date:
    try:
        with st.spinner("Processing data from Snowflake..."):
            # 1. Pull data into DataFrames
            df_calls_pd = get_call_stats(selected_date).to_pandas()
            df_queue_pd = get_queue_behavior_data(selected_date).to_pandas()
            df_sales_pd = get_sales_data(selected_date).to_pandas()

            # 2. Aggregate Queue Stats at Agent Level
            if not df_queue_pd.empty:
                # Only use session-end rows (STATUS is null) to avoid duplicate summing
                session_end_rows = df_queue_pd[df_queue_pd["STATUS"].isna()]
                queue_summary = session_end_rows.groupby(["NAME", "DEPARTMENT", "MANAGER"]).agg({
                    "CALL_DURATION_MINUTES": "sum",
                    "SESSION_TOTAL_TIME": "sum"
                }).reset_index()
                
                # Convert minutes to hours
                queue_summary["CALL_DURATION_HOURS"] = (queue_summary["CALL_DURATION_MINUTES"] / 60).round(2)
                queue_summary["SESSION_TOTAL_HOURS"] = (queue_summary["SESSION_TOTAL_TIME"] / 60).round(2)
                queue_summary = queue_summary.drop(columns=["CALL_DURATION_MINUTES", "SESSION_TOTAL_TIME"])
            else:
                queue_summary = pd.DataFrame(columns=["NAME", "DEPARTMENT", "MANAGER", "CALL_DURATION_HOURS", "SESSION_TOTAL_HOURS"])
                
            # 3. Master Merge (Outer join to ensure we keep everyone)
            master_df = df_calls_pd.merge(queue_summary, on=["NAME", "DEPARTMENT", "MANAGER"], how="outer")
            master_df = master_df.merge(df_sales_pd, on=["NAME", "DEPARTMENT", "MANAGER"], how="outer").fillna(0)

            # --- Filters ---
            col1, col2 = st.columns(2)
            
            with col1:
                departments = ["All"] + sorted(master_df["DEPARTMENT"].unique().tolist())
                selected_dept = st.selectbox("Department", departments)
            
            with col2:
                # Filter managers based on selected department
                if selected_dept == "All":
                    available_managers = master_df["MANAGER"].unique().tolist()
                else:
                    available_managers = master_df[master_df["DEPARTMENT"] == selected_dept]["MANAGER"].unique().tolist()
                managers = ["All"] + sorted(available_managers)
                selected_manager = st.selectbox("Manager", managers)

            # Apply filters
            filtered_df = master_df.copy()
            if selected_dept != "All":
                filtered_df = filtered_df[filtered_df["DEPARTMENT"] == selected_dept]
            if selected_manager != "All":
                filtered_df = filtered_df[filtered_df["MANAGER"] == selected_manager]

            # --- Display ---
            st.subheader(f"Agent Activity for {selected_date}")
            cols = ["NAME", "DEPARTMENT", "MANAGER", "TOTAL_CALLS", "BILLABLE_CALLS", "SALES", "TOTAL_PREMIUM", "CALL_DURATION_HOURS", "SESSION_TOTAL_HOURS"]
            st.dataframe(filtered_df[cols].sort_values("TOTAL_CALLS", ascending=False), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please select a date to generate the analytics.")

import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
from snowflake.snowpark.window import Window

# --- Snowflake Connection ---
@st.cache_resource
def get_session():
    connection_params = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
        "role": st.secrets["snowflake"]["role"]
    }
    return Session.builder.configs(connection_params).create()

session = get_session()

def get_call_stats(selected_date):
    """Calculates Total Calls and Billable Calls from the Call Stats table."""
    c = session.table("RAW.DIC.CALL_STATS")
    r = session.table("DBGA_TEST_ANALYTICS.DBGA_TEST_ORG_DATA.DIM_ACTIVE_AGENT_ROSTER_FLAT")
    o = session.table("DBGA_TEST_ANALYTICS.DBGA_TEST_BERMUDA.OLYMPUS")
    
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
    
    df = base_data.with_column("QUEUE_SESSION_ID", F.sum(F.when(F.col("STATUS") == 1, 1).otherwise(0)).over(window_session))
    
    df = df.with_column("FIRST_CALL_WAIT_TIME", 
                        F.first_value(F.when(F.col("STATUS") == 3, F.col("WAIT_TIME")).otherwise(None), ignore_nulls=True).over(window_metrics))
    
    df = df.with_column("SESSION_TOTAL_TIME", 
                        F.first_value(F.when(F.col("STATUS").is_null(), F.col("TIME_SPENT_IN_QUEUE")).otherwise(None), ignore_nulls=True).over(window_metrics))
    
    return df.with_column("CALL_DURATION_MINUTES", F.col("SESSION_TOTAL_TIME") - F.col("FIRST_CALL_WAIT_TIME"))

def get_sales_data(selected_date):
    """Calculates Sales count per agent from GTL FTP data."""
    gtl = session.table("DBGA_TEST_ANALYTICS.DBGA_TEST_SALE_DATA.GTL_FTP_POLICY_STAGE_DATA_REFINED_FACT_WITH_PRODUCT_TYPES")
    cal = session.table("DBGA_TEST_ANALYTICS.DBGA_TEST_CALENDAR_DATA.DIM_CALENDAR_SF")
    
    gtl_with_date = gtl.with_column(
        "APP_DATE",
        F.to_date(F.call_builtin("CONVERT_TIMEZONE", F.lit("UTC"), F.lit("America/Chicago"), 
                                  F.col("NEW_APP_RECEIVED_DATE").cast("TIMESTAMP_NTZ")))
    )
    
    return gtl_with_date.join(cal, F.col("APP_DATE") == F.to_date(cal["CALENDAR_DATE"]))\
            .filter(F.to_date(cal["CALENDAR_DATE"]) == str(selected_date))\
            .group_by(gtl["AGENT_NAME"], gtl["DEPARTMENT"], gtl["MANAGER"])\
            .agg(F.count("*").alias("SALES"))

# --- UI Setup ---
st.set_page_config(layout="wide")
st.title("📊 Agent Performance Dashboard")

selected_date = st.date_input("Select Date", value=None)

if selected_date:
    try:
        with st.spinner("Processing data from Snowflake..."):
            df_calls_pd = get_call_stats(selected_date).to_pandas()
            df_queue_pd = get_queue_behavior_data(selected_date).to_pandas()
            df_sales_pd = get_sales_data(selected_date).to_pandas()
            
            df_sales_pd = df_sales_pd.rename(columns={"AGENT_NAME": "NAME"})

            if not df_queue_pd.empty:
                session_end_rows = df_queue_pd[df_queue_pd["STATUS"].isna()]
                queue_summary = session_end_rows.groupby(["NAME", "DEPARTMENT", "MANAGER"]).agg({
                    "CALL_DURATION_MINUTES": "sum",
                    "SESSION_TOTAL_TIME": "sum"
                }).reset_index()
                
                queue_summary["CALL_DURATION_HOURS"] = (queue_summary["CALL_DURATION_MINUTES"] / 60).round(2)
                queue_summary["SESSION_TOTAL_HOURS"] = (queue_summary["SESSION_TOTAL_TIME"] / 60).round(2)
                queue_summary = queue_summary.drop(columns=["CALL_DURATION_MINUTES", "SESSION_TOTAL_TIME"])
            else:
                queue_summary = pd.DataFrame(columns=["NAME", "DEPARTMENT", "MANAGER", "CALL_DURATION_HOURS", "SESSION_TOTAL_HOURS"])
                
            master_df = df_calls_pd.merge(queue_summary, on=["NAME", "DEPARTMENT", "MANAGER"], how="outer")
            master_df = master_df.merge(df_sales_pd, on=["NAME", "DEPARTMENT", "MANAGER"], how="outer").fillna(0)

            # --- Filters ---
            col1, col2 = st.columns(2)
            
            with col1:
                departments = ["All"] + sorted(master_df["DEPARTMENT"].unique().tolist())
                selected_dept = st.selectbox("Department", departments)
            
            with col2:
                if selected_dept == "All":
                    available_managers = master_df["MANAGER"].unique().tolist()
                else:
                    available_managers = master_df[master_df["DEPARTMENT"] == selected_dept]["MANAGER"].unique().tolist()
                managers = ["All"] + sorted(available_managers)
                selected_manager = st.selectbox("Manager", managers)

            filtered_df = master_df.copy()
            if selected_dept != "All":
                filtered_df = filtered_df[filtered_df["DEPARTMENT"] == selected_dept]
            if selected_manager != "All":
                filtered_df = filtered_df[filtered_df["MANAGER"] == selected_manager]

            # --- Display ---
            st.subheader(f"Agent Activity for {selected_date}")
            cols = ["NAME", "DEPARTMENT", "MANAGER", "TOTAL_CALLS", "BILLABLE_CALLS", "SALES", "CALL_DURATION_HOURS", "SESSION_TOTAL_HOURS"]
            st.dataframe(filtered_df[cols].sort_values("TOTAL_CALLS", ascending=False), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please select a date to generate the analytics.")

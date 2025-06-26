import streamlit as st
from components.yahoo_data import get_top_gainers, get_top_losers, get_most_active

def show_impact_stocks():
    st.subheader("📈 الأعلى ارتفاعًا")
    gainers = get_top_gainers()
    if not gainers.empty:
        st.dataframe(gainers, use_container_width=True)
    else:
        st.warning("⚠️ لم يتم العثور على بيانات للأسهم المرتفعة.")

    st.subheader("📉 الأعلى هبوطًا")
    losers = get_top_losers()
    if not losers.empty:
        st.dataframe(losers, use_container_width=True)
    else:
        st.warning("⚠️ لم يتم العثور على بيانات للأسهم الهابطة.")

    st.subheader("🔥 الأكثر تداولًا")
    active = get_most_active()
    if not active.empty:
        st.dataframe(active, use_container_width=True)
    else:
        st.warning("⚠️ لم يتم العثور على بيانات للأسهم الأكثر تداولًا.")

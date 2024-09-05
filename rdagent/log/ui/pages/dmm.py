import streamlit as st
from streamlit_javascript import st_javascript
from PIL import Image
icon = Image.open("./git_ignore_folder/rd_icon.png")
st.set_page_config(layout="wide", page_title="Medical Model Implementation Demo", page_icon=icon)

lang = st.query_params.get("lang")
header_c1, header_c3 = st.columns([1, 6], vertical_alignment="center")
with header_c1:
    st.image("https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RE1Mu3b?ver=5c31")
with header_c3:
    st.markdown(
        """
    <h1>
        RD-Agent:<br><span style="font-size: 32px;">LLM-based autonomous evolving agents for industrial data-driven R&D</span>
    </h1>
    """,
        unsafe_allow_html=True,
    )


lc, rc = st.columns([2,7])
with lc:
    st.markdown("➡️ [**Demo App**](..)")
    st.subheader("Demo videos🎥", divider='violet')
    st.markdown("""
- 💹[**Finance Model Implementation**](model_loop)

- 💹[**Finance Data Building**](factor_loop)

- 💹[**Finance Data Building (from Reports)**](report_factor)

- 🏭[**General Model Implementation**](report_model)

- 🩺[**Medical Model Implementation**](dmm)

""")
    if not lang:
        try:
            lang = str(st_javascript("window.navigator.language"))
        except:
            lang = "en"

with rc:
    col1, col2 = st.columns([5,1], vertical_alignment="center")
    with col2:
        use_cn = st.radio(label="language", options=["**中文**", "**English**"], index=0 if "zh" in lang else 1, label_visibility="collapsed", horizontal=True) == "**中文**"
    with col1:
        st.subheader("🩺Medical Model Implementation Demo" + (" (中文)" if use_cn else " (English)"))
    
    if use_cn:
        st.video("videos/medical_cn.mp4")
    else:
        st.video("videos/medical_en.mp4")
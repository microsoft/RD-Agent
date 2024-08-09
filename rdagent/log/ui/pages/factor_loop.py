import streamlit as st
st.set_page_config(layout="wide", page_title="Finance Data Building Demo", page_icon="💹")

header_c1, header_c3 = st.columns([1, 6], vertical_alignment="center")
with st.container():
    with header_c1:
        st.image("https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RE1Mu3b?ver=5c31")
    with header_c3:
        st.markdown(
            """
        <h1>
            RD-Agent:<br>LLM-based autonomous evolving agents for industrial data-driven R&D
        </h1>
        """,
            unsafe_allow_html=True,
        )

lc, rc = st.columns([1,5])
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
with rc:
    st.subheader("💹Finance Data Building Demo")
    st.video("videos/factor_loop_cn.mp4")
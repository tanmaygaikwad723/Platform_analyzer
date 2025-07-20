import streamlit as st
st.set_page_config(layout="wide")
from PIL import Image
from api import generate_output



st.session_state["container2"] = False

# st.title("Platform Analyzer")
st.markdown(
    """
    <style>
    .custom-title {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        padding: 10px;
        z-index: 99999999;
        text-align: center;
        font-family: Inter, monospace;
        font-size: 60px;
        font-weight: 700;
        line-height: 120%;
        padding-top: 24px;
    }
    </style>
    <span class='custom-title'>Platform Analyzer</span>
    """,
    unsafe_allow_html=True
)

with st.container(height=200, border=False):
    st.subheader("Enter the link of url")
    url = st.text_input("", width=1000, label_visibility="collapsed", placeholder="https://reddit.com/")
    if st.button("Analyze"):
        negatives_sorted, neutral_sorted, positive_sorted = generate_output(url)
        st.session_state["container2"] = True

if st.session_state["container2"]:
  with st.container():
    col1, col2, col3 = st.columns((1,1,1), gap="medium", vertical_alignment="center", border=True)
    col1.header("Negative Cloud")
    col1.image(Image.open("negative_cloud.png"), use_container_width=True)
    col1.dataframe(negatives_sorted[0], use_container_width=True, hide_index=True, row_height=30)
    col2.header("Neutral Cloud")
    col2.image(Image.open("neutral_cloud.png"), use_container_width=True)
    col2.dataframe(neutral_sorted[0], use_container_width=True, hide_index=True, row_height=30)
    col3.header("Positive Cloud")
    col3.image(Image.open("positive_cloud.png"), use_container_width=True)
    col3.dataframe(positive_sorted[0], use_container_width=True, hide_index=True, row_height=30)
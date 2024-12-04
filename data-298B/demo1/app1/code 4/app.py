import streamlit as st

from tuneavideo_app import tune_a_video
from zeroscope_app import zero_scope
from cogvideo_app import cogvideox_2b
from videocrafter_app import videocrafter
from t2v_zero_app import t2v_zero
from videolavit_app import videolavit
from higen_app import higen

def reset_session():
    st.session_state.clear()

st.sidebar.button("Reset", on_click=reset_session)

def main():
    st.title('Text to Video')

    models = {
        "None": lambda: None,
        "ZeroScope": zero_scope,
        "CogVideoX-2b": cogvideox_2b,
        # "VideoCrafter": videocrafter,
        # "Text2Video-Zero": t2v_zero,
        "VideoLaViT": videolavit,
        "higen": higen,
        # "Tune-A-Video": tune_a_video,
    }

    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "None"

    page = st.selectbox(
        "Select a Model", 
        list(models.keys()), 
        index=list(models.keys()).index(st.session_state["selected_page"]),
        key="selected_page_dropdown",
    )

    if page != st.session_state["selected_page"]:
        reset_session()
        st.session_state["selected_page"] = page

    func = models[page]
    func()

if __name__ == "__main__":
    main()
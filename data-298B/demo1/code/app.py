import streamlit as st

from tuneavideo_app import tune_a_video
from zeroscope_app import zero_scope
# from cogvideo_app import cogvideox_5b
from videocrafter_app import videocrafter
from t2v_zero_app import t2v_zero
from video_editing_app import edit_pix2pix

def main():
    st.title('Text to Video')

    models = {
        "None": lambda: None,
        "ZeroScope": zero_scope,
        "Tune-A-Video": tune_a_video,
        # "CogVideoX-5b": cogvideox_5b,
        "VideoCrafter": videocrafter,
        "Text2Video-Zero": t2v_zero,
        "edit_pix2pix": edit_pix2pix
    }

    page = st.selectbox("Select a Model", list(models.keys()))
    func = models[page]
    func()

if __name__ == "__main__":
    main()


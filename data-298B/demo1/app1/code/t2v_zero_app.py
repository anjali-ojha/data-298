import torch
import streamlit as st

from diffusers import TextToVideoZeroPipeline
import imageio

def t2v_zero():
    # https://huggingface.co/runwayml/stable-diffusion-v1-5
    ckpt_path = "/data/checkpoints/stable-diffusion-v1-5"

    st.info('Loading Text2VideoZero model')
    pipe = TextToVideoZeroPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16).to("cuda")

    st.info("Model loaded")
    prompt = st.text_input("Enter your text prompt:")

    if st.button('Generate'):
        st.info("Generating video")

        fn = f"/tmp/text2video_{prompt.replace(' ','_')}.mp4"
        result = pipe(prompt=prompt).images
        result = [(r * 255).astype("uint8") for r in result]
        imageio.mimsave(fn, result, fps=4)
        
        with open(fn, 'rb') as f:
            video_bytes = f.read()
            st.markdown('Generated video')
            st.video(video_bytes, format='video/mp4', start_time=0)
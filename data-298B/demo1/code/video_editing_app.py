import torch
import streamlit as st
import imageio

from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

def edit_pix2pix():
    # https://huggingface.co/timbrooks/instruct-pix2pix
    ckpt_path = "/data/checkpoints/instruct-pix2pix"

    video_file = st.file_uploader("Choose a source video file", type=["mp4", "avi", "mov", "mkv"])
    if video_file is not None:
        st.markdown('Source video')
        st.video(video_file)
    else:
        st.write("Please upload a video file.")
        return


    st.info('Loading Pix2Pix model')

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16).to("cuda")
    pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
    st.info("Model loaded")



    prompt = st.text_input("Enter your text prompt:")



    if st.button('Generate'):
        st.info("Generating video")
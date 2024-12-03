import os
import sys
import torch
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), 'Tune-A-Video'))

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid

def tune_a_video():
    video_file = st.file_uploader("Choose a source video file", type=["mp4", "avi", "mov", "mkv"])
    if video_file is not None:
        st.markdown('Source video')
        st.video(video_file)
    else:
        st.write("Please upload a video file.")
        return

    st.info('Loading Tune-A-Video model')

    # Source: https://huggingface.co/CompVis/stable-diffusion-v1-4
    pretrained_model_path = "/data/Tune-A-Video/checkpoints/stable-diffusion-v1-4"

    # Get this model using 
    # accelerate launch train_tuneavideo.py --config="configs/rabbit-watermelon.yaml"
    my_model_path = "/data/Tune-A-Video/outputs/rabbit-watermelon/"

    unet = UNet3DConditionModel.from_pretrained(
        my_model_path, subfolder='unet', torch_dtype=torch.float16
    ).to('cuda')

    pipe = TuneAVideoPipeline.from_pretrained(
        pretrained_model_path, unet=unet, torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    ddim_inv_latent = torch.load(
        f"{my_model_path}/inv_latents/ddim_latent-500.pt"
    ).to(torch.float16)

    st.info("Model loaded")
    prompt = st.text_input("Enter your text prompt:")

    if st.button('Generate'):
        st.info("Generating video")
        video = pipe(
            prompt,
            latents=ddim_inv_latent,
            video_length=24,
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=12.5
        ).videos

        fn = f"/tmp/tune_a_video_{prompt.replace(' ', '_')}.gif"
        save_videos_grid(video, fn)
        st.markdown('Generated video')
        st.image(fn, caption=prompt)


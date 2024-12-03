import streamlit as st
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

st.title('Text to Video')

video_file = './data/rabbit-watermelon.mp4'
with open(video_file, 'rb') as f:
    video_bytes = f.read()
    st.markdown('Source video')
    st.video(video_bytes, format='video/mp4', start_time=0)

st.info('Loading the model')
pretrained_model_path = "/data/Tune-A-Video/checkpoints/stable-diffusion-v1-4"
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

    fn = f"/tmp/{prompt.replace(' ', '_')}.gif"
    save_videos_grid(video, fn)
    st.image(fn, caption=prompt)

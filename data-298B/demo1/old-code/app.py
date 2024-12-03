import os
import sys
import streamlit as st
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'Tune-A-Video'))

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid

from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils.export_utils import export_to_video

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
    # pretrained_model_path = "/data/Tune-A-Video/checkpoints/stable-diffusion-v1-4"
    pretrained_model_path = "/Users/hims/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4"

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


def zero_scope():
    # Source: https://huggingface.co/cerspense/zeroscope_v2_576w/tree/main
    # model_path = "/data/checkpoints/zeroscope_v2_576w"
    model_path = "/Users/hims/.cache/huggingface/hub/models--cerspense--zeroscope_v2_576w/snapshots/6963642a64dbefa93663d1ecebb4ceda2d9ecb28"

    st.info('Loading ZeroScope model')

    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32).to("cpu")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload(device="cpu")

    st.info("Model loaded")
    prompt = st.text_input("Enter your text prompt:")

    if st.button('Generate'):
        st.info("Generating video")
        video = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames[0]
        
        fn = f"/tmp/text2video_{prompt.replace(' ','_')}.mp4"
        export_to_video(video, output_video_path=fn)
        with open(fn, 'rb') as f:
            video_bytes = f.read()
            st.markdown('Generated video')
            st.video(video_bytes, format='video/mp4', start_time=0)

def main():
    st.title('Text to Video')

    page = st.selectbox("Select a Model", [None, "ZeroScope", "Tune-A-Video"])

    if page == "Tune-A-Video":
        tune_a_video()
    elif page == 'ZeroScope':
        zero_scope()
  

if __name__ == "__main__":
    main()


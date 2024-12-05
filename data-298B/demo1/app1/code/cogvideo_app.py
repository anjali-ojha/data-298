import torch
import streamlit as st

from diffusers import CogVideoXPipeline

from diffusers.utils.export_utils import export_to_video

def cogvideox_2b():
    # https://huggingface.co/THUDM/CogVideoX-2b
    model_path = "/data/checkpoints/CogVideoX-2b"
    if 'pipe' not in st.session_state:
        st.session_state.pipe = None

    if st.session_state.pipe is None:
        print('Loading CogVideoX model')
        st.info('Loading CogVideoX model')
        pipe = CogVideoXPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        st.info("Model loaded")
        st.session_state.pipe = pipe
        st.rerun()
    else:
        st.info("Model is ready for video generation.")
        prompt = st.text_input("Enter your text prompt:")

        if st.button('Generate'):
            st.info("Generating video")

            video = st.session_state.pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]

            fn = f"/tmp/cogvideox_{prompt.replace(' ','_')}.mp4"
            export_to_video(video, output_video_path=fn, fps=8)
            with open(fn, 'rb') as f:
                video_bytes = f.read()
                st.markdown('Generated video')
                st.video(video_bytes, format='video/mp4', start_time=0)


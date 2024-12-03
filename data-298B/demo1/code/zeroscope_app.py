import streamlit as st

from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils.export_utils import export_to_video


def zero_scope():
    # Source: https://huggingface.co/cerspense/zeroscope_v2_576w/tree/main
    model_path = "/data/checkpoints/zeroscope_v2_576w"

    st.info('Loading ZeroScope model')

    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

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

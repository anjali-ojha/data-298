import torch
import streamlit as st
import imageio
from PIL import Image
import tempfile


from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

def edit_pix2pix():
    # https://huggingface.co/timbrooks/instruct-pix2pix
    ckpt_path = "/data/checkpoints/instruct-pix2pix"

    video_path = st.file_uploader("Choose a source video file", type=["mp4", "avi", "mov", "mkv"])
    if video_path is not None:
        st.markdown('Source video')
        st.video(video_path)
    else:
        st.write("Please upload a video file.")
        return

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_path.read())
        temp_video_path = tmp_file.name

    reader = imageio.get_reader(temp_video_path, "ffmpeg")
    frame_count = 32
    video = [Image.fromarray(reader.get_data(i)).resize((320, 320)) for i in range(frame_count)]
    st.write("Video loaded successfully!")

    st.info('Loading Pix2Pix model')

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16).to("cuda")
    pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=1))
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_tiling()

    st.info("Model loaded")

    prompt = st.text_input("Enter your text prompt:")

    if st.button('Generate'):

        st.info("Generating video")

        result = pipeline(prompt=[prompt] * len(video), image=video).images
        fn = f"/tmp/edited_{prompt.replace(' ','_')}.mp4"
        imageio.mimsave(fn, result, fps=8)

        with open(fn, 'rb') as f:
            video_bytes = f.read()
            st.markdown('Generated video')
            st.video(video_bytes, format='video/mp4', start_time=0)

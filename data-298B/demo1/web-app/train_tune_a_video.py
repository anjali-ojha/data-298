import os
import sys
import torch
import streamlit as st
import yaml
import subprocess
import re
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), 'Tune-A-Video'))

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid


def train_tune_a_video():
    # if 'model_trained' not in st.session_state:
    #     st.session_state.model_trained = False
    # if 'pipe' not in st.session_state:
    #     st.session_state.pipe = None
    # if 'output_dir' not in st.session_state:
    #     st.session_state.output_dir = None

    video_file = st.file_uploader("Choose a source video file", type=["mp4", "avi", "mov", "mkv"])
    if video_file is not None:
        st.markdown('Source video')
        st.video(video_file)
    else:
        st.write("Please upload a video file.")
        return

    src_prompt = st.text_input("Enter your text prompt for this video:")

    # if st.session_state.model_trained is False:
    if st.button('Train the model') is False:
        return
    st.info('Training Tune-A-Video model')

    # Source: https://huggingface.co/CompVis/stable-diffusion-v1-4
    pretrained_model_path = "/data/Tune-A-Video/checkpoints/stable-diffusion-v1-4"
    output_dir = f"/data/Tune-A-Video/outputs/tune_a_video_{src_prompt.replace(' ', '_')}"

    with open("./Tune-A-Video/configs/training.yaml", "r") as file:
        config = yaml.safe_load(file)

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        temp_video_path = tmp_file.name

        # Modify specific values
        max_train_steps = 100  # 500
        config['pretrained_model_path'] = pretrained_model_path
        config['output_dir'] = output_dir
        config['max_train_steps'] = max_train_steps
        config['validation_steps'] = max_train_steps
        config['train_data']['video_path'] = temp_video_path
        config['train_data']['prompt'] = src_prompt
        config['validation_data']['prompts'] = [src_prompt]

        with open("/tmp/config.yaml", "w") as file:
            yaml.safe_dump(config, file)

        command = ["accelerate", "launch", "train_tuneavideo.py", "--config=/tmp/config.yaml"]
        process = subprocess.Popen(command, cwd='./Tune-A-Video', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)

        progress_bar = st.progress(0)
        step_info = st.empty()
        for line in process.stdout:
            print(line, end="")
            match = re.search(r"Steps:\s+(\d+)%\|.*\|\s+(\d+)/(\d+).*", line)
            match2 = re.search(r"(\d+)%\|.*\|\s+(\d+)/(\d+).*", line)
            if match:
                # Extract details from the parsed line
                percent = int(match.group(1))
                current_step = int(match.group(2))
                total_steps = int(match.group(3))

                # Update the Streamlit progress bar and display additional metrics
                progress_bar.progress(percent / 100)
                step_info.write(f"Step {current_step} of {total_steps}")

            elif match2:
                # Extract details from the parsed line
                percent = int(match2.group(1))
                current_step = int(match2.group(2))
                total_steps = int(match2.group(3))

                # Update the Streamlit progress bar and display additional metrics
                progress_bar.progress(percent / 100)
                step_info.write(f"Validation {current_step} of {total_steps}")

            sys.stdout.flush()

        stdout, stderr = process.communicate()
        if stderr:
            print("\nError output:\n", stderr)
            return

    # Get this model using
    # accelerate launch train_tuneavideo.py --config="configs/rabbit-watermelon.yaml"
    st.info("Training Done. Loading model...")
    st.session_state.model_trained = True

    unet = UNet3DConditionModel.from_pretrained(
        output_dir, subfolder='unet', torch_dtype=torch.float16
    ).to('cuda')

    pipe = TuneAVideoPipeline.from_pretrained(
        pretrained_model_path, unet=unet, torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    ddim_inv_latent = torch.load(
        f"{output_dir}/inv_latents/ddim_latent-{max_train_steps}.pt"
    ).to(torch.float16)

    st.session_state.pipe = pipe
    st.session_state.ddim_inv_latent = ddim_inv_latent
    st.info("Model loaded")
    st.experimental_rerun()
    # else:
    #     st.info("Model is ready for video generation.")
    #     prompt = st.text_input("Enter your text prompt for the tuned video:")
    #
    #     if st.button('Generate'):
    #         st.info("Generating video")
    #         video = st.session_state.pipe(
    #             prompt,
    #             latents=st.session_state.ddim_inv_latent,
    #             video_length=24,
    #             height=512,
    #             width=512,
    #             num_inference_steps=50,
    #             guidance_scale=12.5
    #         ).videos
    #
    #         fn = f"/tmp/tune_a_video_{prompt.replace(' ', '_')}.gif"
    #         save_videos_grid(video, fn)
    #         st.markdown('Generated video')
    #         st.image(fn, caption=prompt)


def app(clear=True):
    train_tune_a_video()
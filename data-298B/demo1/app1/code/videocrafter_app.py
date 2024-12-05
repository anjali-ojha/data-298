import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'VideoCrafter'))

from omegaconf import OmegaConf
from vc_utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint,save_videos, batch_ddim_sampling

import streamlit as st


def videocrafter():
    # https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt
    ckpt_path = "/data/checkpoints/VideoCrafter2/model.ckpt"

    config_file='./VideoCrafter/configs/inference_t2v_512_v2.0.yaml'
    config = OmegaConf.load(config_file)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint']=False   

    st.info('Loading VideoCrafter model')

    model = instantiate_from_config(model_config)
    assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    model.cuda()
    st.info("Model loaded")
    prompt = st.text_input("Enter your text prompt:")

    if st.button('Generate'):
        st.info("Generating video")

        fps = 16
        batch_size=1
        channels = model.model.diffusion_model.in_channels
        frames = model.temporal_length
        h, w = 320 // 8, 512 // 8
        noise_shape = [batch_size, channels, frames, h, w]
        steps = 40
        eta = 1.0
        cfg_scale = 12.0

        text_emb = model.get_learned_conditioning([prompt])
        cond = {"c_crossattn": [text_emb], "fps": fps}
        batch_samples = batch_ddim_sampling(model, cond,
                                            noise_shape,
                                            n_samples=1,
                                            ddim_steps=steps,
                                            ddim_eta=eta,
                                            cfg_scale=cfg_scale)
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
        prompt_str = prompt_str[:30]

        os.makedirs("/tmp/videocrafter/", exist_ok=True)
        save_videos(batch_samples, "/tmp/videocrafter/",
                     filenames=[prompt_str], fps=8)

        fn = f"/tmp/videocrafter/{prompt_str}.mp4"
        with open(fn, 'rb') as f:
            video_bytes = f.read()
            st.markdown('Generated video')
            st.video(video_bytes, format='video/mp4', start_time=0)
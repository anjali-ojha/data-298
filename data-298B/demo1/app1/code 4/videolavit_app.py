import streamlit as st

import os
import sys
import torch
from diffusers.utils import load_image, export_to_video


def videolavit():
    # Source: https://huggingface.co/rain1011/Video-LaVIT-v1
    model_path = "/data/checkpoints/Video-LaVIT-v1"

    model_dtype='fp16'
    torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16

    sys.path.append(os.path.join(os.path.dirname(__file__), 'VideoLaVIT'))
    from models import build_model

    if 'model' not in st.session_state:
        st.session_state.model = None

    if st.session_state.model is None:
        print('Loading Video-LaVIT model')
        st.info('Loading Video-LaVIT model')

        model = build_model(model_path=model_path, model_dtype=model_dtype, local_files_only=True, 
                device_id=None, use_xformers=True, understanding=False,)
        model.cuda()
        st.session_state.model = model
        st.info("Model loaded")
        st.rerun()
    else:
        st.info("Model is ready for video generation.")
        prompt = st.text_input("Enter your text prompt:")

        if st.button('Generate'):
            st.info("Generating video")
            ratio_dict = {
                '1:1' : (1024, 1024),
                '1:2' : (576, 1024),
            }
            ratio = '1:2'
            height, width = ratio_dict[ratio]
            if ratio == '1:2':
                video_width = 576
                video_height = 320
            else:
                assert ratio == '1:1'
                video_width = 512
                video_height = 512

            with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                videos, keyframes = st.session_state.model.generate_video(prompt, width=width, height=height, num_return_images=1, 
                        video_width=video_width, video_height=video_height, guidance_scale_for_llm=4.0, 
                        guidance_scale_for_decoder=7.0, num_inference_steps=50, top_k=50,)
            st.info(f"Finished generating video")

            fn = f"/tmp/videolavit_{prompt.replace(' ','_')}.mp4"
            export_to_video(videos[0], output_video_path=fn)
            with open(fn, 'rb') as f:
                video_bytes = f.read()
                st.markdown('Generated video')
                st.video(video_bytes, format='video/mp4', start_time=0)
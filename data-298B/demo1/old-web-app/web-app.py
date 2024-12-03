from streamlit_chatbox import *
import os
import sys
import streamlit as st
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'Tune-A-Video'))

# from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
# from tuneavideo.models.unet import UNet3DConditionModel
# from tuneavideo.util import save_videos_grid
# from diffusers import CogVideoXPipeline

from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils.export_utils import export_to_video


import time

def main():
    models = ["Tune-A-Video Model", "CogVideoX-5b Model", "ZeroScope Model"]
    print("starting from beginning")

    llm = FakeLLM()

    chat_box = ChatBox(
        use_rich_markdown=True,  # use streamlit-markdown
        user_theme="green",  # see streamlit_markdown.st_markdown for all available themes
        assistant_theme="blue",
    )
    chat_box.use_chat_name("chat1")  # add a chat conversation

    def on_chat_change():
        chat_box.use_chat_name(st.session_state["chat_name"])
        chat_box.context_to_session()  # restore widget values to st.session_state when chat name changes

    with st.sidebar:
        st.subheader('Start Chat using Streamlit')
        chat_name = st.selectbox("Chat Session:", ["default", "chat1"], key="chat_name", on_change=on_chat_change)
        chat_box.use_chat_name(chat_name)
        streaming = st.checkbox('Streaming', key="streaming")
        in_expander = st.checkbox('Show messages in expander', key="in_expander")
        show_history = st.checkbox('Show session state', key="show_history")
        chat_box.context_from_session(exclude=["chat_name"])  # save widget values to chat context

        st.divider()

        btns = st.container()

    chat_box.init_session()
    chat_box.output_messages()

    def on_feedback(feedback, chat_history_id: str = "", history_index: int = -1):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)  # convert emoji to integer
        st.session_state["need_rerun"] = False

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Welcome to Feedback",
    }

    btns.download_button(
        "Export Markdown",
        "".join(chat_box.export2md()),
        file_name=f"chat_history.md",
        mime="text/markdown",
    )

    btns.download_button(
        "Export Json",
        chat_box.to_json(),
        file_name="chat_history.json",
        mime="text/json",
    )

    if btns.button("Clear History"):
        chat_box.init_session(clear=True)
        st.experimental_rerun()

    if show_history:
        st.write(st.session_state)
        

    picked_model = None if 'model' not in st.session_state else st.session_state['model']
    print(f"{picked_model = }")
    
    if query := st.chat_input('Input your question here'):
        query = query.strip()
        chat_box.user_say(query)
        # st.session_state.query = "query"
        if query.startswith(("generate", "Generate", "create", "show")):
            show_model_flag = True
            chat_box.ai_say([
                Markdown("Pick a model for video generation ", in_expander=in_expander, expanded=True, title="answer")
            ])

            cols = st.columns(len(models))
            
            def click_button(model, query):
                print("\tgot i = ", i)
                st.session_state.model = model
                st.session_state.query = query
            
            for i in range(len(models)) :
                # if cols[i].button(models[i], on_click=click_button(models[i], query))
                if cols[i].button(models[i], on_click=lambda m=models[i], q=query: click_button(m, q)):
                    print("model")
        

        else:
            if streaming:
                generator = llm.chat_stream(query)
                elements = chat_box.ai_say([
                    Markdown("thinking", in_expander=in_expander, expanded=True, title="answer")
                ])
                time.sleep(1)
                text = ""
                for x, docs in generator:
                    text += x
                    chat_box.update_msg(text, element_index=0, streaming=True)
                chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
                chat_box.update_msg("\n\n".join(docs), element_index=1, streaming=False, state="complete")
                chat_history_id = "some id"
                chat_box.show_feedback(
                    **feedback_kwargs,
                    key=chat_history_id,
                    on_submit=on_feedback,
                    kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1}
                )
            else:
                text, docs = llm.chat(query)
                chat_box.ai_say([
                    Markdown(text, in_expander=in_expander, expanded=True, title="answer"),
                ])

    if picked_model is not None:
        query = None if 'query' not in st.session_state else st.session_state['query']

        chat_box.ai_say([
            Markdown(f"You have chosen the model = {picked_model}", in_expander=in_expander, expanded=True, title="answer"),
            Markdown(f"Generating Video for query = '{query}'", in_expander=in_expander, expanded=True, title="answer"),
        ])
        # progress_bar = st.progress(0)
        
        # with st.spinner("Processing... Please wait, this may take a few minutes"):
        #     video_bytes = get_video()
        try:
            get_video(picked_model, query)
            st.session_state.query = None
            st.session_state.model = None   
        except Exception as e:
            print("error")
            st.session_state.query = None
            st.session_state.model = None   
            raise e
 
def get_video(model, query):
    time.sleep(0.5)
    print("starting ... ")
    if model == "Tune-A-Video Model":
        fn = tune_a_video(query)
    elif model == 'ZeroScope Model':
        fn = zero_scope(query)
    elif model == 'CogVideoX-5b Model':
        fn = cogvideox_5b(query)
    
        
    # with open("/Users/hims/Downloads/1066674784.mp4", 'rb') as f:
    #     video_bytes = f.read()

    # with open(fn, 'rb') as f:
    #     video_bytes = f.read()
    #     st.markdown('Generated video')
    #     st.video(video_bytes, format='video/mp4', start_time=0)
        
    # return video_bytes           

def tune_a_video(prompt):
    # video_file = st.file_uploader("Choose a source video file", type=["mp4", "avi", "mov", "mkv"])
    # Future work
    # if video_file is not None:
    #     st.markdown('Source video')
    #     st.video(video_file)
    # else:
    #     st.write("Please upload a video file.")
    #     return

    st.info('Loading Tune-A-Video model')

    # Source: https://huggingface.co/CompVis/stable-diffusion-v1-4
    # pretrained_model_path = "/data/Tune-A-Video/checkpoints/stable-diffusion-v1-4"
    pretrained_model_path = "/home/ubuntu/stable-diffusion-v1-4"

    # Get this model using 
    # accelerate launch train_tuneavideo.py --config="configs/rabbit-watermelon.yaml"
    # my_model_path = "/data/Tune-A-Video/outputs/rabbit-watermelon/"
    my_model_path = "/home/ubuntu/Tune-A-Video/outputs/rabbit-watermelon/"

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
    # prompt = st.text_input("Enter your text prompt:")

    # if st.button('Generate'):
    #     st.info("Generating video")
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
    return fn


def zero_scope(prompt):
    # Source: https://huggingface.co/cerspense/zeroscope_v2_576w/tree/main
    # model_path = "/data/checkpoints/zeroscope_v2_576w"
    model_path = "/home/ubuntu/.cache/huggingface/hub/models--cerspense--zeroscope_v2_576w/snapshots/6963642a64dbefa93663d1ecebb4ceda2d9ecb28"

    st.info('Loading ZeroScope model')

    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    st.info("Model loaded")
    # prompt = st.text_input("Enter your text prompt:")

    # if st.button('Generate'):
    #     st.info("Generating video")
    video = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames[0]

    fn = f"/tmp/text2video_{prompt.replace(' ','_')}.mp4"
    export_to_video(video, output_video_path=fn)
    with open(fn, 'rb') as f:
        video_bytes = f.read()
        st.markdown('Generated video')
        st.video(video_bytes, format='video/mp4', start_time=0)
    return fn


def cogvideox_5b(prompt):
    # https://huggingface.co/THUDM/CogVideoX-5b
    # model_path = "/data/checkpoints/CogVideoX-5b"
    model_path = "/home/ubuntu/.cache/huggingface/hub/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad"

    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    st.info("Model loaded")
    # prompt = st.text_input("Enter your text prompt:")

    # if st.button('Generate'):
    #     st.info("Generating video")

    video = pipe(
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
    return fn
    





# Ensure main is called when the script is executed
if __name__ == "__main__":
    # print("starting main", st.session_state)
    st.title("Text To Video Application")
    main()

from datetime import datetime

from diffusers import DiffusionPipeline
from streamlit_chatbox import ChatBox, FakeLLM, Markdown
import os
import sys
import streamlit as st
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../code/Tune-A-Video/'))
print(os.path.join(os.path.dirname(__file__), '../code/Tune-A-Video/'))

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils.export_utils import export_to_video
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init(expanded=True):
    print("init the page", expanded)
    container = st.container()
    container.title("Zero-Scope Model")
    expander = container.expander("Model Details and Architecture ", expanded=expanded)

    expander.write("""The ZeroScope Text-to-Video Model is a zero-shot video generation model designed to convert text prompts into video without additional training on video-specific data. Building on pre-trained image-generation models like Stable Diffusion, ZeroScope applies a temporal adaptation to these models, enabling them to generate sequential frames that make up a video. Unlike traditional video models, ZeroScope can operate "out of the box," leveraging the extensive visual understanding encoded in large-scale text-to-image models to create videos directly from textual descriptions. This approach eliminates the need for extensive video training datasets, making ZeroScope more flexible and accessible for video creation from text.

ZeroScope introduces a spatio-temporal attention mechanism that maintains continuity and consistency across frames, ensuring objects, actions, and scenes evolve naturally as they would in real life. This attention mechanism allows the model to carry over visual information from one frame to the next, providing a coherent motion flow and reducing artifacts like flickering or jittering between frames. ZeroScope's focus on efficient diffusion in a compressed latent space allows for quick processing and video output while retaining high visual fidelity, making it an exciting tool for creative applications where generating video from text was previously impractical.
    """)
    # expander.image("assets/images/tune-a-video.png", caption="Zero-Scope Model System Diagram")

    container.write("This is a demo of the Zero-Scope model. You can generate a video from a text prompt. ")


def get_variants(query):
    variants = [query, query]
    return variants


def get_file_name(query):
    now = datetime.now()
    filename = now.strftime("%Y%m%d-%H%M%S")
    print(f"For {query = }, {filename}")
    return filename


def get_video_from_model(prompt):
    st.info('Loading Zero-Scope model')

    # Source: https://huggingface.co/cerspense/zeroscope_v2_576w/tree/main
    # model_path = "/data/checkpoints/zeroscope_v2_576w"
    fn = f"/home/ubuntu/videos/zero_scope_{get_file_name(prompt)}.mp4"
    print(f"Writing video at the path = ${fn}")

    model_path = "/home/ubuntu/.cache/huggingface/hub/models--cerspense--zeroscope_v2_576w/snapshots/6963642a64dbefa93663d1ecebb4ceda2d9ecb28"
    st.info('Loading ZeroScope model')
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    st.info("Model loaded")
    video = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames[0]

    export_to_video(video, output_video_path=fn)
    with open(fn, 'rb') as f:
        video_bytes = f.read()
        st.markdown('Generated video')
        st.video(video_bytes, format='video/mp4', start_time=0)
    return fn


def get_video(model, query):
    time.sleep(0.5)
    print("starting ... ")
    fn = get_video_from_model(query)
    print("done ... ")


def main():
    status = False if 'video_height' in st.session_state else True
    init(status)

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
        # st.subheader('Start Chat using Streamlit')
        video_height = st.sidebar.slider("Video Height", 300, 800, 400, 50, key="video_height")
        video_width = st.sidebar.slider("Video Height", 300, 800, 400, 50, key="video_width")
        video_length_s = st.sidebar.slider("Video Length in Sec", 5, 10, 20, 30, key="video_length_s")
        print(f"{video_height = }, {video_width = }, {video_length_s = }")
        st.divider()
        chat_name = st.selectbox("Chat Session:", ["default", "chat1"], key="chat_name", on_change=on_chat_change)
        chat_box.use_chat_name(chat_name)

        # streaming = st.checkbox('Streaming', key="streaming")
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

    btns.download_button("Export Markdown", "".join(chat_box.export2md()), file_name=f"chat_history.md", mime="text/markdown")

    btns.download_button("Export Json", chat_box.to_json(), file_name="chat_history.json", mime="text/json" )

    if btns.button("Clear History"):
        chat_box.init_session(clear=True)
        st.experimental_rerun()

    if show_history:
        st.write(st.session_state)

    picked_model = None if 'model' not in st.session_state else st.session_state['model']
    print(f"{picked_model = }")

    if query := st.chat_input('Input your question here'):
        st.session_state["expander"] = True
        query = query.strip()
        chat_box.user_say(query)
        # st.session_state.query = "query"
        if query.startswith(("generate", "Generate", "create", "show")):
            # show_model_flag = True
            chat_box.ai_say([
                Markdown("Pick a text for video generation ",
                         in_expander=in_expander, expanded=True, title="answer")
            ])
            picked_model = 'Tune-A-Video'
            # cols = st.columns(2)

            # def click_button(model, query):
            #     print("\tgot i = ", model, query)
            st.session_state.model = picked_model
            st.session_state.query = query

            # for i in range(2):
            #     # if cols[i].button(models[i], on_click=click_button(models[i], query))
            #     if cols[i].button(models[i], on_click=lambda m=models[i], q=query: click_button(m, q)):
            #         print("model")

        else:
            text, docs = llm.chat(query)
            chat_box.ai_say([
                Markdown(text, in_expander=in_expander, expanded=True, title="answer"),
            ])


    if picked_model is not None:
        query = None if 'query' not in st.session_state else st.session_state['query']

        chat_box.ai_say([
            Markdown(f"You have chosen the model = {picked_model}", in_expander=in_expander, expanded=True,
                     title="answer"),
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


def app(clear=True):
    main()
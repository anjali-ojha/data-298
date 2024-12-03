from datetime import datetime

from streamlit_chatbox import ChatBox, FakeLLM, Markdown
import os
import sys
import streamlit as st
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../code/Tune-A-Video/'))
print(os.path.join(os.path.dirname(__file__), '../code/Tune-A-Video/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../code/VideoCrafter'))
print(os.path.join(os.path.dirname(__file__), '../code/VideoCrafter/'))

from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils.export_utils import export_to_video
import time

from omegaconf import OmegaConf
from vc_utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init(expanded=True):
    print("init the page", expanded)
    container = st.container()
    container.title("Video-Crafter Model")
    expander = container.expander("Model Details and Architecture ", expanded=expanded)

    expander.write("""The novelity of VideoCrafter2 is that it introduces a novel method that combines low-quality video data with high-quality static images to generate high quality videos. Since the model uses LoRA, the pre-trained components of the model retain their strengths and aren't overwhelmed by the finetuning. VideoCrafter2 requires a significant computational resources to train. But, its ability to produce high-quality videos without high-quality video datasets makes it as an attractive solution for creating detailed and visually appealing videos in limited data environments.
    """)

    container.write("This is a demo of the Video-Crafter-2 model. You can generate a video from a text prompt. ")


def get_file_name(query):
    now = datetime.now()
    filename = now.strftime("%Y%m%d-%H%M%S")
    print(f"For {query = }, {filename}")
    return filename


def get_variants(query):
    variants = [query, query]
    return variants


def get_video_from_model(prompt):
    st.info('Loading Video-Crafter model')
    # https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt
    ckpt_path = "/data/checkpoints/VideoCrafter2/model.ckpt"
    ckpt_path = "/home/ubuntu/.cache/huggingface/checkpoints/VideoCrafter2/model.ckpt"

    config_file = '/home/ubuntu/code/VideoCrafter/configs/inference_t2v_512_v2.0.yaml'
    config = OmegaConf.load(config_file)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False

    model = instantiate_from_config(model_config)
    assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    model.cuda()
    st.info("Generating video")

    fps = 16
    batch_size = 1
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

    btns.download_button("Export Markdown", "".join(chat_box.export2md()), file_name=f"chat_history.md",
                         mime="text/markdown")

    btns.download_button("Export Json", chat_box.to_json(), file_name="chat_history.json", mime="text/json")

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
            picked_model = 'Video-Crafter'
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
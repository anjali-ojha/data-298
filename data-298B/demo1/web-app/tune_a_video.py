from datetime import datetime

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
    container.title("Tune-A-Video Model")
    expander = st.expander("Model Details and Architecture ", expanded=expanded)

    expander.write("""Tune-A-Video uses pretrained text-to-image (T2I) diffusion models like for video generation with less training overhead. Tune-A-Video uses a one-shot video tuning method where only a single text-video pair is required that making it superior to other methods that require a lot of training data. This reduces the computational burden significantly. The architecture is based on latent diffusion models adapted for video generation. This approach builds on the insight that T2I models can represent dynamic actions (verbs) in static images and that extending these models to manage multiple frames can preserve content consistency throughout the frames.The model uses a tailored spatio-temporal attention mechanism to generate motion. It also uses a strategic tuning method that updates only certain parameters, preserving the pre-trained model’s knowledge and focusing on achieving temporal coherence in the generated videos.
    """)
    expander.image("assets/images/tune-a-video.png", caption="Tune-A-Video Model System Diagram")

    container.write("This is a demo of the Tune-A-Video model. You can generate a video from a text prompt. ")


def get_variants(query):
    variants = [query, query]
    return variants


def get_file_name(query):
    now = datetime.now()
    filename = now.strftime("%Y%m%d-%H%M%S")
    print(f"For {query = }, {filename}")
    return filename


def get_video_from_model(prompt):
    st.info('Loading Tune-A-Video model')

    # Source: https://huggingface.co/CompVis/stable-diffusion-v1-4
    # pretrained_model_path = "/data/Tune-A-Video/checkpoints/stable-diffusion-v1-4"
    pretrained_model_path = "/home/ubuntu/stable-diffusion-v1-4"
    my_model_path = "/home/ubuntu/Tune-A-Video/outputs/rabbit-watermelon/"

    unet = UNet3DConditionModel.from_pretrained(
        my_model_path, subfolder='unet', torch_dtype=torch.float16
    ).to(device)

    pipe = TuneAVideoPipeline.from_pretrained(
        pretrained_model_path, unet=unet, torch_dtype=torch.float16
    ).to(device)
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


def get_video(model, query):
    time.sleep(0.5)
    print("starting ... ")
    fn = get_video_from_model(query)
    print("done ... ")


def main():
    print(f"{st.session_state = }")
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
import streamlit as st
from streamlit_chatbox import FakeLLM, ChatBox, Markdown
import streamlit_shadcn_ui as ui


def app(clear=True):

    with st.sidebar:
        # st.subheader('Start Chat using Streamlit')
        video_height = st.sidebar.slider("Video Height", 300, 800, 400, 50, key="video_height")
        video_width = st.sidebar.slider("Video Height", 300, 800, 400, 50, key="video_width")
        video_length_s = st.sidebar.slider("Video Length in Sec", 1, 30, 5, 1, key="video_length_s")
        print(f"{video_height = }, {video_width = }, {video_length_s = }")
        st.divider()
        # chat_name = st.selectbox("Chat Session:", ["default", "chat1"], key="chat_name", on_change=on_chat_change)
        # chat_box.use_chat_name(chat_name)

        # streaming = st.checkbox('Streaming', key="streaming")
        in_expander = st.checkbox('Show messages in expander', key="in_expander")
        show_history = st.checkbox('Show session state', key="show_history")
        # chat_box.context_from_session(exclude=["chat_name"])  # save widget values to chat context

        st.divider()

        btns = st.container()

    if "textarea1" not in st.session_state:
        textarea_value = ui.textarea(default_value="Input Prompt ..", placeholder="Enter longer text", key="textarea1")
    else:
        textarea_value = ui.textarea(default_value=st.session_state["textarea1"], placeholder="Enter longer text", key="textarea1")

    # st.write("Textarea Value:", textarea_value)
    if 'styled_btn_tailwind' not in st.session_state:
        gen_button = ui.button(text="Generate Video", key="styled_btn_tailwind", className="bg-orange-500 text-white")
    else:
        gen_button = st.session_state['styled_btn_tailwind']
        gen_button = True

    def slider_change(s):
        print(s)

    if gen_button:
        # with st.form("my_form"):
        #     slider_value = st.slider("My Slider", 0, 100, 50)
        #     st.form_submit_button("Submit")
        #
        # if "my_form" in st.session_state:
        #     st.write("Slider value:", st.session_state["my_form"]["slider_value"])

        # config_container = st.container(border=True)

        def test():
            st.text("test func called")

        with st.form("my_form", clear_on_submit=False,) :

            col1, col2, col3 = st.columns(3)

            height = col1.slider("Video Height", 300, 800, 400, 50, key="v_height", on_change=slider_change("v_height"))
            width = col2.slider("Video Height", 300, 800, 400, 50, key="v_width", on_change=slider_change("v_width"))
            length_s = col3.slider("Video Length in Sec", 1, 30, 5, 1, key="v_length_s", on_change=slider_change("v_length_s"))

            config_button = st.form_submit_button("Submit", on_click=test)

        if "my_form" in st.session_state:
            st.write("Slider value:", st.session_state["my_form"]["slider_value"])

        # if config_button:
        #     st.button("Submitted value")

        # with col1:
        #     height = ui.slider(default_value=[20], min_value=0, max_value=100, step=2, label="Video Height", key="v_height")

    if gen_button:

        ui.badges(badge_list=[(f"Video Height = {st.session_state['v_height']}", "default"),
                              (f"Video Width = {st.session_state['v_width']}", "secondary"),
                              # ("outline", "outline"),
                              # ("Hello", "destructive"),
                              (f"Video Length {st.session_state['v_length_s']}", "destructive")
                              ], class_name="flex gap-2", key="badges1")

def app2(clear=True):
    status = False if 'video_height' in st.session_state else True

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
        video_length_s = st.sidebar.slider("Video Length in Sec", 1, 30, 5, 1, key="video_length_s")
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
            # get_video(picked_model, query)
            st.session_state.query = None
            st.session_state.model = None
        except Exception as e:
            print("error")
            st.session_state.query = None
            st.session_state.model = None
            raise e
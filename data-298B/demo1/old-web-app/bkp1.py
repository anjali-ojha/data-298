import streamlit as st
from streamlit_chatbox import *
import time
import simplejson as json


models = ["Tune-A-Video Model", "CogVideo-5b Model", "Zero Scope Model"]
print("starting from begining")

llm = FakeLLM()

chat_box = ChatBox(
    use_rich_markdown=True, # use streamlit-markdown
    user_theme="green", # see streamlit_markdown.st_markdown for all available themes
    assistant_theme="blue",
)
chat_box.use_chat_name("chat1") # add a chat conversatoin

def on_chat_change():
    chat_box.use_chat_name(st.session_state["chat_name"])
    chat_box.context_to_session() # restore widget values to st.session_state when chat name changed


with st.sidebar:
    st.subheader('start to chat using streamlit')
    chat_name = st.selectbox("Chat Session:", ["default", "chat1"], key="chat_name", on_change=on_chat_change)
    chat_box.use_chat_name(chat_name)
    streaming = st.checkbox('streaming', key="streaming")
    in_expander = st.checkbox('show messages in expander', key="in_expander")
    show_history = st.checkbox('show session state', key="show_history")
    chat_box.context_from_session(exclude=["chat_name"]) # save widget values to chat context

    st.divider()

    btns = st.container()

    # file = st.file_uploader(
    #     "chat history json",
    #     type=["json"]
    # )

    # if st.button("Load Json") and file:
    #     data = json.load(file)
    #     chat_box.from_dict(data)


chat_box.init_session()
chat_box.output_messages()

def on_feedback(
    feedback,
    chat_history_id: str = "",
    history_index: int = -1,
):
    reason = feedback["text"]
    score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index) # convert emoji to integer
    # do something
    st.session_state["need_rerun"] = True


feedback_kwargs = {
    "feedback_type": "thumbs",
    "optional_text_label": "welcome to feedback",
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

if btns.button("clear history"):
    chat_box.init_session(clear=True)
    st.experimental_rerun()


if show_history:
    st.write(st.session_state)


picked = [True for _ in models]
picked_model = None

# def show_models(query):
#     print(query)
    
#     chat_box.ai_say(
#                 [
#                     Markdown("Pick a model", in_expander=in_expander,
#                             expanded=True, title="answer")
#                     # Markdown("\n\n".join(docs), in_expander=in_expander,
#                     #          title="references"),
#                 ]
#             )
    
    
#     for i, model in enumerate(models):
#         if cols[i].button(models[i]):  
#             picked[i] = True
#             chat_box.ai_say([
#                 Markdown(f"You have choose the model = {models[i]} ", in_expander=in_expander, expanded=True, title="answer"),
#                 Markdown(f" Generating Video for query = '{query}'", in_expander=in_expander, expanded=True, title="answer"),
#             ])
#             time.sleep(0.5)
#             # chat_box.ai_say([
#             #     Markdown(f" Generating Video for query = '{query}'", in_expander=in_expander, expanded=True, title="answer"),
#             # ])
#             with open("/Users/hims/Downloads/1066674784.mp4", 'rb') as f:
#                 video_bytes = f.read()
#                 st.video(video_bytes)
    

    # if cols[0].button('show me the multimedia ............. '):
    #     # chat_box.ai_say(
    #     #     Image('https://tse4-mm.cn.bing.net/th/id/OIP-C.cy76ifbr2oQPMEs2H82D-QHaEv?w=284&h=181&c=7&r=0&o=5&dpr=1.5&pid=1.7')
    #     #     )
    #     # st.image("https://tse4-mm.cn.bing.net/th/id/OIP-C.cy76ifbr2oQPMEs2H82D-QHaEv?w=284&h=181&c=7&r=0&o=5&dpr=1.5&pid=1.7", caption="Sample Image", use_column_width=True)
    #     # st.video("http://localhost:8000/Downloads/1066674784.mp4")
        
    #     time.sleep(0.5)
        
        
    #     # print(type(Video("http://localhost:8000/Downloads/1066674784.mp4")))
    #     with open("/Users/hims/Downloads/1066674784.mp4", 'rb') as f:
    #         video_bytes = f.read()
            
    #     chat_box.ai_say(
    #         Video(video_bytes)
    #         # Video("http://localhost:8000/Downloads/1066674784.mp4")
    #         # Video('https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4')
    #         )
    #     # time.sleep(0.5)
    #     # chat_box.ai_say(
    #     #     Audio('https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4'))

    # if cols[1].button('run agent by clicking this button ........'):
    #     chat_box.user_say('run agent')
    #     agent = FakeAgent()
    #     text = ""

    #     # streaming:
    #     chat_box.ai_say() # generate a blank placeholder to render messages
    #     for d in agent.run_stream():
    #         if d["type"] == "complete":
    #             chat_box.update_msg(expanded=False, state="complete")
    #             chat_box.insert_msg(d["llm_output"])
    #             break

    #         if d["status"] == 1:
    #             chat_box.update_msg(expanded=False, state="complete")
    #             text = ""
    #             chat_box.insert_msg(Markdown(text, title=d["text"], in_expander=True, expanded=True))
    #         elif d["status"] == 2:
    #             text += d["llm_output"]
    #             chat_box.update_msg(text, streaming=True)
    #         else:
    #             chat_box.update_msg(text, streaming=False)

# show_models("")

show_model_flag = False

if query := st.chat_input('input your question here'):

    chat_box.user_say(query)
    if query.startswith("generate") or query.startswith("Generate") or query.startswith("create") or query.startswith("show"):
        
        show_model_flag = True
        chat_box.ai_say([
            Markdown("Pick a model", in_expander=in_expander, expanded=True, title="answer")
            ])
        
        cols = st.columns(len(models))

        for i, model in enumerate(models):
            if show_model_flag and cols[i].button(models[i]): 
                picked_model = models[i]
                
                print("picking the model = ", models[i])
                # picked[i] = True
                chat_box.ai_say([
                    Markdown(f"You have choose the model {models[i]} ", in_expander=in_expander, expanded=True, title="answer"),
                    Markdown(f" Generating Video for query = '{query}'", in_expander=in_expander, expanded=True, title="answer"),
                ])
                print("added the markdown")
                time.sleep(0.5)
                # chat_box.ai_say([
                #     Markdown(f" Generating Video for query = '{query}'", in_expander=in_expander, expanded=True, title="answer"),
                # ])
                with open("/Users/hims/Downloads/1066674784.mp4", 'rb') as f:
                    video_bytes = f.read()
                    st.video(video_bytes)

                show_model_flag = False 
        
        
        
    
    else:    
        if streaming:
            generator = llm.chat_stream(query)
            elements = chat_box.ai_say(
                [
                    # you can use string for Markdown output if no other parameters provided
                    Markdown("thinking", in_expander=in_expander,
                            expanded=True, title="answer")
                    # Markdown("", in_expander=in_expander, title="references"),
                ]
            )
            time.sleep(1)
            text = ""
            for x, docs in generator:
                text += x
                chat_box.update_msg(text, element_index=0, streaming=True)
            # update the element without focus
            chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
            chat_box.update_msg("\n\n".join(docs), element_index=1, streaming=False, state="complete")
            chat_history_id = "some id"
            chat_box.show_feedback(**feedback_kwargs,
                                    key=chat_history_id,
                                    on_submit=on_feedback,
                                    kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
        else:
            text, docs = llm.chat(query)
            chat_box.ai_say(
                [
                    Markdown(text, in_expander=in_expander,expanded=True, title="answer"),
                    # Markdown("\n\n".join(docs), in_expander=in_expander,title="references"),
                ]
            )
        
# cols = st.columns(len(models))

# if show_model_flag:
    
#     chat_box.ai_say(
#             [
#                 Markdown("Pick a model", in_expander=in_expander,
#                         expanded=True, title="answer")
#                 # Markdown("\n\n".join(docs), in_expander=in_expander,
#                 #          title="references"),
#             ]
#         )
    
# print(picked, show_model_flag)
# i = 0
if any(picked) and picked_model is not None:
    chat_box.ai_say([
                    Markdown(f"You have choose the model {picked_model} ", in_expander=in_expander, expanded=True, title="answer"),
                    Markdown(f" Generating Video for query = '{query}'", in_expander=in_expander, expanded=True, title="answer"),
                ])

# cols = st.columns(len(models))
    

# for i, model in enumerate(models):
#     if show_model_flag and cols[i].button(models[i]): 
#         picked_model = models[i]
        
#         print("picking the model = ", models[i])
#         # picked[i] = True
#         chat_box.ai_say([
#             Markdown(f"You have choose the model {models[i]} ", in_expander=in_expander, expanded=True, title="answer"),
#             Markdown(f" Generating Video for query = '{query}'", in_expander=in_expander, expanded=True, title="answer"),
#         ])
#         print("added the markdown")
#         time.sleep(0.5)
#         # chat_box.ai_say([
#         #     Markdown(f" Generating Video for query = '{query}'", in_expander=in_expander, expanded=True, title="answer"),
#         # ])
#         with open("/Users/hims/Downloads/1066674784.mp4", 'rb') as f:
#             video_bytes = f.read()
#             st.video(video_bytes)

#         show_model_flag = False 
        
                

if cols[0].button('show me the multimedia ............. '):
        # chat_box.ai_say(
        #     Image('https://tse4-mm.cn.bing.net/th/id/OIP-C.cy76ifbr2oQPMEs2H82D-QHaEv?w=284&h=181&c=7&r=0&o=5&dpr=1.5&pid=1.7')
        #     )
        # st.image("https://tse4-mm.cn.bing.net/th/id/OIP-C.cy76ifbr2oQPMEs2H82D-QHaEv?w=284&h=181&c=7&r=0&o=5&dpr=1.5&pid=1.7", caption="Sample Image", use_column_width=True)
        # st.video("http://localhost:8000/Downloads/1066674784.mp4")
        print("picking the model = ")
        time.sleep(0.5)
        
        
        # print(type(Video("http://localhost:8000/Downloads/1066674784.mp4")))
        with open("/Users/hims/Downloads/1066674784.mp4", 'rb') as f:
            video_bytes = f.read()
        st.video(video_bytes)
        chat_box.ai_say(
            Video(video_bytes)
            # Video("http://localhost:8000/Downloads/1066674784.mp4")
            # Video('https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4')
        )
        

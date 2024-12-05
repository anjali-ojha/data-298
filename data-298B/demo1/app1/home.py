import streamlit as st
from firebase_admin import firestore
from streamlit_option_menu import option_menu
import streamlit_shadcn_ui as ui

from config import *

def app():
#     st.title('  :black[Text-To-Video Application] ')
#
#     st.write(""" Text-to-video generation is a rapidly advancing AI field that transforms written prompts into dynamic, coherent videos by merging natural language processing (NLP), computer vision, and deep learning. This technology has far-reaching applications in marketing, film, education, and virtual reality, prompting active development of models that generate high-quality, relevant, and cohesive videos from simple text inputs. At the core of text-to-video generation are multimodal neural networks, which process both linguistic and visual information using architectures like Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and transformers. GANs, originally developed for image synthesis, are adapted to video by enabling seamless frame generation, where a generator produces frames and a discriminator refines their realism. Diffusion models, specifically Denoising Diffusion Probabilistic Models (DDPMs), are pivotal in this field; they convert Gaussian noise into data representations through repeated denoising steps.
# """)
    st.session_state.clear()
    st.image(f"{assets_path}/images/home1.png")
    container = st.container(border=True)


    # st.button(text="Try Now", key="styled_btn_tailwind", className="bg-orange-500 text-white")
    st.button("Try Now")
    # st.image('assets/images/image_content.png')


    st.header("Sample Generated Videos for different Marketing use cases")

    container = st.container(border=True)

    container.write(
        ':black[Here are the some Sample Generated Videos using this application. As you can see that it covers a wide range of contexts and generates videos accordingly. Our focus is to generate videos that are relevant, coherent, and visually appealing. Wwe focused on the marketing related videos in this demo. As you can see that it covers various contexts and generates videos accordingly.]')
    col1, col2, col3 = container.columns(3)

    with col1:
        col1.video(f"{assets_path}/sample/dog_playing.mp4")
        st.write('  :black[Show a video of a dog playing in the grass in the sunny day wearing a adidas cap. Its a labrador dog and not too old. The sky is blue and grass is green which further highlights the dog. The dog is playing with a ball and running around.]')

    with col2:
        st.video(f"{assets_path}/sample/nike.mp4")
        st.write('  :black[A man running in a snowy day wearing track pants and nike hoodie. He running on the road, then suddenly a billboard comes up where it shows the nike log in black and white color.]')

    with col3:
        st.video(f"{assets_path}/sample/smart_watch.mp4")
        st.write('  :black[A cool smartwatch video highlighting its features. The video shows the smartwatch in different angles and lighting conditions. The watch is shown in a black color and the screen is shown in different colors.]')




    #
    # container = st.container(border=True)
    #
    # container.write(
    #     ':black[Here are the some Sample Generated Videos using this application. As you can see that it covers '
    #     'various contexts and generates videos accordingly.]')
    #
    # container.divider()
    #
    # col1, col2 = container.columns([1, 3])
    # col1.image('https://dp0rksi384o97.cloudfront.net/media/3590/responsive-images/ai-text-to-video-generator___webp_2420_1430.webp')
    # col2.write(''' First step to step to write what kind of video you want to generate. You can write a simple text like "A cat is playing" or "A dog is running". Add as much details as you can to get a better video.''')
    #
    # container.divider()
    #
    # col1, col2 = container.columns([3, 1])
    # col1.write(
    #     ''' First step to step to write what kind of video you want to generate. You can write a simple text like "A cat is playing" or "A dog is running". Add as much details as you can to get a better video.''')
    #
    # col2.image(
    #     'https://dp0rksi384o97.cloudfront.net/media/3590/responsive-images/ai-text-to-video-generator___webp_2420_1430.webp')
    #
    # container.divider()
    #
    # col1, col2 = container.columns([1, 3])
    # col1.image('assets/images/config.png')
    # col2.write(
    #     ''' Select all the configs suitable to your desired video like length, width and pixel etc. In future we can add the subtitiles background music etc. to make it more interactive. In explainer videos we can add the voice over as well. ''')




    # if 'db' not in st.session_state:
    #     st.session_state.db = ''
    #
    # db = firestore.client()
    # st.session_state.db = db
    # # st.title('  :violet[Text-To-Video]  :sunglasses:')
    #
    # ph = ''
    # if st.session_state.username == '':
    #     ph = 'Login to be able to post!!'
    # else:
    #     ph = 'Post your thought'
    # post = st.text_area(label=' :orange[+ New Post]', placeholder=ph, height=None, max_chars=500)
    # if st.button('Post', use_container_width=20):
    #     if post != '':
    #
    #         info = db.collection('Posts').document(st.session_state.username).get()
    #         if info.exists:
    #             info = info.to_dict()
    #             if 'Content' in info.keys():
    #
    #                 pos = db.collection('Posts').document(st.session_state.username)
    #                 pos.update({u'Content': firestore.ArrayUnion([u'{}'.format(post)])})
    #                 # st.write('Post uploaded!!')
    #             else:
    #
    #                 data = {"Content": [post], 'Username': st.session_state.username}
    #                 db.collection('Posts').document(st.session_state.username).set(data)
    #         else:
    #
    #             data = {"Content": [post], 'Username': st.session_state.username}
    #             db.collection('Posts').document(st.session_state.username).set(data)
    #
    #         st.success('Post uploaded!!')
    #
    # st.header(' :violet[Latest Posts] ')
    #
    # docs = db.collection('Posts').get()
    #
    # for doc in docs:
    #     d = doc.to_dict()
    #     try:
    #         st.markdown("""
    #             <style>
    #
    #             .stTextArea [data-baseweb=base-input] [disabled=""]{
    #                 # background-color: #e3d8c8;
    #                 -webkit-text-fill-color: white;
    #             }
    #             # </style>
    #             """, unsafe_allow_html=True)
    #
    #         st.text_area(label=':green[Posted by:] ' + ':orange[{}]'.format(d['Username']), value=d['Content'][-1],
    #                      height=20, disabled=True)
    #
    #         # st.text_area(label=':green[Posted by:] '+':orange[{}]'.format(d['Username']),value=d['Content'][-1],height=20)
    #     except:
    #         pass

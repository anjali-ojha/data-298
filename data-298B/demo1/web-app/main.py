import pathlib

import streamlit as st

from streamlit_option_menu import option_menu
import os

# from dotenv import load_dotenv

# load_dotenv()

import home
# import account, about, tune_a_video, edit_pix2pix, video_crafter, zero_scope, t2v_zero, train_tune_a_video, cog_video_5b


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.set_page_config(
    page_title="Text-To-Video",
    layout='wide',
)

st.markdown(
    """
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src=f"https://www.googletagmanager.com/gtag/js?id={os.getenv('analytics_tag')}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', os.getenv('analytics_tag'));
            [data-testid="stAppViewContainer"] {
                background-color: #e5e5f7;
                opacity: 0.8;
                background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #e5e5f7 10px ), repeating-linear-gradient( #444cf755, #444cf7 );
            }
            
            [data-testid="stHeader"] {
                background-color: rgba(0, 0, 0, 0);
                color: white;   
            }
            
            [data-testid="stToolbar"] {
                background-color: #000000;
                color: white;
                right: 2em;
            }
            
            [data-testid="stSidebar"] {
                backGround-image: url('https://www.transparenttextures.com/patterns/brushed-alum.png');
            }
        </script>
    """, unsafe_allow_html=True)
print(os.getenv('analytics_tag'))


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        load_css(pathlib.Path('assets/style.css'))
        # app = st.sidebar(
        with st.sidebar:
            # st.sidebar.image('https://i.gifer.com/embedded/download/VYY.gif')
            st.sidebar.image('assets/images/logo.png')
            app = option_menu(
                menu_title='Text-To-Video ',
                options=['Home', 'Tune-A-Video', 'CogVideoX-5b', 'ZeroScope', 'Text-To-Vide Zero', 'Video-Crafter', 'Edit-Pix-2-Pix', 'About', 'Account'],
                icons=['house-fill', 'person-circle', 'trophy-fill', 'chat-fill', 'info-circle-fill'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": 'black'},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px",
                                 "--hover-color": "blue"},
                    "nav-link-selected": {"background-color": "#02ab21"}, }

            )

        if app == "Home":
            home.app()
        # if app == "Tune-A-Video":
        #     st.session_state.clear()
        #     tune_a_video.app()
        # if app == "CogVideoX-5b":
        #     st.session_state.clear()
        #     cog_video_5b.app()
        # if app == "ZeroScope":
        #     st.session_state.clear()
        #     zero_scope.app()
        # if app == "Text-To-Vide Zero":
        #     st.session_state.clear()
        #     t2v_zero.app()
        # if app == "Video-Crafter":
        #     st.session_state.clear()
        #     video_crafter.app()
        # if app == "Edit-Pix-2-Pix":
        #     st.session_state.clear()
        #     edit_pix2pix.app()
        # if app == "Train Model using A video":
        #     st.session_state.clear()
        #     train_tune_a_video.app()
        #
        # if app == 'About':
        #     about.app()
        # if app == "Account":
        #     st.session_state.clear()
        #     account.app()


if __name__ == "__main__":
    runner = MultiApp()
    runner.run()

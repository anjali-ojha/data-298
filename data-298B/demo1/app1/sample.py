import streamlit as st

from config import *


def app():
    st.header("Sample Generated for different Marketing use cases")
    st.text("")

    container = st.container(border=True)
    container.write(
        ':black[Here are the some Sample Generated Videos using this application. As you can see that it covers a wide range of contexts and generates videos accordingly. Our focus is to generate videos that are relevant, coherent, and visually appealing. Wwe focused on the marketing related videos in this demo. As you can see that it covers various contexts and generates videos accordingly.]')

    container1 = st.container(border=True)
    container1_col1, container1_col2 = container1.columns([2,3])
    container1_col1.video(f"{assets_path}/sample/dog_playing.mp4")
    container1_col2.write(
            '  :black[Show a video of a dog playing in the grass in the sunny day wearing a adidas cap. Its a labrador dog and not too old. The sky is blue and grass is green which further highlights the dog. The dog is playing with a ball and running around.]')

    container2 = st.container(border=True)
    container2_col1, container2_col2 = container2.columns([2, 3])
    container2_col1.video(f"{assets_path}/sample/nike.mp4")
    container2_col2.write(
        '  :black[A man running in a snowy day wearing track pants and nike hoodie. He running on the road, then suddenly a billboard comes up where it shows the nike log in black and white color.]')

    container3 = st.container(border=True)
    container3_col1, container3_col2 = container3.columns([2, 3])
    container3_col1.video(f"{assets_path}/sample/smart_watch.mp4")
    container3_col2.write(
        '  :black[A cool smartwatch video highlighting its features. The video shows the smartwatch in different angles and lighting conditions. The watch is shown in a black color and the screen is shown in different colors.]')



    # container.write(
    #     ':black[Here are the some Sample Generated Videos using this application. As you can see that it covers a wide range of contexts and generates videos accordingly. Our focus is to generate videos that are relevant, coherent, and visually appealing. Wwe focused on the marketing related videos in this demo. As you can see that it covers various contexts and generates videos accordingly.]')
    # col1, col2, col3 = container.columns(3)
    #
    # with col1:
    #     col1.video(f"{assets_path}/sample/dog_playing.mp4")
    #     st.write(
    #         '  :black[Show a video of a dog playing in the grass in the sunny day wearing a adidas cap. Its a labrador dog and not too old. The sky is blue and grass is green which further highlights the dog. The dog is playing with a ball and running around.]')
    #
    # with col2:
    #     st.video(f"{assets_path}/sample/nike.mp4")
    #     st.write(
    #         '  :black[A man running in a snowy day wearing track pants and nike hoodie. He running on the road, then suddenly a billboard comes up where it shows the nike log in black and white color.]')
    #
    # with col3:
    #     st.video(f"{assets_path}/sample/smart_watch.mp4")
    #     st.write(
    #         '  :black[A cool smartwatch video highlighting its features. The video shows the smartwatch in different angles and lighting conditions. The watch is shown in a black color and the screen is shown in different colors.]')


import streamlit as st
from gait_analysis import GaitAnalysis
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType, initialize_agent, load_tools
import os
import uuid
from PIL import Image

# Display UI using Streamlit
class StreamlitApp:
    def __init__(self):
        st.set_page_config(
        page_title="Gait Analyzer",
        page_icon="./images/logo.png")
        st.title("Gait Analyzer")
        # Set the sidebar for navigation
        st.sidebar.title("Gait Analyzer")
        st.sidebar.markdown("Run Gait Analyzer on your computer using [Docker](https://hub.docker.com/r/abishekmuthian/gaitanalyzer).")
        st.sidebar.write("Built by Abishek Muthian.")
        st.sidebar.markdown("---") 
        
        image = Image.open("./images/logo.png")
        st.caption("Analyze your gait for health disorders at the comfort of your home in your own personal computer.")
        st.image(image)
        st.header("Video Upload")
        uploaded_file = st.file_uploader("Choose a short video of you moving from left to right (or) right to left covering your entire body", type="mp4")
        
        if uploaded_file is not None:
            input_directory = "input_videos"
            if not os.path.exists(input_directory):
                os.makedirs(input_directory)
            input_video_filename = uuid.uuid4().hex+".mp4"    
            with open(os.path.join("input_videos",input_video_filename),"wb") as f: 
                f.write(uploaded_file.getbuffer())    
            gait_analysis = GaitAnalysis(os.path.join("input_videos",input_video_filename))
            output_video, df, result, plt = gait_analysis.process_video()
            
            st.header("Annotated video:")
            video_file = open(output_video, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes, format="video/webm", start_time=0)

            st.header("Plotting the Distances, Peaks and Minima")

            st.subheader("Left Leg: ")
            st.pyplot(plt.figure(1), clear_figure=True)

            st.subheader("Right Leg: ")
            st.pyplot(plt.figure(2), clear_figure=True)

            st.header("Gait Data:")
            st.dataframe(df)

            csv = self.convert_df(df)

            st.download_button(
            "Download Gait Data",
            csv,
            "gait_analysis.csv",
            "text/csv",
            key='download-csv'
            )

            st.header("Your gait pattern explanation:")

            prompt = "This is my gait data, explain the contents of this gait data and explain my gait pattern from the given gait data - "+result
            self.run_model(prompt)


    # Download the dataframe as a .csv file    
    @staticmethod
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')   

    # Run llama2 model via ollama and display the response on screen
    @staticmethod
    def run_model(prompt):
        output_container = st.empty()
        output_container = output_container.container()
        answer_container = output_container.chat_message("assistant", avatar="🤖")
        st_callback = StreamlitCallbackHandler(answer_container)
        llm = Ollama(model="llama2", 
        callback_manager = CallbackManager([st_callback]))
        try:
            llm(prompt)
        except:
            st.error("Cannot access ollama service, restart the webpage and try again.")

if __name__ == "__main__":
    app = StreamlitApp()
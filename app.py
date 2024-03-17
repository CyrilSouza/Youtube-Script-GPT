# Bring in deps
import os 
from fpdf import FPDF
from io import BytesIO
import streamlit as st 
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper



api_key=os.getenv("OPEN_API_KEY")
# App framework
st.title('🔗 YouTube Script GPT ')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}.'
)


script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a YouTube video script based on the topic: {title} while leveraging this wikipedia research:{wikipedia_research} to fetch relevant data and provide a comprehensive analysis. Ensure the script is detailed and complete, covering all key aspects related to the topic.'
)


# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9,max_tokens=1500) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)  

    def download_script_pdf(script):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="YouTube Video Script", ln=True, align="C")
        pdf.cell(200, 10, txt="", ln=True, align="C")
        pdf.multi_cell(0, 10, txt=script)

        # Create a BytesIO object to store the PDF data
        pdf_output = BytesIO()

        # Write the PDF data to the BytesIO object
        pdf_output.write(pdf.output(dest='S').encode('latin-1', 'replace'))

        # Seek to the beginning of the BytesIO object
        pdf_output.seek(0)

        # Get the bytes data from the BytesIO object
        pdf_bytes = pdf_output.read()

        return pdf_bytes


    # Define the on-click function for the download button
    def on_download_button_click(script):
        pdf_bytes = download_script_pdf(script)
        st.download_button(
            label="Download Script as PDF",
            data=pdf_bytes,
            file_name="script.pdf",
            mime="application/pdf"
        )

    # Call the on-click function with the script as an argument
    on_download_button_click(script)

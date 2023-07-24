import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras import add_vertical_space as avs
from langchain.callbacks import get_openai_callback
import base64
import os


with st.sidebar:
    st.title('🕮 Bible Study App')
    st.markdown('''
    ## About
    This app is a LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                

 
    ''')
    avs.add_vertical_space(5)
    st.write('Made with ❤️ by [Johan](https://github.com/TripleJ160)')


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def show_content(book):
    folder_path = "./Books/Bible/NewTestament"
    if book == "Matthew":
        pdf_file = f"{folder_path}/MatthewOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Mark":
        pdf_file = f"{folder_path}/MarkOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Luke":
        pdf_file = f"{folder_path}/LukeOEV.pdf"
        show_pdf(pdf_file)
    elif book == "John":
        pdf_file = f"{folder_path}/JohnOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Acts":
        pdf_file = f"{folder_path}/ActsOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Romans":
        pdf_file = f"{folder_path}/RomansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "1 Corinthians":
        pdf_file = f"{folder_path}/1 CorinthiansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "2 Corinthians":
        pdf_file = f"{folder_path}/2 CorinthiansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Galatians":
        pdf_file = f"{folder_path}/GalatiansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Ephesians":
        pdf_file = f"{folder_path}/EphesiansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Philippians":
        pdf_file = f"{folder_path}/PhilippiansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Colossians":
        pdf_file = f"{folder_path}/ColossiansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "1 Thessalonians":
        pdf_file = f"{folder_path}/1 ThessaloniansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "2 Thessalonians":
        pdf_file = f"{folder_path}/2 ThessaloniansOEV.pdf"
        show_pdf(pdf_file)
    elif book == "1 Timothy":
        pdf_file = f"{folder_path}/1 TimothyOEV.pdf"
        show_pdf(pdf_file)
    elif book == "2 Timothy":
        pdf_file = f"{folder_path}/2 TimothyOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Titus":
        pdf_file = f"{folder_path}/TitusOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Philemon":
        pdf_file = f"{folder_path}/PhilemonOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Hebrews":
        pdf_file = f"{folder_path}/HebrewsOEV.pdf"
        show_pdf(pdf_file)
    elif book == "James":
        pdf_file = f"{folder_path}/JamesOEV.pdf"
        show_pdf(pdf_file)
    elif book == "1 Peter":
        pdf_file = f"{folder_path}/1 PeterOEV.pdf"
        show_pdf(pdf_file)
    elif book == "2 Peter":
        pdf_file = f"{folder_path}/2 PeterOEV.pdf"
        show_pdf(pdf_file)
    elif book == "1 John":
        pdf_file = f"{folder_path}/1 JohnOEV.pdf"
        show_pdf(pdf_file)
    elif book == "2 John":
        pdf_file = f"{folder_path}/2 JohnOEV.pdf"
        show_pdf(pdf_file)
    elif book == "3 John":
        pdf_file = f"{folder_path}/3 JohnOEV.pdf"
        show_pdf(pdf_file)
    elif book == "Jude":
        pdf_file = f"{folder_path}/JudeOEV.pdf"
        show_pdf(pdf_file)

load_dotenv()
def main():
    st.sidebar.title("Select a Book")
    selected_book = st.sidebar.selectbox("Choose a Book", [
        "Matthew", "Mark", "Luke", "John", "Acts",
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy",
        "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
        "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
        "Jude", "Revelation"
    ])

    st.header(f"Content for {selected_book}")
    show_content(selected_book)
    

    #pdf = st.file_uploader("Upload your PDF!", type='pdf')
    #show_pdf('MatthewFBV.pdf')
    #pdf = 'MatthewFBV.pdf'
    #pdf_reader = PdfReader('MatthewFBV.pdf')

    folder_path = "./Books/Bible/NewTestament"
    pdf = f"{folder_path}/{selected_book}OEV.pdf"
    print(pdf)
    pdf_reader = PdfReader(pdf)
    text = " "

    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
        )
    chunks = text_splitter.split_text(text=text)

    #embedding
    
    store_name = pdf[:-4]
    #st.write(store_name)
    
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        st.write('Embeddings Loaded from the Disk')
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
        st.write('Embeddings Operation Complete')

    # Input User Questions
    query = st.text_input("Ask questions about the selected book below: ")

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm = llm, chain_type = "stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question = query)
            print(cb)
        st.write(response)



    #st.write(chunks)
    #st.write(text)

if __name__ == '__main__':
    main()
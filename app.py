import os
import tempfile

import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Replicate, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain


def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'human_messages' not in st.session_state:
        st.session_state['human_messages'] = ['Hi!']  # Initial user message

    if 'ai_messages' not in st.session_state:
        st.session_state['ai_messages'] = ['Hello! Ask me anything.']  # Initial assistant response


def display_settings():
    st.sidebar.header('Settings')

    name = st.sidebar.text_input('Name', 'My Assistant')

    tone = st.sidebar.selectbox('Tone', ['Formal', 'Informal'], index=0)

    assistant = st.sidebar.text_area('Assistant', placeholder='Description of assistant. (e.g., You are a helpful assistant.)')

    user = st.sidebar.text_area('User', placeholder='Description of user. (e.g., I am a non-technical person.)')

    return name, tone, assistant, user


def display_api_configuration():
    st.sidebar.header('API Configuration')

    llm_model= st.sidebar.selectbox('LLM model for generating responses', ['text-davinci-003 on OpenAI', 'llama-2-70b-chat on Replicate'], index=0)

    embedding_model = st.sidebar.selectbox('Embedding model for making documents searchable', ['text-embedding-ada-002 on OpenAI', 'all-MiniLM-L6-v2 on Hugging Face (Free)'], index=0)

    is_openai_api_key_required = 'OpenAI' in llm_model or 'OpenAI' in embedding_model
    is_replicate_api_token_required = 'Replicate' in llm_model or 'Replicate' in embedding_model

    tab1, tab2 = st.sidebar.tabs(['Enter OpenAI API Key', 'Get Help'])
    with tab1:
        openai_api_key = st.text_input('OpenAI API key', placeholder='Required' if is_openai_api_key_required else 'Not required for this configuration', type='password')
        os.environ['OPENAI_API_KEY'] = openai_api_key

    with tab2:
        st.info('Get your key from https://platform.openai.com/account/api-keys.')


    tab1, tab2 = st.sidebar.tabs(['Enter Replicate API Token', 'Get Help'])
    with tab1:
        replicate_api_token = st.text_input('Replicate API token', placeholder='Required' if is_replicate_api_token_required else 'Not required for this configuration', type='password')
        os.environ['REPLICATE_API_TOKEN'] = replicate_api_token

    with tab2:
        st.info('Get your token from https://replicate.com/account/api-tokens.')

    return llm_model, embedding_model, is_openai_api_key_required, is_replicate_api_token_required, openai_api_key, replicate_api_token


def display_file_uploader(is_openai_api_key_required, is_replicate_api_token_required, openai_api_key, replicate_api_token):
    if is_openai_api_key_required and not openai_api_key:
        st.error('Please enter your OpenAI API key.')
        return
    
    if is_replicate_api_token_required and not replicate_api_token:
        st.error('Please enter your Replicate API token.')
        return

    uploaded_files = st.file_uploader('Upload documents to start conversation.', accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'doc'])
    return uploaded_files


@st.cache_data
def create_vector_store(uploaded_files, embedding_model):
    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == '.pdf':
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == '.docx' or file_extension == '.doc':
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == '.txt':
                loader = TextLoader(temp_file_path)
            
            if loader:
                current_text = loader.load()
                text.extend(current_text)
            
            os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        if embedding_model == 'text-embedding-ada-002 on OpenAI':
            embeddings = OpenAIEmbeddings(model_name='text-embedding-ada-002')

        elif embedding_model == 'all-MiniLM-L6-v2 on Hugging Face (Free)':
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        
        else:
            assert False
        
        # all-MiniLM-L6-v2: 'By default, input text longer than 256 word pieces is truncated.'

        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        return vector_store


@st.cache_data
def create_llm(llm_model):
    if llm_model == 'text-davinci-003 on OpenAI':
        llm = OpenAI(model_name='text-davinci-003', temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)
    
    elif llm_model == 'llama-2-70b-chat on Replicate':
        llm = Replicate(model='replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781',
                        input={'temperature': 0.1}, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], verbose=True)
        # See https://replicate.com/meta/llama-2-70b-chat/api for "input" options and defaults.

    else:
        assert False

    return llm


def create_conversational_chain(llm, vector_store):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 2})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff', retriever=retriever, memory=memory)
    return chain


def display_chat(chain):
    chat_history_container = st.container()
    question_container = st.container()

    with question_container:
        with st.form(key='my_form', clear_on_submit=True):
            question = st.text_input('Question:', placeholder='Ask about your documents.', key='input')
            submit= st.form_submit_button(label='Send')

        if submit and question:
            with st.spinner('Generating response...'):
                # question = query = user_input = human_message
                # answer = response = model_output = ai_message
                answer = chain({'question': question, 'chat_history': st.session_state['chat_history']})['answer']  # This is where we call the model.
                st.session_state['chat_history'].append((question, answer))

            st.session_state['human_messages'].append(question)
            st.session_state['ai_messages'].append(answer)

    with chat_history_container:
        for human_message, ai_message in zip(st.session_state['human_messages'], st.session_state['ai_messages']):
            message(human_message, is_user=True, avatar_style='thumbs')
            message(ai_message, avatar_style='bottts-neutral', seed='Buster')  # Shadow is also good.
            # See "Dice Bear" for avatar styles and seeds.


def main():
    initialize_session_state()

    name, tone, assistant, user = display_settings()
    llm_model, embedding_model, is_openai_api_key_required, is_replicate_api_token_required, openai_api_key, replicate_api_token = display_api_configuration()
    
    st.title(name)

    uploaded_files = display_file_uploader(is_openai_api_key_required, is_replicate_api_token_required, openai_api_key, replicate_api_token)

    if not uploaded_files:
        return
    
    with st.spinner('Creating vector store...'):
        vector_store = create_vector_store(uploaded_files, embedding_model)

    with st.spinner('Creating LLM...'):
        llm = create_llm(llm_model)
    
    with st.spinner('Creating conversational chain...'):
        chain = create_conversational_chain(llm, vector_store)
            
    display_chat(chain)


if __name__ == '__main__':
    main()

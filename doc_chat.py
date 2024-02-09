import streamlit as st
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
import langchain_core
import tempfile


# loader = WebBaseLoader(
#     web_paths=(doc_path,),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
if 'doc_path' not in st.session_state:
    st.session_state.doc_path=''
with st.sidebar:
    # doc_path = "Documentation.pdf"

    # loader = WebBaseLoader(
    #     web_paths=(doc_path,),
    #     bs_kwargs=dict(
    #         parse_only=bs4.SoupStrainer(
    #             class_=("post-content", "post-title", "post-header")
    #         )
    #     ),
    # )
    if st.button('Store History'):
        store_history = []
        for msg in st.session_state.chat_history:
            if isinstance(msg, langchain_core.messages.human.HumanMessage):
                store_history.append('Q. '+msg.content)
            elif isinstance(msg, langchain_core.messages.ai.AIMessage):
                store_history.append(msg.content+'\n')

            
        with open('chat_history.txt','w') as file:
            file.write('\n'.join(store_history))
    
    uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False,type=['pdf'])
    
    if st.button("Load Doc") and uploaded_file is not None:
        # docs = loader.load()
        st.session_state.doc_path = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        # Retrieve and generate using the relevant snippets of the blog.
        st.session_state.retriever = vectorstore.as_retriever()



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def context_loader():
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context from Chinese language in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    return contextualize_q_chain

def sys_prompt():
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    The context maybe in Chinese Language, and you need to understand the context and respond in English. \
    If you don't know the answer, just say that you don't know. \
    Provide a detailed answer.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    return qa_prompt

def contextualized_question(input: dict):
    res = context_loader()
    if input.get("chat_history"):
        return res
    else:
        return input["question"]



if 'chat_history' not in st.session_state:
    asst = f"Hi, I am a doc_bot. Ask me any question about the document {st.session_state.doc_path}."#
    # st.chat_message("assistant").write(asst)
    st.session_state['chat_history'] =  [AIMessage(asst)]

for msg in st.session_state.chat_history: 
    if isinstance(msg, langchain_core.messages.human.HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, langchain_core.messages.ai.AIMessage):
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input(placeholder="Ask a question about the document"):
    st.chat_message("user").write(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    qa_prompt = sys_prompt()
    
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | st.session_state.retriever | format_docs
        )
        | qa_prompt
        | llm
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        ai_msg = rag_chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append(ai_msg)
        st.write(ai_msg.content)
        

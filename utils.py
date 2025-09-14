from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings



def qa_agent(qianwen_api_key, memory, uploaded_file,question):
    model = ChatTongyi(model="qwen-plus",dashscope_api_key=qianwen_api_key)

    #读取用户上传文档,二进制
    file_content = uploaded_file.read()
    #用于储存pdf数据的临时文件路径
    temp_file_path = "temp.pdf"
    #把读取的内容进行写入
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)

    loader = PyPDFLoader(temp_file_path)
    #得到加载器的文档列表
    docs = loader.load()
    #分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "!", "?", "，", "、",""]
    )
    texts = text_splitter.split_documents(docs)
    #向量嵌入并导入数据库,并得到检索器
    embeddings_model = DashScopeEmbeddings(model="text-embedding-v1",dashscope_api_key=qianwen_api_key)
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
     
    #创建带记忆的检索增强对话链
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"question":question})
    return response





import streamlit as st
from utils import qa_agent
from langchain.memory import ConversationBufferMemory

st.title("AI智能PDF问答工具")

#侧边栏
with st.sidebar:
    qianwen_api_key=st.text_input("请输入通义千问 API密钥：",type="password")
    st.markdown("[获取通义千问 Key](https://dashscope.console.aliyun.com/apiKey)")


#记忆初始化,不能每次都初始化
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,         #储存的是消息列表而不是字符串
        memory_key="chat_history",    #后端用到的ConversationalRetrievalChain里记忆对应的键是chat_history
        output_key="answer"
    )

#上传文件的组件
uploaded_file = st.file_uploader("上传你的PDF文件：",type="pdf")
#输入问题的输入框,还没输入文件时禁止提问
question = st.text_input("对PDF的内容进行提问",disabled=not uploaded_file)

if uploaded_file and question and not qianwen_api_key:
    st.info("请输入你的DashScope API密钥")

if uploaded_file and question and qianwen_api_key:
    with st.spinner("AI正在思考中，请稍等..."):    #加载组件
        response = qa_agent(qianwen_api_key,st.session_state["memory"],
                            uploaded_file,question)
        st.write("###答案")
        st.write(response["answer"])

        #把历史对话储存在对话状态中
        st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):     #可展开/收起的区域
        for i in range(0,len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i<len(st.session_state["chat_history"])-2:
                st.divider()         #每对消息之间用分割线隔开

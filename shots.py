from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large')
database = Chroma(collection_name='chroma-food', persist_directory="./chroma", embedding_function=embedding)
db = database.similarity_search("식당이름 :, 음식 종류 :, 리뷰: ", k=100)

condition = f"""
        answer 할 때 하기 단계별로 시행해 주세요.
        1.input에서 음식 종류를 파악해 주세요.
        2.{db}에서 유사도가 가장 높은 음식 종류 - 식당이름 - 리뷰 쌍으로 구분해주세요.
        3.구분한 쌍에서 input에서 요청한 사항과 가장 알맞은 해답을 리뷰를 통해서 출력해주세요.
        추가적으로 리뷰를 간단하게 요약해서 뒤에 붙여주세요.
        """
answer_examples = [
    {

        "input" : condition+"따뜻한 국물 요리를 추천해주세요",
        "output" : "따뜻한 국물 요리로는 부대찌개가 있으며 경원식당 의정부를 추천합니다. 최근 리뷰 : 일행이 부대찌개 좋아해서 일부러 찿아갔어요, 부대찌개에 김치 고기 파 햄 소세지 당면이 얼근한 양념과 보글보글 끓여서 라면과 당면 그리고 햄과 소세지 추가 했어요"
        
    },

    {
        "input" : condition+"재료도 신선하고 국물도 맛있는 오래된 부대찌개 집을 추천해주세요",
        "output" : "오뎅식당 의정부를 추천합니다. 최근 리뷰: 초반에는 맛있는 김치찌개의 맛, 끓이면서 햄의 맛이 스며들어 점점 더 맛있어집니다"
    }
]
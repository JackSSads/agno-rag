from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.knowledge import Knowledge

from os import getenv
from dotenv import load_dotenv

class AgnoRAGAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = getenv("OPENAI_API_KEY")

        self.vector_db = self._create_vector_db()
        self.knowledge = self._create_knowledge()
        self.db = self._create_database()
        self.agent = self._create_agent()

    def _create_vector_db(self):
        return ChromaDb(
            collection="agno_agent_collection",
            path="temp/chromadb",
            persistent_client=True
        )

    def _create_knowledge(self):
        knowledge = Knowledge(vector_db=self.vector_db)

        knowledge.add_content(
            url="http://s3.sa-east-1.amazonaws.com/static.grendene.aatb.com.br/releases/2417_2T25.pdf",
            metadata={"source": "manual do produto"},
            skip_if_exists=True,
        )

        return knowledge

    def _create_database(self):
        return SqliteDb(db_file="temp/db_agno_agent.db")

    def _create_agent(self):
        return Agent(
            name="multiagente_rag",
            model=OpenAIChat(id="gpt-5-nano", api_key=self.api_key),
            instructions="""
                Você é um analista.
                Sempre que o usuário informar preferências (formato, nível de detalhe),
                salve essas informações na memória do usuário.
            """,
            db=self.db,
            add_history_to_context=True,
            num_history_runs=3,
            enable_user_memories=True,
            enable_agentic_memory=True,
            knowledge=self.knowledge,
            add_memories_to_context=True,
        )

    def to_ask(self, ask: str):
        response = self.agent.run(ask)
        return response.content if hasattr(response, "content") else response


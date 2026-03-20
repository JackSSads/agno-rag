from fastapi import FastAPI
from uvicorn import run

from agent import AgnoRAGAgent

app = FastAPI(
    title="Agents API",
    description="API for Agents",
    version="1.0.0",
    contact={
        "name": "Jackson Souza",
        "email": "contatojacksonsouza@hotmail.com",
        "url": "https://www.jacksonsouza-devzone.site/"
    },
)

@app.post("/ask")
def ask_question(ask: str):
    agno = AgnoRAGAgent()
    return agno.to_ask(ask)

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)

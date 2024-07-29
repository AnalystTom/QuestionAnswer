import streamlit as st
import openai
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Configuration for AI71 API
AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = "ai71-api-6fdf9128-ea22-4c89-98e6-d59547abe24f"

# Initialize OpenAI client with AI71 configuration
client = openai.OpenAI(
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
)

class Message(TypedDict):
    role: str
    content: str

class AgentState(TypedDict):
    messages: Annotated[List[Message], operator.add]

memory = SqliteSaver.from_conn_string(":memory:")

class Agent:
    def __init__(self, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_ai71)
        graph.set_entry_point("llm")
        graph.add_edge("llm", END)
        self.graph = graph.compile(checkpointer=memory)

    def call_ai71(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [{"role": "system", "content": self.system}] + messages
        response = client.chat.completions.create(
            model="tiiuae/falcon-180b-chat",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        message = response.choices[0].message
        return {'messages': [{"role": message.role, "content": message.content}]}

# System prompt
prompt = """You are a smart interview assistant. Your task is to generate and ask interview questions based on the given job title and question type.
Generate thoughtful and relevant questions without using external search capabilities."""

# Initialize Agent
abot = Agent(system=prompt)

def generate_questions(job_title: str, question_type: str) -> List[str]:
    messages = [{"role": "user", "content": f"Generate 10 interview questions for the job title: {job_title}. Question type: {question_type}"}]
    thread = {"configurable": {"thread_id": job_title}}
    questions = []
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            if v['messages'][0]["role"] == "assistant":
                questions = v['messages'][0]["content"].split('\n')
    return questions

def generate_similar_question(question: str) -> str:
    messages = [{"role": "user", "content": f"Generate a semantically similar question to: '{question}'"}]
    thread = {"configurable": {"thread_id": "similar_question"}}
    similar_question = ""
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            if v['messages'][0]["role"] == "assistant":
                similar_question = v['messages'][0]["content"]
    return similar_question

# Streamlit app
st.title("AI71-Powered Interactive Interview Question Generator")

job_title = st.text_input("Enter the desired job title:")
question_type = st.selectbox("Select question type:", ["Generic", "Job-Specific", "Mixed"])

if 'questions' not in st.session_state:
    st.session_state.questions = []
    st.session_state.current_question_index = 0
    st.session_state.user_answers = []

if st.button("Generate Questions"):
    if job_title:
        st.session_state.questions = generate_questions(job_title, question_type)
        st.session_state.current_question_index = 0
        st.session_state.user_answers = []
        st.experimental_rerun()
    else:
        st.write("Please enter a job title")

if st.session_state.questions:
    st.subheader("Interview Session")
    
    current_question = st.session_state.questions[st.session_state.current_question_index]
    st.write(f"Question: {current_question}")
    
    user_answer = st.text_area("Your Answer:", key=f"answer_{st.session_state.current_question_index}")
    
    if st.button("Submit Answer"):
        st.session_state.user_answers.append(user_answer)
        
        if st.session_state.current_question_index < len(st.session_state.questions) - 1:
            st.session_state.current_question_index += 1
            similar_question = generate_similar_question(current_question)
            st.session_state.questions.insert(st.session_state.current_question_index, similar_question)
        else:
            st.write("Interview session completed!")
        
        st.experimental_rerun()

    if st.session_state.user_answers:
        st.subheader("Previous Answers")
        for q, a in zip(st.session_state.questions[:st.session_state.current_question_index], st.session_state.user_answers):
            st.write(f"Q: {q}")
            st.write(f"A: {a}")
            st.write("---")
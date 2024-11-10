import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import random

# Initialize Streamlit app
st.title("ML/DS Interview Preparation Assistant")
st.markdown("*Your AI companion for interview preparation and learning*")

# Configure Hugging Face credentials
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]

# Define templates for different functionalities
INTERVIEW_TOPICS = {
    "Machine Learning": ["algorithms", "model selection", "evaluation metrics", "feature engineering", "ensemble methods"],
    "Deep Learning": ["neural networks", "CNN", "RNN", "transformers", "optimization"],
    "Reinforcement Learning": ["Q-learning", "policy gradients", "SARSA", "DQN", "environment modeling"],
    "Computer Vision": ["image processing", "object detection", "segmentation", "CNNs for vision", "transformation"],
    "Recommendation Systems": ["collaborative filtering", "content-based", "hybrid systems", "matrix factorization", "evaluation"]
}

INTERVIEW_TEMPLATE = """You are an expert ML/DS technical interviewer specializing in {topic}.
Generate content following this format:

QUESTION:
[Ask a challenging but clear technical question about {topic}, focusing on {subtopic}]

EXPECTED_POINTS:
[Key technical points specific to {topic} that should be covered]

FEEDBACK_FORMAT:
1. Strengths 💪
2. Areas for Improvement 🎯
3. Additional Tips 💡

FOLLOW_UP:
[A related follow-up question building on {topic}]

Current conversation:
{history}
Human: {input}
Assistant: Let me generate that content following the specified format."""

PROBLEM_SOLVING_TEMPLATE = """You are an experienced ML/DS mentor helping students practice problem-solving.
Current conversation:
{history}
Human: {input}
Assistant: Let me help you think through this systematically and provide constructive feedback."""

CONCEPT_EXPLANATION_TEMPLATE = """You are a helpful ML/DS teacher who explains concepts clearly at different levels.
Current conversation:
{history}
Human: {input}
Assistant: I'll explain this concept in a clear and structured way."""

CODE_REVIEW_TEMPLATE = """You are a supportive senior ML engineer reviewing code.
Current conversation:
{history}
Human: {input}
Assistant: I'll review your code thoroughly and provide constructive suggestions."""

# Initialize the model with conversation history
@st.cache_resource
def init_model(template):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    return conversation

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    "Mock Interview", 
    "Problem Solving",
    "Concept Explanations", 
    "Code Review"
])

# Mock Interview Tab
with tab1:
    st.header("Interactive Mock Interview")
    st.markdown("*Have a natural interview conversation with AI*")
    
    # Initialize all session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "interview_active" not in st.session_state:
        st.session_state.interview_active = False
    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    # Topic selection when interview is not active
    if not st.session_state.interview_active:
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.selectbox(
                "Select Interview Topic:",
                list(INTERVIEW_TOPICS.keys())
            )
            print("\n")
        with col2:
            if st.button("Start Interview"):
                st.session_state.selected_topic = topic
                st.session_state.chat_history = []
                st.session_state.interview_active = True
                
                # Initialize conversation instance
                st.session_state.conversation = init_model(INTERVIEW_TEMPLATE)
                
                # Generate first question
                subtopic = random.choice(INTERVIEW_TOPICS[topic])
                formatted_prompt = INTERVIEW_TEMPLATE.format(
                    topic=topic,
                    subtopic=subtopic,
                    history="",
                    input=f"Generate a technical interview question about {topic}, focusing on {subtopic}"
                )
                
                response = st.session_state.conversation.predict(input=formatted_prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

    # Display chat history with cleaner formatting
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                content = message["content"]
                if "QUESTION:" in content:
                    # For initial questions, only show the question
                    question = content.split("QUESTION:")[1].split("EXPECTED_POINTS:")[0].strip()
                    st.write(question)
                elif "Strengths 💪" in content:
                    # Extract only strengths, areas for improvement, and follow-up
                    lines = content.split("\n")
                    feedback_section = []
                    follow_up = ""
                    
                    for line in lines:
                        if any(marker in line for marker in ["Strengths 💪", "Areas for Improvement 🎯", "Additional Tips 💡"]):
                            feedback_section.append(line)
                        elif line.startswith("FOLLOW-UP:"):
                            follow_up = line.replace("FOLLOW-UP:", "").strip()
                    
                    # Display only feedback and follow-up
                    if feedback_section:
                        st.markdown("\n".join(feedback_section))
                    if follow_up:
                        st.markdown("---")
                        st.markdown(follow_up)
                else:
                    st.write(content)
            else:
                # Show user responses as is
                st.markdown(f"> {message['content']}")

    # Handle user input
    if st.session_state.interview_active:
        user_input = st.chat_input("Your answer (type 'new interview' to restart):")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            if user_input.lower() == "new interview":
                st.session_state.interview_active = False
                st.session_state.conversation = None
                st.rerun()
            else:
                feedback_prompt = f"""Evaluate this answer for a {st.session_state.selected_topic} interview question:

EVALUATION:
1. Strengths 💪
[List strengths specific to {st.session_state.selected_topic}]

2. Areas for Improvement 🎯
[List technical areas to improve]

3. Additional Tips 💡
[Provide {st.session_state.selected_topic}-specific tips]

FOLLOW-UP:
[Ask a related follow-up question about {st.session_state.selected_topic}]"""
                
                feedback = st.session_state.conversation.predict(
                    input=f"Answer to evaluate: {user_input}\n{feedback_prompt}"
                )
                st.session_state.chat_history.append({"role": "assistant", "content": feedback})
                st.rerun()

# Problem Solving Tab        
with tab2:
    st.header("Scenario-Based Problem Solving")
    st.markdown("*Practice solving real-world ML/DS challenges*")
    
    conversation = init_model(PROBLEM_SOLVING_TEMPLATE)
    
    problem_types = [
        "Data Preprocessing",
        "Model Selection",
        "Hyperparameter Tuning",
        "Model Evaluation"
    ]
    
    selected_type = st.selectbox("Select Problem Type:", problem_types)
    
    if st.button("Generate Problem"):
        prompt = f"""Create a realistic machine learning scenario about {selected_type}.
        Include:
        1. Context 🎯
        2. Problem Statement 📝
        3. Available Data 📊
        4. Constraints (if any) ⚠️"""
        scenario = conversation.predict(input=prompt)
        st.write("Scenario:", scenario)
        
    solution = st.text_area("Describe Your Approach:")
    if st.button("Get Feedback"):
        feedback = conversation.predict(
            input=f"""Evaluate this solution approach: {solution}
            Provide feedback on:
            1. Methodology 📈
            2. Technical Understanding 🧠
            3. Best Practices 🎯
            4. Suggestions for Improvement 💡"""
        )
        st.write("Feedback:", feedback)

# Concept Explanations Tab
with tab3:
    st.header("Theory and Concepts")
    
    concept = st.text_input("Enter ML/DS Concept to Explain:")
    depth = st.select_slider(
        "Explanation Depth:",
        options=["Basic", "Intermediate", "Advanced"]
    )
    
    if st.button("Explain"):
        prompt = f"Explain {concept} at a {depth} level"
        explanation = conversation.predict(input=prompt)
        st.write(explanation)

# Code Review Tab        
with tab4:
    st.header("Code Review Assistant")
    
    code = st.text_area("Paste your code here:", height=200)
    review_type = st.multiselect(
        "Select Review Focus:",
        ["Optimization", "Best Practices", "Error Detection", "Documentation"]
    )
    
    if st.button("Review Code"):
        prompt = f"Review this code focusing on {', '.join(review_type)}:\n{code}"
        review = conversation.predict(input=prompt)
        st.write("Review Comments:", review)
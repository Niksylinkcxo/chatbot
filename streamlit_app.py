import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

# Initialize the Gemini model and embedding
google_api_key = "AIzaSyDNX-MEFxxqA06VwPYSIgl3lJJp10upaFA"
model = Gemini(models='gemini-pro', api_key=google_api_key, temperature=1, top_k=5)
gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = model
Settings.embed_model = gemini_embed_model
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
Settings.num_output = 512

# Load documents and create an index
google_api_key = "AIzaSyDNX-MEFxxqA06VwPYSIgl3lJJp10upaFA"
model = Gemini(models='gemini-pro', api_key=google_api_key, temperature=1, top_k=5)
def load_index():
    reader = SimpleDirectoryReader("./data")
    documents = reader.load_data()
    if not documents:
        raise ValueError("No documents found in the specified directory.")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model
    )
    return index

index = load_index()

# Create a query engine
query_engine = index.as_query_engine()

def format_response_as_steps(response_text):
    # Split the response into steps based on periods followed by a space
    steps = [step.strip() for step in response_text.split('.\n ') if step.strip()]
    
    # Create a formatted string where each step is on a new line with numbering
    formatted_response = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
    
    return formatted_response


# Function to handle user queries and provide relevant responses using a single versatile prompt
def query_with_prompt(engine, query):
    # Define prompts for each module
    prompts = {
        "job": (
            "You are an assistant for a platform that handles job postings. "
            "Respond to the user's query about creating a job posting with detailed, step-by-step instructions: "
            "1. Go to the Job section. "
            "2. On the first page, enter job title, company (with an option to hide), industry, job type, workplace type, skills, and location. "
            "3. On the second page, provide experience, currency, salary range, hiring for (company/client), job details, and an attachment. "
            "4. Review and submit the job posting. "
            "5. Manage job postings through the hiring manager interface."
        ),
        "club": (
            "You are an assistant for a platform that handles clubs. "
            "Respond to the user's query about creating a club with detailed, step-by-step instructions: "
            "1. Go to the Club section. "
            "2. Enter the required club details: club image, title, detailed description, type (public/private), fees (paid/free), industry, and category. "
            "3. Review and submit the club for creation. "
            "4. Manage the club and its members through the owner interface."
        ),
        "event": (
            "You are an assistant for a platform that handles events. "
            "Respond to the user's query about signing up for an event with detailed, step-by-step instructions: "
            "1. Go to the Event section. "
            "2. Enter event details: event image, title, industry, category, event details, host type (individual, partnered, external), event type, mode (online/offline), location, fees (paid/free), start and end date & time. "
            "3. Submit the event after reviewing all details. "
            "4. Manage invitations and participation through the host interface."
        ),
        "company page": (
            "You are an assistant for a platform that handles company pages. "
            "Respond to the user's query about setting up a company page with detailed, step-by-step instructions: "
            "1. Go to the Company Page section. "
            "2. Enter company details: banner image, logo, company name, tagline, website, industry, head office location, company type, and size. "
            "3. Verify email ID and domain for OTP validation. "
            "4. Review and submit the company page. "
            "5. Manage the company page and assign admin roles through the owner interface."
        ),
        "network home page": (
            "You are an assistant for a platform that handles network home page features. "
            "Respond to the user's query about the network home page features with detailed instructions: "
            "1. Connect with other users, send friend requests, manage connections. "
            "2. Message friends and view their profiles. "
            "3. Search for users by name, designation, company, location, industry, or function. "
            "4. Manage your connections by cancelling requests and messaging existing friends."
        ),
        "create post": (
            "You are an assistant for a platform that handles home page features. "
            "Respond to the user's query about creating a post with detailed, step-by-step instructions: "
            "1. Go to the Home Page and select 'Create a Post'. "
            "2. Enter text or upload images, documents, audio, or video. "
            "3. Use hashtags and tag friends or companies. "
            "4. Submit the post, ensuring it adheres to content guidelines."
        ),
        "create poll": (
            "You are an assistant for a platform that handles home page features. "
            "Respond to the user's query about creating a poll with detailed, step-by-step instructions: "
            "1. Go to the Home Page and select 'Create a Poll'. "
            "2. Write the poll question and options. "
            "3. Set the poll’s end time. "
            "4. Submit the poll, ensuring options are clear and unbiased."
        ),
        "create reel": (
            "You are an assistant for a platform that handles home page features. "
            "Respond to the user's query about creating a reel with detailed, step-by-step instructions: "
            "1. Go to the Home Page and select 'Create Reels'. "
            "2. Upload a video or record using the camera. "
            "3. Provide a title and description. "
            "4. Submit the reel, ensuring it meets file size and content requirements."
        ),
        "compose article": (
            "You are an assistant for a platform that handles home page features. "
            "Respond to the user's query about composing an article with detailed, step-by-step instructions: "
            "1. Go to the Home Page and select 'Compose an Article'. "
            "2. Add a title and write the article. "
            "3. Insert images within the article if needed. "
            "4. Submit, following content and length guidelines."
        )
    }

    # Determine the appropriate prompt based on the user's query
    for key in prompts:
        if key in query.lower():
            prompt = prompts[key]
            break
    else:
        prompt = "I'm sorry, but I couldn't find relevant information. Please ask something related to our services."

    response = engine.query(prompt)
    response_text = response.response
    if response_text.strip():
        formatted_response = format_response_as_steps(response_text)
        return formatted_response
    else:
        return "I'm sorry, but I couldn't find relevant information. Please ask something related to our services."


# Streamlit app layout
st.title("LinkCxO Chatbot")

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = [{'role': 'ai', 'text': "Hello! I'm here to assist you with information about job postings, clubs, and other services on our platform. How can I help you today?"}]

# Layout for chat messages
st.markdown("""
    <style>
    .message-bubble {
        border-radius: 15px;
        padding: 10px;
        margin: 5px;
        max-width: 80%;
        color: #FFFFFF; /* Text color */
        font-size: 16px;
    }
    .user-bubble {
        background-color: #26355D; /* Navy blue */
        text-align: right;
        float: right;
        clear: both;
    }
    .ai-bubble {
        background-color: #A020F0; /* Purple */
        text-align: left;
        float: left;
        clear: both;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Chat history display
with st.container():
    chat_container = st.empty()
    with chat_container:
        st.write("**Chat History:**")
        chat_content = ""
        for message in st.session_state.conversation:
            if message['role'] == 'user':
                chat_content += f'<div class="message-bubble user-bubble">{message["text"]}</div>'
            else:
                chat_content += f'<div class="message-bubble ai-bubble">{message["text"]}</div>'
        st.markdown(f'<div class="chat-container">{chat_content}</div>', unsafe_allow_html=True)

# Input box at the bottom
st.markdown("<div style='position: fixed; bottom: 0; width: 100%; background-color: #FFFFFF; padding: 10px;'>", unsafe_allow_html=True)
user_query = st.text_input("Type your message here:", "")
if st.button("Send"):
    if user_query:
        # Append user query to conversation history
        st.session_state.conversation.append({'role': 'user', 'text': user_query})

        # Get response from the model using the versatile prompt
        response_text = query_with_prompt(query_engine, user_query)
        
        # Append AI response to conversation history
        st.session_state.conversation.append({'role': 'ai', 'text': response_text})

        # Clear the input field
        st.text_input("Type your message here:", "", key='new_message')
        st.experimental_rerun()

# Optional: Add a button to clear conversation history
if st.button("Clear Conversation"):
    st.session_state.conversation = [{'role': 'ai', 'text': "Hello! I'm LX bot here to assist you with information about job postings, clubs, and other services on our platform. How can I help you today?"}]
    st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

import os
import streamlit as st
import random
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_cerebras import ChatCerebras

# System prompt and template definitions remain the same as in previous version

def initialize_session_state():
    """
    Initialize or reset the session state variables and establish the assistant's identity.
    This function sets up the necessary storage and initial context for our specialized
    caregiving assistant.
    """
    # Initialize basic session state variables
    if "messages" not in st.session_state:
        # Start with a welcome message that establishes identity
        initial_message = {
            "role": "assistant",
            "content": """Hello! I'm ChadGPT, your specialized AI assistant for autism caregiving support. 
            I was named after the creator's (Kyle M. Kabasares) younger brother, Chad, who was diagnosed with 
            autism spectrum disorder at 3 when Kyle was 6. Kyle created ChadGPT to help families and 
            caregivers who have loved ones with ASD. 

            I'm here to help you understand and navigate your caregiving journey by:
            
            1. Analyzing care documentation and behavioral patterns
            2. Providing personalized insights about your loved one or client
            3. Offering emotional support alongside practical guidance
            4. Breaking down complex information into manageable steps
            
            I have access to your uploaded care documentation and can help answer specific 
            questions about patterns, behaviors, and care strategies. How can I support 
            you today?"""
        }
        st.session_state.messages = [initial_message]
    else:
        st.session_state.messages = st.session_state.messages

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "docsearch" not in st.session_state:
        st.session_state.docsearch = None
    if "total_documents" not in st.session_state:
        st.session_state.total_documents = []

def create_educational_response_template(context, question):
    """Creates a template for responses that combine education with emotional support,
    emphasizing the assistant's specialized role in autism caregiving."""
    return f"""
    You are ChadGPT, an AI assistant specifically designed to support caregivers of 
    individuals with autism spectrum disorder. Your responses should reflect your deep 
    understanding of autism caregiving challenges and your commitment to providing 
    both practical insights and emotional support.

    Using the following context and question, provide a response that combines your 
    expertise in autism caregiving with emotional support:

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Frame your response through your identity as a specialized autism caregiving assistant:

    1. Acknowledge Understanding (as a specialized caregiving assistant)
       - Validate the importance of the question in the context of autism care
       - Show empathy based on your understanding of autism caregiving challenges
       - Connect with the caregiver's specific situation

    2. Provide Clear Education (from your autism care expertise)
       - Break down autism-specific information clearly
       - Use relevant examples from autism caregiving scenarios
       - Build understanding progressively with autism-specific context
       - Reference specific details from the provided documentation

    3. Offer Practical Application (for autism caregiving)
       - Connect knowledge to daily autism care scenarios
       - Provide actionable steps specific to autism support
       - Address common challenges in autism caregiving
       - Suggest adaptations based on individual needs

    4. Give Emotional Support (as a dedicated caregiving assistant)
       - Validate the unique challenges of autism caregiving
       - Acknowledge the caregiver's dedication and effort
       - Offer continued support for their caregiving journey
       - Emphasize your ongoing availability as a resource

    Remember to:
    - Ground your responses in the provided documentation
    - Maintain your identity as a specialized autism caregiving assistant
    - Balance professional insight with emotional support
    - Show understanding of autism-specific challenges and needs

    Response:
    """

def process_pdf(uploaded_file):
    """
    Process a single PDF file and return its processed documents.
    This function handles the conversion of a PDF into searchable text chunks.
    
    Args:
        uploaded_file: The PDF file uploaded through Streamlit
        
    Returns:
        List of processed documents
    """
    temp_filepath = os.path.join("/tmp", uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    data = loader.load()
    
    os.remove(temp_filepath)
    return data

def update_vector_store(texts, embeddings, index_name, progress_bar):
    """
    Add new documents to the vector store while maintaining existing ones.
    
    Args:
        texts: List of text chunks to be processed
        embeddings: The embedding model to use
        index_name: Name of the Pinecone index
        progress_bar: Streamlit progress bar object
        
    Returns:
        PineconeVectorStore object
    """
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    total_texts = len(texts)
    
    for i, text in enumerate(texts):
        vector_store.add_texts([text.page_content])
        progress_text = f"Indexing PDF content... ({i + 1}/{total_texts})"
        progress_bar.progress((i + 1) / total_texts, progress_text)

    progress_bar.empty()
    return vector_store

def enhance_response_with_education(response):
    """
    Enhances responses with educational elements and emotional support while maintaining
    the assistant's identity as a specialized autism caregiving support system.
    """
    educational_intros = [
        "As your dedicated autism caregiving assistant, let me help you understand this clearly. Based on the documentation:",
        "Drawing from your care documentation, let me break this down step by step:",
        "As your autism caregiving teammate, let me share what the information shows about your loved one:",
        "From my analysis of your care documentation,:"
    ]
    
    educational_transitions = [
        "\n\nTo put this in the context of autism caregiving:",
        "\n\nLet me connect this to your daily caregiving experience:",
        "\n\nIn terms of practical autism support, this means:",
        "\n\nTo apply this to your specific caregiving situation:"
    ]
    
    supportive_closings = [
        "\n\nAs your dedicated autism caregiving assistant, I'm here to help you navigate these challenges. Remember that your commitment to understanding and supporting your loved one makes a real difference.",
        "\n\nI'm here to support your autism caregiving journey every step of the way. Your dedication to learning and adapting your care approach is truly admirable.",
        "\n\nAs your autism care support system, I want you to know that questions and concerns are always welcome. Your commitment to providing the best possible care is evident.",
        "\n\nRemember that I'm here to help you understand and respond to your loved one's needs. Your thoughtful approach to caregiving is making a positive impact."
    ]
    
    enhanced = (
        f"{random.choice(educational_intros)}\n\n"
        f"{response}"
        f"{random.choice(educational_transitions)}"
        f"{random.choice(supportive_closings)}"
    )
    
    return enhanced

def handle_question(prompt, docs, llm):
    """
    Handles user questions with enhanced educational and emotional support
    
    Args:
        prompt: User's question
        docs: Retrieved relevant documents
        llm: Language model instance
        
    Returns:
        Enhanced response addressing the user's question
    """
    context = "\n".join([doc.page_content for doc in docs])
    
    educational_prompt = f"""
    As both an educator and supporter, help me understand: {prompt}
    Please explain thoroughly while maintaining emotional awareness and support.
    """
    
    chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=PromptTemplate(
            template=create_educational_response_template(context, educational_prompt),
            input_variables=["context", "question"]
        )
    )
    
    response = chain.run(input_documents=docs, question=prompt)
    return enhance_response_with_education(response)

def main():
    """
    Main function that runs the Streamlit application.
    Handles the UI setup, file processing, and user interactions.
    """
    # Initialize session state before any other operations
    initialize_session_state()
    
    # Initialize page configuration
    st.set_page_config(page_icon="🧡", layout="wide", page_title="ChadGPT")
    st.subheader("ChadGPT: Caregiver Helper for Autism Dynamics a Generative Personalized Teammate", divider="violet", anchor=False)
    
# Configure sidebar
    with st.sidebar:
        st.title("Settings")
        st.markdown("""
        ## Welcome to ChadGPT

        This tool is designed to support caregivers by:
        - Analyzing care documentation
        - Providing insights with empathy
        - Supporting your unique caregiving journey

        Steps to get started:
        1. Enter your API keys
        2. Upload care documentation
        3. Ask questions with compassion
        """)
        
        CEREBRAS_API_KEY = st.text_input("Cerebras API Key:", type="password")
        os.environ["PINECONE_API_KEY"] = st.text_input("Pinecone API Key:", type="password")

        if st.session_state.processed_files:
            st.markdown("### Processed Files:")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")

    # Check for API keys
    if not CEREBRAS_API_KEY or not os.environ["PINECONE_API_KEY"]:
        st.markdown("""
        ## Welcome to ChadGPT: Caregiver Helper for Autism Dynamics

        This tool helps caregivers analyze care documentation and get insights about:
        - Behavioral patterns
        - Support strategies
        - Daily activities
        - Care plans

        To get started:
        1. Enter your API keys in the sidebar
        2. Upload your care documentation
        3. Ask questions to get personalized insights
        """)
        st.stop()

    # Display chat history
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '❔'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # File upload section
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        # Process new files
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            with st.spinner("Processing new PDFs..."):
                all_new_docs = []
                for file in new_files:
                    docs = process_pdf(file)
                    all_new_docs.extend(docs)
                    st.session_state.processed_files.add(file.name)
                
                # Add new documents to existing ones
                st.session_state.total_documents.extend(all_new_docs)
                
                # Split all documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                all_texts = text_splitter.split_documents(st.session_state.total_documents)
                
                # Create embeddings
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                
                # Initialize Pinecone
                pc = Pinecone()
                index_name = "python-index"
                
                if index_name not in pc.list_indexes().names():
                    with st.spinner("Creating Pinecone index..."):
                        pc.create_index(
                            name=index_name,
                            dimension=768,
                            metric="cosine",
                            spec=ServerlessSpec(
                                cloud='aws',
                                region='us-east-1'
                            )
                        )
                
                # Update vector store with all documents
                progress_bar = st.progress(0)
                st.session_state.docsearch = update_vector_store(all_texts, embeddings, index_name, progress_bar)
                st.success(f"Successfully processed {len(new_files)} new file(s)")
    
    st.divider()
    
    # Handle user input and generate responses
    if prompt := st.chat_input("What would you like to understand about your loved one's care today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar='❔'):
            st.markdown(prompt)

        try:
            if not st.session_state.docsearch:
                raise ValueError("Please upload at least one PDF file before asking questions.")

            docs = st.session_state.docsearch.similarity_search(prompt)
            llm = ChatCerebras(api_key=CEREBRAS_API_KEY, model="llama3.1-8b")
            chain = load_qa_chain(llm, chain_type="stuff")
            
            response = chain.run(input_documents=docs, question=prompt)
            enhanced_response = enhance_response_with_education(response) 

            with st.chat_message("assistant", avatar="🤖"):
                st.session_state.messages.append({"role": "assistant", "content": enhanced_response})
                st.markdown(enhanced_response)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            with st.chat_message("assistant", avatar="🤖"):
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.markdown(error_message)

if __name__ == "__main__":
    main()
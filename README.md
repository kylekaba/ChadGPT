# ChadGPT: Caregiver Helper for Autism Dynamics - a Generative Personalized Trainer

<img src="assets/chadgpt-interface.png" width="600" alt="ChadGPT Interface">

ChadGPT is an AI-powered assistant leveraging Retrieval Augmented Generation (RAG) to support caregivers of individuals diagnosed with Autism Spectrum Disorder (ASD) by analyzing an individual's documentation (behavioral reports, medical reports, anecdotal documents) and providing personalized insights. Named after the creator's brother Chad, this tool aims to help families and caregivers navigate their autism care journey by leveraging the power of artificial intelligence.

## Features

- 📊 Analysis of documentaiton to uncover behavioral patterns
- 🧠 Personalized insights about your loved one or client b
- 💡 Real-time suggestions based on historical data 
- 🎯 Task breakdown and practical guidance

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.8 or higher
- Docker installed on your local machine
- API Keys for:
  - Cerebras Cloud
  - Pinecone
- Sufficient storage space for model downloads

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChadGPT.git
cd ChadGPT
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up Docker for Ollama:
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull nomic-embed-text
```

## Configuration

1. Create a Pinecone Account:
   - Sign up at [Pinecone](https://www.pinecone.io/)
   - Create a new project
   - Generate an API key
   - Note your environment and project details

2. Get Cerebras API Access:
   - Sign up for Cerebras Cloud access
   - Generate an API key
   - Save your credentials securely

3. Environment Setup:
   Create a `.env` file in the project root:
   ```
   CEREBRAS_API_KEY=your_cerebras_key
   PINECONE_API_KEY=your_pinecone_key
   ```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Access the web interface:
   - Open your browser to `http://localhost:8501`
   - Enter your API keys in the sidebar
   - Upload care documentation PDFs
   - Start asking questions about your care documentation

## Documentation Upload Guidelines

For optimal results when uploading care documentation:

- Use clear, well-scanned PDF documents
- Ensure text is machine-readable
- Include relevant behavioral data and observations
- Upload multiple documents for comprehensive analysis

## Project Structure

```
ChadGPT/
├── .env                  # Environment variables (create this)
├── .gitignore
├── README.md
├── requirements.txt
├── main.py              # Main application code
├── src/
│   ├── __init__.py
│   ├── prompts.py       # Prompt templates
│   ├── processors.py    # Document processing
│   └── utils.py         # Utility functions
├── configs/
│   └── app_config.py    # Application configuration
└── tests/
    └── __init__.py
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Privacy and Data Security

ChadGPT handles sensitive behavioral and medical data. Please ensure:

- All uploaded documents are properly anonymized
- No personally identifiable information is shared
- Follow all applicable privacy regulations
- Use secure API keys and credentials

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Powered by Cerebras, Pinecone, and Ollama
- Named in honor of Chad Kabasares

## Future Features Roadmap

- [ ] Multi-modal support for audio/visual inputs
- [ ] Enhanced pattern recognition
- [ ] Custom behavioral tracking
- [ ] Mobile application support

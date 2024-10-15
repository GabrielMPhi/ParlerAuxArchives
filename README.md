# Local Knowledge assistant

## Features

- Provides an easy way to deploy a chatbot for chatting with your own data
- Let you configure and/or translate UI messages and llm prompt

## Installation

```bash
git clone https://github.com/isingasimplesong/paa.git
cd paa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## Configuration

- you can change or translate the prompt sent to the model in `template.txt`,
- you can change or translate UI messages in `messages.json`,
- you can configure the Chatbot UI in `ui.json`

## Roadmap

- Provide a simpler way to change llm & embedding models

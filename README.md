# News Summarizer with OpenAI Assistant and NewsAPI

This project demonstrates the integration of OpenAI's Assistant API with function calling to [NewsAPI](https://newsapi.org/), and a user interface built using Streamlit. The goal is to create an interactive application that allows users to search for and retrieve news articles using OpenAI's language models and display the results in a user-friendly interface.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
## Features

- **NewsAPI Integration**: Search and retrieve the latest news articles from NewsAPI.
- **OpenAI Assistant API Integration**: Use OpenAI's language models to process and interpret news data.
- **Streamlit User Interface**: A simple and interactive UI for users to interact with the application.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/tdkhoavn/news_summarizer.git
    cd news_summarizer
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure the OpenAI API, NewsAPI key**:
    Create a `.env` file in the root directory of the project and add the following line:
    ```bash
    OPENAI_API_KEY=<YOUR_API_KEY>
    NEWS_API_KEY=<YOUR_API_KEY>
    ```
## Usage

To run the application, use the following command:

```bash
streamlit run main.py
```
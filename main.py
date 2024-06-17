import json
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta import AssistantStreamEvent
from typing_extensions import override
from openai import AssistantEventHandler

load_dotenv()

model = "gpt-3.5-turbo-16k"

client = OpenAI()


def get_news(topic):
    news_api_key = os.environ.get("NEWS_API_KEY")
    if news_api_key is None:
        raise ValueError("NEWS_API_KEY is not set")
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5"
    )
    response = requests.get(url)
    response.raise_for_status()
    if response.status_code == 200:
        news_str = json.dumps(response.json(), indent=4)
        news_json = json.loads(news_str)

        data = news_json

        status = data["status"]
        total_results = data["totalResults"]
        articles = data["articles"]
        final_articles = []

        for article in articles:
            source_name = article["source"]["name"]
            author = article["author"]
            title = article["title"]
            description = article["description"]
            url = article["url"]
            content = article["content"]
            title_description = f"""
                Title: {title},
                - Author: {author},
                - Source: {source_name},
                - Description: {description},
                - Content: {content},
                - URL: {url}
            """
            final_articles.append(title_description)
        return final_articles
    return []


class AssistantManager:
    assistant_id = os.environ.get("OPENAI_ASST_ID")
    thread_id = os.environ.get("OPENAI_ASST_THREAD_ID")

    def __init__(self, model: str = model):
        self.client = client
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None
        self.summary = None
        if AssistantManager.assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(
                assistant_id=AssistantManager.assistant_id
            )
        if AssistantManager.thread_id:
            self.thread = self.client.beta.threads.retrieve(
                thread_id=AssistantManager.thread_id
            )

    def create_assistant(self, name, instructions, tools):
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                model=self.model,
                tools=tools,
            )
            self.assistant = assistant_obj
            AssistantManager.assistant_id = self.assistant.id
            print(f"Assistant ID: {AssistantManager.assistant_id}")

    def create_thread(self):
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            self.thread = thread_obj
            AssistantManager.thread_id = self.thread.id
            print(f"Thread ID: {AssistantManager.thread_id}")

    def add_message_to_thread(self, role, content):
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content,
            )

    def run_assistant(self):
        with client.beta.threads.runs.stream(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                event_handler=EventHandler(),
        ) as stream:
            stream.until_done()


class EventHandler(AssistantEventHandler):
    @override
    def on_event(self, event: AssistantStreamEvent) -> None:
        if event.event == "thread.run.requires_action":
            run_id = event.data.id
            self.handle_requires_action(event.data, run_id)

    def handle_requires_action(self, data, run_id):
        submit_tool_outputs = data.required_action.submit_tool_outputs.model_dump()

        tool_outputs = []
        for tool in submit_tool_outputs["tool_calls"]:
            func_name = tool["function"]["name"]
            arguments = json.loads(tool["function"]["arguments"])

            if func_name == "get_news":
                output = get_news(arguments["topic"])
                final_str = ""
                for item in output:
                    final_str += "".join(item)

                tool_outputs.append({"tool_call_id": tool["id"], "output": final_str})

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs)

    def submit_tool_outputs(self, tool_outputs):
        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=self.current_run.thread_id,
                run_id=self.current_run.id,
                tool_outputs=tool_outputs,
                event_handler=EventHandler(),
        ) as stream:
            for text in stream.text_deltas:
                st.session_state.msg.append(text)


if "msg" not in st.session_state:
    st.session_state.msg = []


def data_streamer():
    import time
    for word in st.session_state.msg:
        yield word
        time.sleep(0.02)


def main():
    # Streamlit
    st.title("News Summarizer")
    st.write(
        f"""
        - This app summarizes news articles using Assistant API with Function Calling Tool.
        - The app uses the News API to get the news articles.
        - The Assistant will summarize the news articles in Vietnamese.
        """
    )
    with st.form(key="user_input_form"):
        topic = st.text_area("Enter topic")
        submit_button = st.form_submit_button(label="Run Assistant")

        if submit_button:
            if not topic:
                st.warning("Please enter a topic")
            else:
                manager = AssistantManager()
                manager.create_assistant(
                    name="News Summarizer",
                    instructions="You are a personal article summarizer Assistant who knows how to take a list of article's titles and descriptions and then write a short summary of all the news articles",
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "get_news",
                            "description": "Get the list of articles/news for the given topic",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "topic": {
                                        "type": "string",
                                        "description": "The topic for the news, e.g. bitcoin"
                                    }
                                },
                                "required": ["topic"],
                            }
                        }
                    }]
                )

                manager.create_thread()

                manager.add_message_to_thread(
                    role="user",
                    content=f"Summarize the news on this topic {topic} in Vietnamese?"
                )

                manager.run_assistant()

                with st.chat_message("assistant"):
                    st.write_stream(data_streamer)


if __name__ == "__main__":
    main()

import json
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

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
            assistant_obj = client.beta.assistants.create(
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
            thread_obj = client.beta.threads.create()
            self.thread = thread_obj
            AssistantManager.thread_id = self.thread.id
            print(f"Thread ID: {AssistantManager.thread_id}")

    def add_message_to_thread(self, role, content):
        if self.thread:
            if self.run:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )
                if run_status.status != "completed":
                    print("Wait for the current run to complete before adding a new message.")
                    return
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content
            )

    def run_assistant(self, instructions):
        if self.thread and self.assistant:
            self.run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions
            )

    def process_message(self):
        if self.thread:
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            summary = []
            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)
            self.summary = "\n".join(summary)
            print(f"SUMMARY: {role.capitalize()}: => {response}")

            # for msg in messages:
            #     role = msg.role
            #     content = msg.content[0].text.value
            #     print(f"SUMMARY: {role.capitalize()}: => {content}")

    # for streamlit
    def get_summary(self):
        return self.summary

    def wait_for_completion(self):
        if self.run and self.thread:
            while True:
                time.sleep(5)
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )
                print(f"RUN STATUS: {run_status.model_dump_json(indent=4)}")
                if run_status.status == "completed":
                    self.process_message()
                    break
                if run_status.status == "requires_action":
                    print("FUNCTION CALL TO HANDLE ACTION")
                    required_actions = run_status.required_action.submit_tool_outputs.model_dump()
                    self.call_required_functions(required_actions=required_actions)

    def call_required_functions(self, required_actions):
        if not self.run:
            return
        tool_outputs = []
        for action in required_actions["tool_calls"]:
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_news":
                output = get_news(arguments["topic"])
                print(f"STUFF: {output}")
                final_str = ""
                for item in output:
                    final_str += "".join(item)

                tool_outputs.append({"tool_call_id": action["id"], "output": final_str})
            else:
                raise ValueError("Function not found")

        print("Summiting outputs back to the Assistant...")
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=tool_outputs
        )

    def run_steps(self):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id,
            run_id=self.run.id
        )
        print(f"Steps--->{run_steps}")
        return run_steps.data


def main():
    manager = AssistantManager()
    # Streamlit
    st.title("News Summarizer")
    st.write("This app summarizes news articles using GPT-3.5.")
    with st.form(key="user_input_form"):
        instructions = st.text_area("Enter topic")
        submit_button = st.form_submit_button(label="Run Assistant")

        if submit_button:
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
                                    "description": "The topic to search for in the news articles"
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
                content=f"Summarize the news on this topic {instructions} in Vietnamese?"
            )
            manager.run_assistant(instructions="Summarize the news")

            manager.wait_for_completion()

            summary = manager.get_summary()

            st.write(summary)

            st.text("Run Steps:")
            st.code(manager.run_steps(), line_numbers=True)


if __name__ == "__main__":
    main()

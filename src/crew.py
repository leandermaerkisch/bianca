from crewai import Crew

import os
from textwrap import dedent

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

os.environ["OPENAI_API_KEY"] = "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"


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

search_tool = DuckDuckGoSearchRun()

researcher = Agent(
    role="Market Research Analyst",
    goal="Identify potential customers in the pathology and digital pathology sectors",
    backstory="""You are a market analyst specializing in the healthcare technology sector,
                  with a focus on pathology and digital pathology markets.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    api_key=api_key,
)

writer = Agent(
    role="Report Writer",
    goal="Compile the market research findings into a comprehensive report",
    backstory="""You are an experienced writer known for your ability to transform complex data into clear, actionable reports.""",
    verbose=True,
    allow_delegation=True,
    api_key=api_key,
)

task1 = Task(
    description=dedent("""\
        Identify potential customers in the pathology and digital pathology sectors including hospitals, diagnostic labs, pathology departments, research institutions, and biotech companies. Focus on organizations in the United States, Europe, and Asia known for adopting new technology and having significant pathology departments or high sample processing needs."""),
    expected_output=dedent("""\
        A list of potential customers with details such as contact person, email/phone number, type of customer, and geography. This list will be used to reach out to these entities to initiate customer discovery and research processes."""),
    agent=researcher,
)

task2 = Task(
    description=dedent("""\
        Using the list and details provided by the market research analyst, compile a comprehensive report that summarizes the potential customers, their relevance, and strategic value. The report should be structured to facilitate easy understanding and quick decision-making for outreach strategies."""),
    expected_output=dedent("""\
        A comprehensive report detailing potential customers in the pathology and digital pathology sectors, including strategic insights and recommendations for outreach."""),
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=2,
)

if __name__ == "__main__":
    result = crew.kickoff()

    print("######################")
    print(result)
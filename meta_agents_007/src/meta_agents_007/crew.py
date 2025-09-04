from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from .tools import builtin_tools
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
LOCAL_MODEL = os.getenv('LOCAL_MODEL')
BASE_URL = os.getenv('BASE_URL')
API_KEY = os.getenv('API_KEY')

local_llm = LLM(**builtin_tools.LLM_CONFIG)
local_emb = builtin_tools.EMB_CONFIG

@CrewBase
class MetaAgents007():
    """MetaAgents007 crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self._custom_storage()

    def _custom_storage(self):
        root = Path(__file__).parent
        storage_dir = root / 'mem_storage'
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)

    #----- AGENTS -----#
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],  
            llm=local_llm
        )

    @agent
    def agents_prompt_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['agents_prompt_engineer'],  
            llm=local_llm,
            tools=[
                builtin_tools.web_search_tool('https://docs.crewai.com/concepts/agents#yaml-configuration-recommended'),
            ]
        )
    
    @agent
    def task_prompt_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['task_prompt_engineer'],  
            llm=local_llm,
            tools=[
                builtin_tools.web_search_tool('https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended'),
            ]
        )

    #----- TASKS -----#
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  
        )

    @task
    def author_agents_yaml_task(self) -> Task:
        return Task(
            config=self.tasks_config['author_agents_yaml_task'],  
        )
    
    @task
    def author_tasks_yaml_task(self) -> Task:
        return Task(
            config=self.tasks_config['author_tasks_yaml_task'],  
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MetaAgents007 crew"""

        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            memory=True, # Add embedder if True
            cache=True,
            embedder = dict(
                provider='openai',
                config=local_emb
            )
        )

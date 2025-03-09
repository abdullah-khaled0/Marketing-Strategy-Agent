from typing import List
from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field


llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.5)


class MarketStrategy(BaseModel):
	"""Market strategy model"""
	name: str = Field(..., description="Name of the market strategy")
	tactics: List[str] = Field(..., description="List of tactics to be used in the market strategy")
	channels: List[str] = Field(..., description="List of channels to be used in the market strategy")
	KPIs: List[str] = Field(..., description="List of KPIs to be used in the market strategy")

class CampaignIdea(BaseModel):
	"""Campaign idea model"""
	name: str = Field(..., description="Name of the campaign idea")
	description: str = Field(..., description="Description of the campaign idea")
	audience: str = Field(..., description="Audience of the campaign idea")
	channel: str = Field(..., description="Channel of the campaign idea")

class Copy(BaseModel):
	"""Copy model"""
	title: str = Field(..., description="Title of the copy")
	body: str = Field(..., description="Body of the copy")

@CrewBase
class MarketingPostsCrew():
	"""MarketingPosts crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def lead_market_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['lead_market_analyst'],
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			llm=llm,
			verbose=True,
			memory=False,
		)

	@agent
	def chief_marketing_strategist(self) -> Agent:
		return Agent(
			config=self.agents_config['chief_marketing_strategist'],
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			llm=llm,
			verbose=True,
			memory=False,
		)

	@agent
	def creative_content_creator(self) -> Agent:
		return Agent(
			config=self.agents_config['creative_content_creator'],
			llm=llm,
			verbose=True,
			memory=False,
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			agent=self.lead_market_analyst()
		)

	@task
	def project_understanding_task(self) -> Task:
		return Task(
			config=self.tasks_config['project_understanding_task'],
			agent=self.chief_marketing_strategist()
		)

	@task
	def marketing_strategy_task(self) -> Task:
		return Task(
			config=self.tasks_config['marketing_strategy_task'],
			agent=self.chief_marketing_strategist(),
			output_json=MarketStrategy
		)

	@task
	def campaign_idea_task(self) -> Task:
		return Task(
			config=self.tasks_config['campaign_idea_task'],
			agent=self.creative_content_creator(),
			output_json=CampaignIdea
		)

	@task
	def copy_creation_task(self) -> Task:
		return Task(
			config=self.tasks_config['copy_creation_task'],
			agent=self.creative_content_creator(),
			context=[self.marketing_strategy_task(), self.campaign_idea_task()],
			output_file='report.html'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the MarketingPosts crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)

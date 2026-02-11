from langgraph.graph import StateGraph, START , END
from typing import TypedDict, Annotated , List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
from pathlib import Path
import operator
from pydantic import BaseModel, Field
from typing import Literal
from langchain_mistralai import ChatMistralAI
from langgraph.types import Send
import re
import os
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Literal, Dict, Any, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from enum import Enum
import operator
from pathlib import Path

key = "N56JL9JhomtYzc64dQh7K05t6m0G8u9Q"

model = ChatMistralAI(
    model="mistral-large-2512",
    mistral_api_key= key,  # Direct key
    temperature=0.7
)

# ==================== ENUMS ====================

class TaskStatus(str, Enum):
    """Enumeration for task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResearchDepth(str, Enum):
    """Research depth required for task"""
    SURFACE = "surface"
    MODERATE = "moderate"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


class SourceType(str, Enum):
    """Types of sources to prioritize"""
    ACADEMIC = "academic"
    NEWS = "news"
    INDUSTRY = "industry"
    SOCIAL_MEDIA = "social_media"
    MIXED = "mixed"


# ==================== MODELS ====================

class TaskMetadata(BaseModel):
    """Metadata for task execution tracking"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    retry_count: int = Field(default=0, ge=0, le=5)
    error_message: Optional[str] = None


class Task(BaseModel):
    """Enhanced task model with comprehensive configuration"""
    
    # Core fields (keeping your original structure)
    id: int = Field(..., description="Unique task identifier", ge=1)
    title: str = Field(..., min_length=3, max_length=200, description="Concise task title")
    brief: str = Field(
        ..., 
        min_length=10,
        description="Detailed description of what needs to be covered"
    )
    Systemprompt: str = Field(  # Keeping your original naming
        ..., 
        min_length=20,
        description="Generate a system prompt for each task"
    )
    
    # Enhanced fields
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority level for execution ordering"
    )
    
    research_depth: ResearchDepth = Field(
        default=ResearchDepth.MODERATE,
        description="Depth of research required"
    )
    
    source_types: List[SourceType] = Field(
        default=[SourceType.MIXED],
        description="Preferred source types for research"
    )
    
    depends_on: List[int] = Field(
        default_factory=list,
        description="List of task IDs that must complete before this task"
    )
    
    estimated_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        le=100000,
        description="Estimated token count for output"
    )
    
    keywords: List[str] = Field(
        default_factory=list,
        description="Key terms and concepts to focus on"
    )
    
    min_word_count: int = Field(
        default=300,
        ge=50,
        description="Minimum word count for task output"
    )
    
    max_word_count: int = Field(
        default=1500,
        ge=100,
        description="Maximum word count for task output"
    )
    
    required_citations: int = Field(
        default=3,
        ge=0,
        description="Minimum number of citations/sources required"
    )
    
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task execution status"
    )
    
    metadata: TaskMetadata = Field(
        default_factory=TaskMetadata,
        description="Task execution metadata and metrics"
    )
    
    output: Optional[str] = Field(
        default=None,
        description="Generated content for this task"
    )
    
    sources_used: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of sources cited in the output"
    )
    
    custom_instructions: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Additional custom instructions for this specific task"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Categorization tags for the task"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "title": "Introduction to Self-Attention",
                "brief": "Explain the basic concept and motivation behind self-attention",
                "Systemprompt": "You are a technical writer explaining ML concepts..."
            }
        }
    )
    
    @field_validator('min_word_count', 'max_word_count')
    @classmethod
    def validate_word_counts(cls, v, info):
        """Ensure word count constraints are logical"""
        if info.field_name == 'max_word_count' and 'min_word_count' in info.data:
            if v < info.data['min_word_count']:
                raise ValueError('max_word_count must be >= min_word_count')
        return v
    
    def mark_in_progress(self):
        """Mark task as in progress"""
        self.status = TaskStatus.IN_PROGRESS
        self.metadata.started_at = datetime.utcnow()
        self.metadata.updated_at = datetime.utcnow()
    
    def mark_completed(self, output: str, sources: List[Dict[str, str]] = None):
        """Mark task as completed with output"""
        self.status = TaskStatus.COMPLETED
        self.output = output
        if sources:
            self.sources_used = sources
        self.metadata.completed_at = datetime.utcnow()
        self.metadata.updated_at = datetime.utcnow()
        if self.metadata.started_at:
            self.metadata.execution_time_seconds = (
                self.metadata.completed_at - self.metadata.started_at
            ).total_seconds()
    
    def mark_failed(self, error: str):
        """Mark task as failed with error message"""
        self.status = TaskStatus.FAILED
        self.metadata.error_message = error
        self.metadata.updated_at = datetime.utcnow()
        self.metadata.retry_count += 1


class Plan(BaseModel):
    """Enhanced research plan with metadata and configuration"""
    
    # Core field (keeping your original structure)
    blog_title: str = Field(
        ..., 
        min_length=10,
        max_length=200,
        description="Compelling, SEO-optimized blog title"
    )
    
    tasks: List[Task] = Field(
        ..., 
        min_length=1,
        description="Ordered list of research tasks"
    )
    
    # Enhanced fields
    plan_id: str = Field(
        default_factory=lambda: f"plan_{datetime.utcnow().timestamp()}",
        description="Unique plan identifier"
    )
    
    subtitle: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Optional subtitle or tagline"
    )
    
    target_word_count: int = Field(
        default=2000,
        ge=500,
        le=50000,
        description="Target total word count for final blog"
    )
    
    target_audience: Literal["general", "intermediate", "expert", "academic"] = Field(
        default="general",
        description="Intended audience sophistication level"
    )
    
    tone: Literal["professional", "casual", "academic", "conversational", "technical"] = Field(
        default="professional",
        description="Writing tone for the content"
    )
    
    seo_keywords: List[str] = Field(
        default_factory=list,
        description="Primary SEO keywords to target"
    )
    
    meta_description: Optional[str] = Field(
        default=None,
        max_length=160,
        description="SEO meta description"
    )
    
    minimum_citations: int = Field(
        default=10,
        ge=1,
        description="Minimum total citations across all tasks"
    )
    
    fact_check_required: bool = Field(
        default=True,
        description="Whether fact-checking is required"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    estimated_completion_time_minutes: Optional[int] = Field(
        default=None,
        description="Estimated time to complete all tasks"
    )
    
    include_introduction: bool = Field(default=True)
    include_conclusion: bool = Field(default=True)
    include_references: bool = Field(default=True)
    include_table_of_contents: bool = Field(default=False)
    
    custom_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom sections to include (section_name: requirements)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "blog_title": "Understanding Self-Attention: A Deep Dive",
                "target_word_count": 3000,
                "target_audience": "expert",
                "tasks": []
            }
        }
    )
    
    @field_validator('tasks')
    @classmethod
    def validate_task_ids(cls, v):
        """Ensure task IDs are unique"""
        task_ids = [task.id for task in v]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError('Task IDs must be unique')
        return v
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """Retrieve task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with specific status"""
        return [task for task in self.tasks if task.status == status]
    
    def get_next_task(self) -> Optional[Task]:
        """Get next pending task respecting dependencies"""
        completed_ids = {task.id for task in self.tasks if task.status == TaskStatus.COMPLETED}
        
        for task in sorted(self.tasks, key=lambda t: t.priority.value, reverse=True):
            if task.status == TaskStatus.PENDING:
                if all(dep_id in completed_ids for dep_id in task.depends_on):
                    return task
        return None
    
    @property
    def total_estimated_tokens(self) -> int:
        """Calculate total estimated tokens across all tasks"""
        return sum(task.estimated_tokens or 1000 for task in self.tasks)
    
    @property
    def completion_percentage(self) -> float:
        """Calculate plan completion percentage"""
        if not self.tasks:
            return 0.0
        completed = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        return (completed / len(self.tasks)) * 100


class State(TypedDict):
    """Enhanced state management - keeping your original structure"""
    topic: str
    plan: Plan
    # reducer: results from workers can get connected automatically
    Section: Annotated[List[str], operator.add]
    final_blog: str
    
    # Optional enhanced fields (won't break existing code if not used)
    all_sources: Annotated[List[Dict[str, str]], operator.add]
    total_word_count: int
    total_citations: int
    errors: Annotated[List[str], operator.add]
    started_at: datetime


# ==================== WORKFLOW FUNCTIONS ====================

def orchestrator(State: State):
    """Enhanced orchestrator with better planning"""
    llm_with_structure_output = model.with_structured_output(Plan)
    
    # Enhanced system prompt
    system_prompt = f"""Create a comprehensive blog plan with 5-7 well-structured sections.
    
    For each section (task):
    - Provide a clear, descriptive title
    - Write a detailed brief explaining what should be covered
    - Generate a specialized system prompt for that specific section
    - Set appropriate priority (high/medium/low)
    - Specify research depth needed (surface/moderate/deep)
    - Include relevant keywords
    
    Ensure logical flow and comprehensive coverage of the topic.
    Target audience: {State.get('target_audience', 'general')}
    Desired tone: {State.get('tone', 'professional')}
    """
    
    plan = llm_with_structure_output.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Topic: {State['topic']}")
    ])
    
    # Initialize state tracking
    return {
        "plan": plan,
        "started_at": datetime.utcnow(),
        "all_sources": [],
        "errors": []
    }


def condition_toget_workers(State: State):
    """Enhanced conditional routing with dependency awareness"""
    # Get next available tasks (respecting dependencies)
    plan = State["plan"]
    available_tasks = []
    
    # For now, send all pending tasks (dependency logic can be added)
    for task in plan.tasks:
        if task.status == TaskStatus.PENDING:
            available_tasks.append(task)
    
    return [
        Send("worker", {
            "task": task,
            "topic": State["topic"],
            "plan": State["plan"]
        }) for task in available_tasks
    ]


def worker(payload: dict) -> dict:
    """Enhanced worker with status tracking and error handling"""
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]
    blog_title = plan.blog_title
    
    try:
        # Mark task as in progress
        task.mark_in_progress()
        
        # Enhanced prompt with additional context
        section_prompt = [
            SystemMessage(content=f"{task.Systemprompt}\n\nAdditional Instructions:\n- Target word count: {task.min_word_count}-{task.max_word_count} words\n- Include {task.required_citations} relevant citations/references\n- Focus on keywords: {', '.join(task.keywords) if task.keywords else 'N/A'}\n- Research depth: {task.research_depth.value}"),
            HumanMessage(content=f"""Blog Title: {blog_title}
Topic: {topic}
Section Title: {task.title}
Section Brief: {task.brief}

{f'Custom Instructions: {task.custom_instructions}' if task.custom_instructions else ''}

Write a comprehensive section that:
1. Addresses all points in the brief
2. Maintains the specified tone: {plan.tone}
3. Is appropriate for {plan.target_audience} audience
4. Follows markdown formatting
5. Includes relevant examples and explanations

Return ONLY the section content in Markdown format.""")
        ]
        
        response = model.invoke(section_prompt).content.strip()
        
        # Mark task as completed
        task.mark_completed(output=response)
        
        return {
            "Section": [response],
            "all_sources": [],  # Can be populated if extracting citations
            "errors": []
        }
        
    except Exception as e:
        task.mark_failed(str(e))
        return {
            "Section": [f"## {task.title}\n\n*Error generating this section: {str(e)}*"],
            "all_sources": [],
            "errors": [f"Task {task.id} failed: {str(e)}"]
        }


def reducer(State: State) -> dict:
    """Enhanced reducer with metadata and quality checks"""
    
    title = State["plan"].blog_title
    plan = State["plan"]
    sections = State["Section"]
    
    # Build the document
    parts = [f"# {title}"]
    
    # Add metadata if available
    if plan.subtitle:
        parts.append(f"\n*{plan.subtitle}*")
    
    if plan.meta_description:
        parts.append(f"\n> {plan.meta_description}")
    
    # Add table of contents if requested
    if plan.include_table_of_contents:
        parts.append("\n## Table of Contents\n")
        for i, task in enumerate(plan.tasks, 1):
            parts.append(f"{i}. {task.title}")
        parts.append("")
    
    # Add introduction if needed
    if plan.include_introduction and not any("introduction" in s.lower() for s in sections):
        parts.append("\n## Introduction\n")
    
    # Add all sections
    body = "\n\n".join(sections).strip()
    parts.append(f"\n{body}")
    
    # Add conclusion if requested
    if plan.include_conclusion and not any("conclusion" in s.lower() for s in sections):
        parts.append("\n\n## Conclusion\n")
        parts.append(f"This comprehensive overview of {State['topic']} covers the essential aspects needed to understand this important topic.")
    
    # Add references if requested
    if plan.include_references:
        parts.append("\n\n## References\n")
        parts.append("*References and citations to be added*")
    
    # Calculate metrics
    final_md = "\n".join(parts)
    word_count = len(final_md.split())
    
    # Save to file
    filename = "research_paper5.md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")
    
    # Also save metadata
    metadata = {
        "title": title,
        "word_count": word_count,
        "sections": len(sections),
        "created_at": State.get("started_at", datetime.utcnow()).isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "plan_id": plan.plan_id,
        "target_audience": plan.target_audience,
        "tone": plan.tone
    }
    
    import json
    metadata_path = Path("research_paper2_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    return {
        "final_blog": final_md,
        "total_word_count": word_count,
        "total_citations": 0  # Can be calculated from sources
    }


# ==================== GRAPH SETUP ====================
graph = StateGraph(State)

graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", condition_toget_workers, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

workflow = graph.compile()

# ==================== EXECUTION ====================

result = workflow.invoke({
    "topic": "Planning Agents",
    "Section": [],
    "all_sources": [],
    "errors": [],
    "total_word_count": 0,
    "total_citations": 0
})

print(result)
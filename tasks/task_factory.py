# CrewAI task creation logic
from crewai import Task
import re
from typing import Optional

_VECTOR_ROUTER = None  # lazy global

def configure_vector_router(router):
    """Inject a vector router (set externally)."""
    global _VECTOR_ROUTER
    _VECTOR_ROUTER = router

def create_tasks(agents):
    """Create task objects using an existing list of agents.

    Expects agents in order: analyzer, preparer, generator, scorer.
    """
    tasks = [
        Task(
            description="""Analyze the provided resume against the job description with professional formatting. Structure your response as follows:\n\n## 📊 RESUME ANALYSIS REPORT\n...\nUse bullet points, clear headers, and professional language throughout.""",
            agent=agents[0],
            expected_output="A professionally formatted resume analysis report with clear sections, ratings, and actionable recommendations."
        ),
        Task(
            description="""Generate 5 resume-based technical interview questions with detailed answer guidance and example responses. Structure your response as follows:\n\n## 🎯 TECHNICAL INTERVIEW PREPARATION GUIDE\n...\nGenerate exactly 5 questions specifically based on the technical skills, tools, technologies, and experiences mentioned in the candidate's resume. Focus on practical, real-world technical scenarios that would validate the candidate's claimed expertise. Do not include behavioral questions - only technical questions based on resume content. For each question, provide a comprehensive example response that demonstrates the expected level of technical detail and professional communication.""",
            agent=agents[1],
            expected_output="A comprehensive technical interview preparation guide with 5 resume-based questions, detailed answer guidance, and example responses for each question."
        ),
        Task(
            description="""Provide detailed resume improvement suggestions with professional formatting. Structure your response as follows:\n\n## 💡 RESUME OPTIMIZATION GUIDE\n...\nProvide specific examples and before/after comparisons where applicable.""",
            agent=agents[2],
            expected_output="A comprehensive resume optimization guide with prioritized improvements, specific examples, and actionable checklists."
        ),
        Task(
            description="""Evaluate job fit with detailed scoring and professional formatting. Structure your response as follows:\n\n## ⭐ JOB FIT ASSESSMENT REPORT\n...\n**Job Fit Score: [X]**""",
            agent=agents[3],
            expected_output="A comprehensive job fit assessment with detailed scoring, competitive analysis, and clear hiring recommendations."
        )
    ]
    return tasks


def get_task_from_query(query: str):
    """Route a natural language query to a task index.

    Modes:
      - If a vector router is configured, use semantic similarity against task prototypes.
      - Else, fallback keyword mapping.
    Returns: (task_index, intent_label)
    """
    if not query:
        return 0, "analysis"

    if _VECTOR_ROUTER is not None:
        try:
            match = _VECTOR_ROUTER.route(query)
            if match:
                return match['index'], match['label']
        except Exception:
            pass  # silent fallback

    # Fallback keyword logic
    q = query.lower()
    def has(word):
        return re.search(rf"\b{re.escape(word)}\b", q) is not None
    if any(has(w) for w in ["interview", "question", "questions", "technical"]):
        return 1, "interview"
    if any(has(w) for w in ["resume", "improve", "optimiz", "enhance"]):
        return 2, "suggestions"
    if any(has(w) for w in ["fit", "score", "match", "suitability"]):
        return 3, "job_fit"
    return 0, "analysis"

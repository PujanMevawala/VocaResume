from crewai import Task
import re

def create_tasks(agents):
    tasks = [
        Task(description="""Analyze the provided resume against the job description with professional formatting. Structure your response as follows:\n\n## 📊 RESUME ANALYSIS REPORT\n...\nUse bullet points, clear headers, and professional language throughout.""", agent=agents[0], expected_output="A professionally formatted resume analysis report with clear sections, ratings, and actionable recommendations."),
        Task(description="""Generate 5 resume-based technical interview questions with detailed answer guidance and example responses. Structure your response as follows:\n\n## 🎯 TECHNICAL INTERVIEW PREPARATION GUIDE\n...\nGenerate exactly 5 questions specifically based on the technical skills, tools, technologies, and experiences mentioned in the candidate's resume. Focus on practical, real-world technical scenarios that would validate the candidate's claimed expertise. Do not include behavioral questions - only technical questions based on resume content. For each question, provide a comprehensive example response that demonstrates the expected level of technical detail and professional communication.""", agent=agents[1], expected_output="A comprehensive technical interview preparation guide with 5 resume-based questions, detailed answer guidance, and example responses for each question."),
        Task(description="""Provide detailed resume improvement suggestions with professional formatting. Structure your response as follows:\n\n## 💡 RESUME OPTIMIZATION GUIDE\n...\nProvide specific examples and before/after comparisons where applicable.""", agent=agents[2], expected_output="A comprehensive resume optimization guide with prioritized improvements, specific examples, and actionable checklists."),
        Task(description="""Evaluate job fit with detailed scoring and professional formatting. Structure your response as follows:\n\n## ⭐ JOB FIT ASSESSMENT REPORT\n...\n**Job Fit Score: [X]**""", agent=agents[3], expected_output="A comprehensive job fit assessment with detailed scoring, competitive analysis, and clear hiring recommendations.")
    ]
    return tasks


def get_task_from_query(query: str):
    if not query:
        return 0, "analysis"
    q = query.lower()
    def has(word):
        return re.search(rf"\b{re.escape(word)}\b", q) is not None
    if any(has(w) for w in ["resume", "improve"]):
        return 2, "suggestions"
    if any(has(w) for w in ["fit", "score"]):
        return 3, "job_fit"
    if any(has(w) for w in ["interview", "questions"]):
        return 1, "interview"
    return 0, "analysis"

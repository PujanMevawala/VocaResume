# CrewAI agent creation logic
from crewai import Agent
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

def create_agents(model_info):
    """Creates all the agents for the crew, configured with the selected LLM."""
    
    provider = model_info.get('provider')
    model_name = model_info.get('model')
    
    # Select the appropriate LLM based on the provider
    if provider == 'google':
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=settings.GOOGLE_API_KEY, temperature=0.4)
    elif provider == 'groq':
        llm = ChatGroq(model_name=model_name, api_key=settings.GROQ_API_KEY, temperature=0.4)
    elif provider == 'perplexity':
        llm = ChatOpenAI(
            base_url="https://api.perplexity.ai",
            api_key=settings.PPLX_API_KEY,
            model=model_name,
            temperature=0.4
        )
    else:
        raise ValueError(f"Unsupported or missing provider in model_info: {provider}")

    # New Parsing Agent
    document_parser = Agent(
        role="Document Parser",
        goal="Parse resumes and job descriptions into a structured JSON format",
        backstory="A meticulous data extraction specialist who excels at converting unstructured text into clean, organized data.",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )
    
    resume_analyzer = Agent(
        role="Resume Analyzer",
        goal="Analyze the resume and provide a detailed assessment",
        backstory="An experienced HR manager skilled in resume evaluation.",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )
    interview_preparer = Agent(
        role="Interview Preparer",
        goal="Generate resume-based technical interview questions with detailed answer guidance",
        backstory="An expert in preparing candidates for job interviews.",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )
    suggestion_generator = Agent(
        role="Resume Suggestion Generator",
        goal="Provide suggestions to improve the resume",
        backstory="A resume writing expert.",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )
    job_fit_scorer = Agent(
        role="Job Fit Scorer",
        goal="Score the resume against the job description",
        backstory="A data-driven recruitment analyst.",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )
    
    return [document_parser, resume_analyzer, interview_preparer, suggestion_generator, job_fit_scorer]

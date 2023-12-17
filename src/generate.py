from pydantic import BaseModel, Field
from typing import List
from jinja2 import Template
import instructor
from openai import OpenAI

from src.config import OPENAI_MODEL

class Question(BaseModel):
    """Correctly resolved question from the given text"""
    id: int
    question: str
    chain_of_thought: str = Field(..., description="Think step by step to elaborate the question")

class SurveyQuestions(BaseModel):
    """Correctly resolved set of survey questions from the given text"""
    questions: List[Question] = Field(..., description="List of survey questions inspired by the given text, should be 3 or less")

QUESTION_GENERATION_SYSTEM_PROMPT = """
The following text is going to be augmented with insights derived from survey data.
Your task is to create survey questions that will help gather these insights.
Please formulate questions that are relevant to the text and suitable for a broad audience.
Ensure that your questions are concise and clear.

Example:

Text: "Quentin Tarantino enjoys eating apples and driving electric cars."

Survey Questions:

"What is your favorite fruit?"
"Are you interested in purchasing an electric car?"
"Do you appreciate Quentin Tarantino's movies?"
"""

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.patch(OpenAI())

def generate_questions(data: str, model: str = OPENAI_MODEL) -> SurveyQuestions:
    return client.chat.completions.create(
        model=model,
        response_model=SurveyQuestions,
        messages=[
            {
                "role": "system",
                "content": QUESTION_GENERATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Create the survey questions inspired by the following text: {data}",
            },
        ],
    )

OUTPUT_GENERATION_SYSTEM_PROMPT = """
The following text is going to be augmented with insights derived from survey data.
Your task is to rewrite the text to include the insights.
Use the survey questions as a guide to what insights are relevant to the text.
Use the writing style of fivethirtyeight.com as a reference.
Ensure that the text is concise and clear.
"""

OUTPUT_GENERATION_USER_PROMPT = Template(
"""
Rewrite the following text to include insights derived from survey data.
Consider the survey questions as a guide for which insights are relevant to include in the rewritten text.
Use the writing style of fivethirtyeight.com, aiming for a data-driven, conversational tone.
------
Original Text:
{{text}}
------
Survey Data for Augmentation:
{{context}}
------
Rewritten Text: 
"""
)

def rewrite_text(text: str, context=str, model: str = OPENAI_MODEL) -> str:
    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": OUTPUT_GENERATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": OUTPUT_GENERATION_USER_PROMPT.render(text=text, context=context)
            },
        ],
    ).choices[0].message.content
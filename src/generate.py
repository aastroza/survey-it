from pydantic import BaseModel, Field
from typing import List
import instructor
from openai import OpenAI

class Question(BaseModel):
    """Correctly resolved question from the given text"""
    id: int
    question: str
    chain_of_thought: str = Field(..., description="Think step by step to elaborate the question")

class SurveyQuestions(BaseModel):
    """Correctly resolved set of survey questions from the given text"""
    questions: List[Question] = Field(..., description="List of survey questions inspired by the given text, should be 3 or less")

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.patch(OpenAI())

system_prompt = """
The following text is to be augmented with insights from survey data. Please consider survey questions that can be asked to a broad audience. Be concise and clear.

Example:

Text: "Quentin Tarantino likes to eat apples and drive electric cars."
Question 1: "Do you like to eat apples?"
Question 2: "Do you like to drive electric cars?"
Question 3: "Do you support Quentin Tarantino?"
"""
def generate_questions(data: str) -> SurveyQuestions:
    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=SurveyQuestions,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Create the survey questions inspired by the following text: {data}",
            },
        ],
    )
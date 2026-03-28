EVAL_PROMPT = ( "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:" 
        "------" 
        "##INSTRUCTIONS: "
        "- Focus on meaningful matches: Assess whether the predicted answer and the correct answer have a meaningful match, not just literal word-for-word matches.\n" 
        "- Criteria for Correctness: The predicted answer is considered correct if it reasonably matches the standard answer, recognizing that synonyms or varied expressions that convey the same or similar meaning are acceptable.\n" 
        "- If the predicted answer's yes/no conclusion conflicts with the correct answer, it is incorrect.\n" 
        "- The predicted answer is considered correct if it contains the core descriptive information and does not contradict the correct answer, even if some non-critical details are missing.\n" 
        "- Flexibility in Evaluation: Use judgment to decide if variations in the predicted answer still correctly address the question, even if they do not directly replicate the correct answer's phrasing.\n"
       )

EVAL_INS = (
                "Provide your evaluation result only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match.\n"
                "Please generate the response in the form of a valid JSON string with keys 'pred' and 'score'.\n"
                "For example: {\"pred\": \"yes\", \"score\": 5}, {\"pred\": \"no\", \"score\": 1}."
            )

SYSTEM_PROMPT = "You are a first-person AI assistant integrated into a head-mounted camera. Your primary mission is to answer questions from the user (the camera wearer) about their own actions, objects, and environment as seen through your lens. \n\nKey Instructions:\n1. First-Person Context: All references to 'I', 'me', or 'my' are about the user.\n2. Strict Grounding: Base your answers exclusively on the provided video content."

OPEN_ENDED_SYSTEM_PROMPT = """You are a first-person AI assistant integrated into a head-mounted camera. Your primary mission is to answer questions from the user (the camera wearer) about their own actions, objects, and environment as seen through your lens. You should answer directly to the user. The question is asked at the moment of the last frame in the video.

Key Instructions:
1. First-Person Context: All references to 'I', 'me', or 'my' are about the user. You must distinguish between the user's actions/objects and those of other people visible in the video.
2. Grounding: Base your answers primarily on the visual evidence in the video. If the information is not present or cannot be reasonably inferred from the video, state that it is not shown.
3. For any question that requires a 'Yes' or 'No' response, you MUST follow it with a brief explanation for your reasoning.
4. All responses must be direct and clear."""

MC_PROMPT_TEMPLATE = """{question}
Options:
{options}
There is only one correct option. Please only response with the letter of the correct option."""


def build_mc_prompt(question: str, options: list, prompt_type: str = "default") -> str:
    options_str = "\n".join([opt for opt in options])
    return MC_PROMPT_TEMPLATE.format(question=question, options=options_str)


def build_oe_prompt(question: str) -> str:
    return question

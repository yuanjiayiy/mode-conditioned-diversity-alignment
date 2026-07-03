PIF_SYSTEM_PROMPT = """You are a helpful AI Assistant designed to provide well-reasoned and detailed responses. If the task involves probabilistic or non-
deterministic reasoning, you must begin by generating a unique and complex random string to serve as a seed. This random string should appear sufficiently complex and unpredictable, with no obvious structure or pattern. Use your judgment to ensure it looks arbitrary and unguessable. If the user explicitly instructs you to sample from a probability
distribution, use the generated seed (the exact contents inside the `<random_string>` tags) to guide any random sampling or stochastic decisions.
Follow these two steps for every instruction:
1. Output the random seed string enclosed within`<random_string>` and`</random_string>` tags.
2. Think deeply and carefully about the user's question, and enclose this reasoning within`<thinking>` and`</thinking>` tags. All probabilistic decisions must be made using the generated seed- the exact contents inside the`<random_string>` tags. Make sure to extract maximum randomness from the string by using all of its content.
3. Provide your final answer, enclosed within`<answer>` and`</answer>` tags. Strictly follow this tag structure, and respond in the following

format:
<random_string>
...
</random_string>
<thinking>
...
</thinking>
<answer>
...
</answer>"""

BASELINE_SYSTEM_PROMPT = """You are a helpful AI Assistant designed to provide well-reasoned and detailed responses. If the user explicitly instructs you to sample from a probability distribution, do stochastic decisions based on the user provided data.
Think deeply and carefully about the user's question, and enclose this reasoning within `<thinking>` and `</thinking>` tags. Then provide your final answer, enclosed within `<answer>` and `</answer>` tags.
Strictly follow this tag structure, and respond in the following format:
<thinking>
...
</thinking>
<answer>
...
</answer>
"""

PIF_SYSTEM_PROMPT_NO_THINKING = """You are a helpful AI Assistant designed to provide well-reasoned and detailed responses. If the task involves probabilistic or non-
deterministic reasoning, you must begin by generating a unique and complex random string to serve as a seed. This random string should appear sufficiently complex and unpredictable, with no obvious structure or pattern. Use your judgment to ensure it looks arbitrary and unguessable. If the user explicitly instructs you to sample from a probability
distribution, use the generated seed (the exact contents inside the `<random_string>` tags) to guide any random sampling or stochastic decisions.
Follow these two steps for every instruction:
1. Output the random seed string enclosed within`<random_string>` and`</random_string>` tags.
2. Provide your final answer, enclosed within`<answer>` and`</answer>` tags. Strictly follow this tag structure, and respond in the following

format:
<random_string>
...
</random_string>
<answer>
...
</answer>"""


MULTI_ANSWER_RL_THINKING_PROMPT = (
    "Output EXACTLY {K} DISTINCT answers.\n"
    "FORMAT ONLY (no extra text):\n"
    "<think> reasoning about different possible candidates and their justification </think>\n"
    "<answer1> candidate_1 </answer1>\n"
    "<answer2> candidate_2 </answer2>\n"
    "... exactly {K} answers ...\n"
    f"IMPORTANT: Each <answer{{{{i}}}}> </answer{{{{i}}}}> tag must contain ONLY the candidate answer. Do NOT write a full sentence in <answer{{{{i}}}}>. Do NOT restate the question in <answer{{{{i}}}}>. If any extra words are included, the answer is incorrect."
)

MULTI_ANSWER_RL_NO_THINKING_PROMPT = (
    "Output EXACTLY {K} DISTINCT answers.\n"
    "FORMAT ONLY (no extra text):\n"
    "<answer1> candidate_1 </answer1>\n"
    "<answer2> candidate_2 </answer2>\n"
    "... exactly {K} answers ...\n"
    f"IMPORTANT: Each <answer{{{{i}}}}> </answer{{{{i}}}}> tag must contain ONLY the candidate answer. Do NOT write a full sentence in <answer{{{{i}}}}>. Do NOT restate the question in <answer{{{{i}}}}>. If any extra words are included, the answer is incorrect. The answer will be graded on exact match with a ground truth answer."
)
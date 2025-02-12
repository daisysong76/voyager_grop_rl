# prompt_optimizer.py
# TODO 2: # Optimizes prompts iteratively based on feedback
# Create a Prompt Optimization Module:

from dspy import DSPyPromptOptimizer

class VoyagerPromptOptimizer:
    def __init__(self, model_name, temperature, max_iterations):
        self.prompt_optimizer = DSPyPromptOptimizer(
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
        )

    def optimize_prompt(self, messages):
        return self.prompt_optimizer.optimize(messages)
    


#     # utils/prompt_optimizer_dspy.py

# def optimize_prompt(prompt, feedback):
#     # Adjust the prompt based on feedback to improve agent understanding or success
#     if "insufficient materials" in feedback:
#         return prompt + " (ensure enough resources are gathered)"
#     return prompt

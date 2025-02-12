# data_generator.py
import json

class DataGenerator:
    def __init__(self, curriculum_agent, last_events):
        self.curriculum_agent = curriculum_agent
        self.last_events = last_events

    def generate_synthetic_data(self, num_samples=100):
        synthetic_data = []
        for _ in range(num_samples):
            task, context = self.curriculum_agent.propose_next_task(self.last_events)
            synthetic_data.append({
                "task": task,
                "context": context,
                "expected_output": "Ideal response based on domain knowledge"
            })
        # Save to file
        with open("synthetic_data.json", "w") as f:
            json.dump(synthetic_data, f)

        print(f"Generated {num_samples} synthetic data samples for fine-tuning.")

#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from meta_agents_007.crew import MetaAgents007

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    while True:
        user_input = input("Enter a topic(or press CTRL + C to quit): ")
        inputs = {
            'topic': user_input,
        }
        try:
            result = MetaAgents007().crew().kickoff(inputs=inputs)
            for i, task_output in enumerate(result.tasks_output):
                print(f"Tasks {i} output: {task_output.raw}")
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")

def train():
    """
    Train the crew for a given number of iterations.
    """
    while True:
        user_input = input("Enter a topic(or press CTRL + C to quit): ")
        inputs = {
            'topic': user_input,
        }
        try:
            MetaAgents007().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

        except Exception as e:
            raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MetaAgents007().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    while True:
        user_input = input("Enter a topic(or press CTRL + C to quit): ")
        inputs = {
            'topic': user_input,
        }
        try:
            MetaAgents007().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

        except Exception as e:
            raise Exception(f"An error occurred while testing the crew: {e}")

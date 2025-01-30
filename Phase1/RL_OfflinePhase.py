import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# UUniFast function to generate hard tasks
def uunifast(num_tasks, total_utilization):
    utilizations = []
    sum_u = total_utilization

    for i in range(1, num_tasks):
        next_u = sum_u * (random.uniform(0, 1) ** (1 / (num_tasks - i)))
        utilizations.append(sum_u - next_u)
        sum_u = next_u

    utilizations.append(sum_u)
    return utilizations

def generate_tasks(num_tasks, total_utilization, min_period, max_period):
    utilizations = uunifast(num_tasks, total_utilization)
    tasks = []

    for i, utilization in enumerate(utilizations):
        period = random.randint(min_period, max_period)
        wcet = utilization * period
        task = {
            "task_id": i,
            "utilization": round(utilization, 4),
            "period": period,
            "wcet": round(wcet, 4),
            "deadline": period
        }
        tasks.append(task)

    return tasks

# RL Environment for task-to-core mapping with DVFS
class DVFSEnvironment:
    def __init__(self, num_cores, frequencies, tasks):
        self.num_cores = num_cores
        self.frequencies = frequencies
        self.tasks = tasks
        self.state = self._initialize_state()
        self.energy_consumption = 0

    def _initialize_state(self):
        return {
            "cores": [{"load": 0, "frequency": self.frequencies[0]} for _ in range(self.num_cores)],
            "task_queue": self.tasks.copy()
        }

    def step(self, action):
        task_id, core_id, freq_id = action

        # Ensure task_id is within the current task queue
        if task_id >= len(self.state["task_queue"]):
            raise IndexError("Task ID out of range for current task queue.")

        task = self.state["task_queue"][task_id]
        core = self.state["cores"][core_id]
        frequency = self.frequencies[freq_id]

        core["load"] += task["wcet"] / frequency
        core["frequency"] = frequency

        energy = frequency * task["wcet"]
        self.energy_consumption += energy

        reward = -energy
        if core["load"] > task["deadline"]:
            reward -= 100

        self.state["task_queue"].pop(task_id)
        done = len(self.state["task_queue"]) == 0

        return self.state, reward, done

    def reset(self):
        self.state = self._initialize_state()
        self.energy_consumption = 0
        return self.state

# Convert state to vector
def state_to_vector(state, num_cores, frequencies, num_tasks):
    cores_info = []
    for core in state["cores"]:
        cores_info.append(core["load"])
        cores_info.append(core["frequency"])

    task_info = []
    for task in state["task_queue"]:
        task_info.append(task["wcet"])
        task_info.append(task["deadline"])

    # Pad task_info to ensure it has 2 * num_tasks elements
    while len(task_info) < 2 * num_tasks:
        task_info.append(0)  # Add padding for WCET and deadline

    return cores_info + task_info

# Deep Q-Network for RL agent
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Main function to train and evaluate RL agent
def train_rl(num_tasks, total_utilization, num_cores, frequencies, min_period, max_period, episodes):
    tasks = generate_tasks(num_tasks, total_utilization, min_period, max_period)
    env = DVFSEnvironment(num_cores, frequencies, tasks)

    state_dim = (2 * num_cores) + (2 * num_tasks)
    action_dim = num_tasks * num_cores * len(frequencies)

    agent = DQN(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    gamma = 0.99  # Discount factor
    final_assignments = {core_id: [] for core_id in range(num_cores)}  # For tracking assignments
    task_to_core = {task["task_id"]: None for task in tasks}  # Map tasks to cores

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_vector = torch.tensor(
                [state_to_vector(state, num_cores, frequencies, num_tasks)], dtype=torch.float32
            )
            q_values = agent(state_vector)

            # Apply masking to valid actions
            valid_actions = []
            for task_id in range(len(state["task_queue"])):
                for core_id in range(num_cores):
                    for freq_id in range(len(frequencies)):
                        valid_actions.append((task_id, core_id, freq_id))

            valid_action_indices = [
                (task_id * num_cores * len(frequencies)) + (core_id * len(frequencies)) + freq_id
                for task_id, core_id, freq_id in valid_actions
            ]

            valid_q_values = q_values[0, valid_action_indices]
            best_action_index = torch.argmax(valid_q_values).item()

            task_id, core_id, freq_id = valid_actions[best_action_index]

            next_state, reward, done = env.step((task_id, core_id, freq_id))

            # Update final assignments for last episode
            if episode == episodes - 1:
                if task_to_core[tasks[task_id]["task_id"]] is None:
                    final_assignments[core_id].append(tasks[task_id])
                    task_to_core[tasks[task_id]["task_id"]] = core_id

            next_state_vector = torch.tensor(
                [state_to_vector(next_state, num_cores, frequencies, num_tasks)], dtype=torch.float32
            )

            target = reward + gamma * torch.max(agent(next_state_vector)) * (not done)
            loss = criterion(q_values[0, valid_action_indices[best_action_index]], target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Energy Consumption: {env.energy_consumption}")

    # Assign unassigned tasks to the least utilized cores
    for task_id, core in task_to_core.items():
        if core is None:
            least_utilized_core = min(final_assignments.keys(), key=lambda x: sum(task["utilization"] for task in final_assignments[x]))
            task = next(task for task in tasks if task["task_id"] == task_id)
            final_assignments[least_utilized_core].append(task)

    # Sort tasks on each core by EDF (Earliest Deadline First)
    for core_id in final_assignments:
        final_assignments[core_id].sort(key=lambda x: x["deadline"])

    # Calculate and display final core utilizations and schedules
    print("\nFinal Task Assignments and Schedules (EDF):")
    core_utilizations = [0] * num_cores
    for core_id, assigned_tasks in final_assignments.items():
        print(f"Core {core_id} (sorted by deadline):")
        current_time = 0  # Track the current time on each core
        for task in assigned_tasks:
            start_time = current_time
            finish_time = start_time + task["wcet"]
            print(f"  Task {task['task_id']} -> Start: {start_time:.2f}, Finish: {finish_time:.2f}, Deadline: {task['deadline']}, Utilization: {task['utilization']}, WCET: {task['wcet']}")
            current_time = finish_time
            core_utilizations[core_id] += task["utilization"]

    print("\nCore Utilizations:")
    for core_id, utilization in enumerate(core_utilizations):
        status = "Valid" if utilization <= 1.0 else "Invalid"
        print(f"Core {core_id}: Utilization = {utilization:.2f} ({status})")


# Example usage
if __name__ == "__main__":
    num_tasks = 50
    total_utilization = 0.95
    num_cores = 3
    frequencies = [0.5, 1.0, 1.3]
    min_period = 1
    max_period = 5
    episodes = 100

    train_rl(num_tasks, total_utilization, num_cores, frequencies, min_period, max_period, episodes)

import copy
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def uunifast(num_tasks, total_utilization):
    utilizations = []
    sum_u = total_utilization

    for i in range(1, num_tasks):
        next_u = sum_u * (random.uniform(0, 1) ** (1 / (num_tasks - i)))
        utilizations.append(sum_u - next_u)
        sum_u = next_u

    utilizations.append(sum_u)
    return utilizations


def generate_tasks(num_tasks, total_utilization, min_period, max_period, num_aperiodic, max_arrival_time):
    utilizations = uunifast(num_tasks - num_aperiodic, total_utilization)  # Adjust for the aperiodic tasks
    tasks = []

    # Generate periodic tasks
    for i, utilization in enumerate(utilizations):
        period = random.randint(1, min_period * 2)
        wcet = utilization * period
        task = {
            "task_id": i,
            "utilization": round(utilization, 4),
            "period": period,
            "wcet": round(wcet, 4),
            "deadline": period,
            "arrival_time": 0,
            "type": "hard"
        }
        tasks.append(task)

    tasks_len = len(tasks)
    # Generate soft aperiodic tasks
    for i in range(num_aperiodic):
        utilization = random.uniform(0, 0.3)  # Random utilization for aperiodic task
        wcet = utilization * max_period  # Soft aperiodic task doesn't have a fixed period, use max_period as a base
        arrival_time = random.randint(0, max_arrival_time)  # Random arrival time
        deadline = random.randint(arrival_time + 1, max_arrival_time + 1)  # Random deadline after arrival

        aperiodic_task = {
            "task_id": tasks_len + i,
            "utilization": round(utilization, 4),
            "wcet": round(wcet / 10, 4),
            "arrival_time": arrival_time,
            "deadline": deadline,
            "type": "soft"
        }
        tasks.append(aperiodic_task)

    return tasks


# RL Environment for task-to-core mapping with DVFS
class DVFSEnvironment:
    def __init__(self, num_cores, frequencies, voltages, tasks, tdp):
        self.num_cores = num_cores
        self.frequencies = frequencies
        self.voltages = voltages
        self.tasks = tasks
        self.state = self._initialize_state()
        self.energy_consumption = 0
        self.tdp = tdp
        self.slack_time = [0] * num_cores  # Track available slack time per core

    def _initialize_state(self):
        return {
            "cores": [{"load": 0, "frequency": self.frequencies[0], "slack_time": 0} for _ in range(self.num_cores)],
            "task_queue": self.tasks.copy()
        }

    def calculate_slack_time(self):
        for core in self.state["cores"]:
            core["slack_time"] = core["frequency"] - core["load"]  # Remaining available time on core

    def step(self, action):
        task_id, core_id, freq_id = action

        if task_id >= len(self.state["task_queue"]):
            raise IndexError("Task ID out of range for current task queue.")

        task = self.state["task_queue"][task_id]
        core = self.state["cores"][core_id]
        frequency = self.frequencies[freq_id]
        voltage = self.voltages[freq_id]

        wcet = task["wcet"] / frequency
        task["frequency"] = frequency
        task["voltage"] = voltage
        core["load"] += wcet / frequency
        core["frequency"] = frequency

        energy = frequency ** 2 * voltage ** 2 * wcet
        self.energy_consumption += energy

        reward = -energy
        if self.energy_consumption > self.tdp:
            reward -= 1000
            self.energy_consumption -= energy
        elif core["load"] > task["deadline"]:
            reward -= 100
        else:
            task["wcet"] = wcet

        if task["type"] == "soft":
            self.calculate_slack_time()
            if task["arrival_time"] > core["load"] or task["wcet"] > core["slack_time"]:
                reward -= 500
            else:
                core["slack_time"] -= task["wcet"]

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
def train_rl(num_tasks, total_utilization, num_cores, frequencies, voltages, min_period, max_period, episodes, tdp, num_aperiodic, max_arrival_time, x_parameter):
    tasks = generate_tasks(num_tasks, total_utilization, min_period, max_period, num_aperiodic, max_arrival_time)
    env = DVFSEnvironment(num_cores, frequencies, voltages, tasks, tdp)

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

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Total Power: {env.energy_consumption:.2f}")

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
    powers = []
    core_powers = {core: 0 for core in range(num_cores)}
    all_powers = []
    quality_of_services = {}
    with open(f"results/scheduling_cores_{num_cores}_utilization_{total_utilization}.txt", "w") as file:
        for core_id, assigned_tasks in final_assignments.items():
            file.write(f"Core {core_id} (sorted by deadline):\n")
            current_time = 0  # Track the current time on each core
            total_power = 0
            for task in assigned_tasks:
                start_time = max(current_time, task["arrival_time"]) if task["type"] == "soft" else current_time
                finish_time = start_time + task["wcet"]
                if task["type"] == "soft":
                    file.write(
                        f"  Task {task['task_id']} -> Type: {task['type']}, Arrival: {task['arrival_time']}, Start: {start_time:.2f}, Finish: {finish_time:.2f}, Deadline: {task['deadline']}\n")
                else:
                    file.write(
                        f"  Task {task['task_id']} -> Type: {task['type']}, Start: {start_time:.2f}, Finish: {finish_time:.2f}, Deadline: {task['deadline']}\n")

                if x_parameter * task['deadline'] >= finish_time > task['deadline']:
                    quality_of_services[task['task_id']] = ((task['deadline'] - finish_time) / (task['deadline'] * (x_parameter - 1))) + 1
                elif finish_time > x_parameter * task['deadline']:
                    quality_of_services[task['task_id']] = 0
                else:
                    quality_of_services[task['task_id']] = 1

                current_time = finish_time
                core_utilizations[core_id] += task["utilization"]
                total_power += task["frequency"] ** 2 * task["voltage"] ** 2 * task['wcet']

                core_powers[core_id] += task['frequency'] ** 2 * task['voltage'] ** 2 * task['wcet']
                all_powers.append(copy.deepcopy(core_powers))



            powers.append(total_power)

    task_ids = list(quality_of_services.keys())
    values = list(quality_of_services.values())
    mean_qos = np.mean(values)

    plot_qos_tasks(num_aperiodic, num_cores, num_tasks, task_ids, total_utilization, values)

    plot_power_per_time(all_powers, num_cores, total_utilization)

    print("\nCore Utilizations:")
    utils = []
    for core_id, utilization in enumerate(core_utilizations):
        status = "Valid" if utilization <= total_utilization else "Invalid"
        print(f"Core {core_id}: Utilization = {utilization:.2f} ({status})")
        utils.append(utilization)

    plot_utilization_of_cores(num_cores, total_utilization, utils)
    return tasks, powers, mean_qos


def plot_power_per_time(all_powers, num_cores, total_utilization):
    time_steps = list(range(len(all_powers)))
    core_values = {i: [entry[i] for entry in all_powers] for i in range(num_cores)}
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / num_cores) for i in range(num_cores)]
    for core in range(num_cores):
        plt.plot(time_steps, core_values[core], linestyle='-', color=colors[core], label=f'Core {core}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Core Values Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/output4_core_{num_cores}_utilization_{total_utilization}.png")
    plt.close()


def plot_qos_tasks(num_aperiodic, num_cores, num_tasks, task_ids, total_utilization, values):
    x_labels = [f"T {i}" for i in task_ids]
    colors = ['r' if task_id >= (num_tasks - num_aperiodic) else 'b' for task_id in task_ids]
    plt.bar(task_ids, values, color=colors, alpha=0.7)
    blue_patch = mpatches.Patch(color='b', label='Hard Tasks')
    red_patch = mpatches.Patch(color='r', label='Soft Tasks')
    plt.legend(handles=[blue_patch, red_patch], loc='upper right')
    plt.xlabel("Tasks")
    plt.ylabel("Qos")
    plt.title("QoS of tasks")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"results/output1_core_{num_cores}_utilization_{total_utilization}.png")
    plt.close()


def plot_utilization_of_cores(num_cores, total_utilization, utils):
    core_ids = list(range(0, num_cores))
    plt.figure(figsize=(8, 4))
    plt.bar(core_ids, utils, color='g', alpha=0.7)
    plt.xlabel("Cores")
    plt.ylabel("Utilization")
    plt.title("Utilizations of cores")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"results/output7_core_{num_cores}_utilization_{total_utilization}.png")
    plt.close()


def output_tasks_table_csv(tasks):
    df = pd.DataFrame(tasks)
    df.set_index('task_id', inplace=True)
    df.index = [f'Task {i}' for i in df.index]
    df["wcet"] = round(df["wcet"], 4)
    df.to_csv(f'results/output6.csv')


def plot_power_cores(num_cores, total_utilization, powers_dict):
    x_labels = [f'Core {i}' for i in range(num_cores)]
    plt.figure(figsize=(12, 6))
    for utilization in total_utilization:
        plt.plot(x_labels, powers_dict[num_cores][utilization], marker='o', label=f'Util {utilization}')
    plt.title('Powers')
    plt.ylabel('Power')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"results/output3_with_{num_cores}_cores.png")
    plt.close()


def parse_schedule(file_path):
    core_tasks = {}
    max_hard_finish = 0

    with open(file_path, 'r') as file:
        current_core = None
        for line in file:
            core_match = re.match(r'Core (\d+)', line)
            task_match = re.match(
                r'\s*Task (\d+) -> Type: (\w+), (?:Arrival: (\d+), )?Start: ([\d\.]+), Finish: ([\d\.]+), Deadline: (\d+)',
                line)

            if core_match:
                current_core = int(core_match.group(1))
                core_tasks[current_core] = []
            elif task_match and current_core is not None:
                task_id = int(task_match.group(1))
                task_type = task_match.group(2)
                start = float(task_match.group(4))
                finish = float(task_match.group(5))

                if task_type == 'hard' and finish > max_hard_finish:
                    max_hard_finish = finish

                core_tasks[current_core].append({
                    'task_id': task_id, 'start': start, 'finish': finish, 'type': task_type
                })

    return core_tasks, max_hard_finish


def scale_time(duration):
    if duration > 0.1:
        return duration
    elif 0.01 <= duration <= 0.1:
        return duration * 10
    elif duration < 0.001:
        return duration * 100
    return duration


def plot_gantt_chart(num_cores, total_utilization, core_tasks, max_hard_finish):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {'hard': 'blue', 'soft': 'red'}  # رنگ‌های جداگانه برای hard و soft

    for core, tasks in core_tasks.items():
        for task in tasks:
            duration = task['finish'] - task['start']
            scaled_duration = scale_time(duration)

            ax.barh(y=core, width=scaled_duration, left=task['start'], height=0.4,
                    color=colors[task['type']], edgecolor='black')
            ax.text(task['start'] + scaled_duration / 2, core,
                    f'T{task["task_id"]}', ha='center', va='center', color='white', fontsize=10, fontweight='bold')

    ax.set_xlim(0, max_hard_finish)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cores')
    ax.set_yticks(list(core_tasks.keys()))
    ax.set_yticklabels([f'Core {core}' for core in core_tasks.keys()])
    ax.set_title('Task Scheduling Gantt Chart')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(f"results/output5_cores_{num_cores}_utilization_{total_utilization}.png")
    plt.close()


def plot_scheduling(num_cores, total_utilization):
    file_path = f'results/scheduling_cores_{num_cores}_utilization_{total_utilization}.txt'
    core_tasks, max_hard_finish = parse_schedule(file_path)
    plot_gantt_chart(num_cores, total_utilization, core_tasks, max_hard_finish)


def plot_system_quality_of_service(core, total_utilization, mean_qos_cores):
    plt.figure(figsize=(8, 5))
    plt.bar(total_utilization, mean_qos_cores, width=0.1, color='blue', alpha=0.7)
    plt.xlabel("Utilization")
    plt.ylabel("System QoS")
    plt.title("System QoS per Utilization")
    plt.xticks(total_utilization)
    plt.ylim(0, 1)
    plt.savefig(f"results/output2_core_{core}.png")
    plt.close()


if __name__ == "__main__":
    num_tasks = 40
    num_aperiodic = 10
    max_arrival_time = 10
    total_utilization = [0.25, 0.5, 0.75, 1.0]
    num_cores = [8, 16, 32]
    frequencies = [0.75, 1.0, 1.1]
    voltages = [0.9, 1.1, 1.5]
    tdp = 20
    min_period = 3
    max_period = 10
    episodes = 100
    x_parameter = 1.1

    powers_dict = {core: {} for core in num_cores}
    for core in num_cores:
        mean_qos_cores = []
        for utilization in total_utilization:
            tasks, powers, mean_qos = train_rl(num_tasks, utilization, core, frequencies, voltages, min_period, max_period, episodes, tdp, num_aperiodic, max_arrival_time, x_parameter)
            mean_qos_cores.append(mean_qos)
            powers_dict[core][utilization] = powers
            plot_scheduling(core, utilization)
        plot_power_cores(core, total_utilization, powers_dict)

        plot_system_quality_of_service(core, total_utilization, mean_qos_cores)

    output_tasks_table_csv(tasks)


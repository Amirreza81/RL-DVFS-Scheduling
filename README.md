# 🚀 **Power Management & Task Scheduling in Real-Time Embedded Systems**  

## **Project Overview**  
This project focuses on **optimizing energy consumption** ⚡ in embedded systems with limited resources, such as battery-powered devices. We utilize **reinforcement learning (RL)** 🧠 and **dynamic voltage & frequency scaling (DVFS)** 🔄 to efficiently manage **task scheduling**, ensuring system constraints are met.  

---

## 🏗 **Project Phases**  

### **🟢 Phase 1 - Offline Scheduling**  
📌 Designed an **RL-based model** to optimize task scheduling on multi-core homogeneous systems.  
📌 Applied **DVFS** to dynamically adjust **voltage & frequency** based on workload.  
📌 Ensured **Thermal Design Power (TDP)** constraints were not violated.  
📌 Used **Earliest Deadline First (EDF)** scheduling for real-time tasks.  

### **🔵 Phase 2 - Online Scheduling**  
📌 Integrated **soft aperiodic tasks** dynamically into the system.  
📌 Developed an **adaptive task mapping strategy** 🗺 for real-time & aperiodic tasks.  
📌 Optimized **Quality of Service (QoS)** while maintaining system constraints.  
📌 Utilized **Slack Stealing Server** to efficiently handle task scheduling.  

---

## 📊 **Results & Deliverables**  
✅ Analysis of **task schedulability** under various workloads.  
✅ Power consumption **trends for each core**.  
✅ QoS **evaluation across different system states**.  
✅ Final **implementation & report**.  

---

## 🖼 **Visuals & Graphs**

<p align="center">
  <img src="/results/output1_core_8_utilization_1.0.png" width="650">
  <img src="/results/output2_core_8.png" width="650">
</p>
<p align="center">
  <img src="/results/output3_with_8_cores.png" width="650">
  <img src="/results/output4_core_16_utilization_1.0.png" width="650">
</p>
<p align="center">
  <img src="/results/output5_cores_8_utilization_0.75.png" width="650">
  <img src="/results/output7_core_8_utilization_1.0.png" width="650">
</p>
<p align="center">
  <img src="/results/output3_with_16_cores.png" width="650">
  <img src="/results/output3_with_32_cores.png" width="650">
</p>

---

📊 **CSV File:**  
- [output_data.csv](./results/output6.csv)

| Task   | Utilization | Period | WCET   | Deadline | Arrival Time | Type  | Frequency | Voltage |
|--------|-------------|--------|--------|----------|--------------|-------|-----------|---------|
| Task 0 | 0.1338      | 4      | 1.2845 | 4        | 0            | hard  | 1.1       | 1.5     |
| Task 1 | 0.0285      | 5      | 3.4144 | 5        | 0            | hard  | 0.75      | 0.9     |
| Task 2 | 0.0057      | 6      | 3.7932 | 6        | 0            | hard  | 0.75      | 0.9     |
| Task 3 | 0.0147      | 1      | 0.6198 | 1        | 0            | hard  | 0.75      | 0.9     |
| Task 4 | 0.0084      | 3      | 1.2931 | 3        | 0            | hard  | 0.75      | 0.9     |
| Task 5 | 0.0011      | 1      | 0.0147 | 1        | 0            | hard  | 0.75      | 0.9     |
| Task 6 | 0.0747      | 5      | 3.0887 | 5        | 0            | hard  | 0.75      | 0.9     |
| Task 7 | 0.0238      | 2      | 0.0702 | 2        | 0            | hard  | 0.75      | 0.9     |
| Task 8 | 0.041       | 3      | 0.2194 | 3        | 0            | hard  | 0.75      | 0.9     |
| Task 9 | 0.0045      | 3      | 0.0239 | 3        | 0            | hard  | 0.75      | 0.9     |
| Task 10| 0.0401      | 3      | 0.7467 | 3        | 0            | hard  | 0.75      | 0.9     |
| Task 11| 0.0154      | 3      | 0.1211 | 3        | 0            | hard  | 0.75      | 0.9     |
| Task 12| 0.0018      | 6      | 0.0812 | 6        | 0            | hard  | 0.75      | 0.9     |
| Task 13| 0.0592      | 3      | 1.7761 | 3        | 0            | hard  | 0.75      | 0.9     |
| Task 14| 0.0083      | 2      | 0.1816 | 2        | 0            | hard  | 0.75      | 0.9     |
| Task 15| 0.0642      | 1      | 0.2705 | 1        | 0            | hard  | 0.75      | 0.9     |
| Task 16| 0.0681      | 5      | 3.4061 | 5        | 0            | hard  | 0.75      | 0.9     |
| Task 17| 0.0186      | 1      | 0.0225 | 1        | 0            | hard  | 0.75      | 0.9     |
| Task 18| 0.0227      | 5      | 1.0325 | 5        | 0            | hard  | 0.75      | 0.9     |
| Task 19| 0.0796      | 1      | 0.0965 | 1        | 0            | hard  | 0.75      | 0.9     |
| Task 20| 0.0045      | 6      | 0.2041 | 6        | 0            | hard  | 0.75      | 0.9     |
| Task 21| 0.0192      | 6      | 1.1507 | 6        | 0            | hard  | 0.75      | 0.9     |
| Task 22| 0.0431      | 4      | 1.8954 | 4        | 0            | hard  | 0.75      | 0.9     |
| Task 23| 0.0042      | 5      | 0.2081 | 5        | 0            | hard  | 0.75      | 0.9     |
| Task 24| 0.0829      | 3      | 0.3019 | 3        | 0            | hard  | 0.75      | 0.9     |
| Task 25| 0.0092      | 2      | 0.0271 | 2        | 0            | hard  | 0.75      | 0.9     |
| Task 26| 0.0219      | 5      | 0.2609 | 5        | 0            | hard  | 0.75      | 0.9     |
| Task 27| 0.0938      | 2      | 0.0876 | 2        | 0            | hard  | 0.75      | 0.9     |
| Task 28| 0.0047      | 3      | 0.0189 | 3        | 0            | hard  | 0.75      | 0.9     |
| Task 29| 0.0023      | 1      | 0.0021 | 1        | 0            | hard  | 0.75      | 0.9     |
| Task 30| 0.2006      |        | 0.5258 | 10       | 9            | soft  | 0.75      | 0.9     |
| Task 31| 0.253       |        | 1.5719 | 9        | 0            | soft  | 0.75      | 0.9     |
| Task 32| 0.2066      |        | 5.4092 | 11       | 10           | soft  | 0.75      | 0.9     |
| Task 33| 0.1854      |        | 3.3038 | 8        | 3            | soft  | 0.75      | 0.9     |
| Task 34| 0.1343      |        | 3.8678 | 11       | 10           | soft  | 0.75      | 0.9     |
| Task 35| 0.2913      |        | 3.2119 | 9        | 6            | soft  | 0.75      | 0.9     |
| Task 36| 0.1569      |        | 3.3772 | 9        | 6            | soft  | 0.75      | 0.9     |
| Task 37| 0.107       |        | 3.7222 | 7        | 5            | soft  | 0.75      | 0.9     |
| Task 38| 0.2978      |        | 6.41   | 11       | 8            | soft  | 0.75      | 0.9     |
| Task 39| 0.2938      |        | 0.1046 | 11       | 10           | soft  | 0.75      | 0.9     |


---

## 📝 TXT Output Samples  

Here are the log files generated for different core configurations and utilization levels:  

### 🔹 **8 Cores**  
- [scheduling_cores_8_utilization_0.25.txt](./results/scheduling_cores_8_utilization_0.25.txt)  
- [scheduling_cores_8_utilization_0.5.txt](./results/scheduling_cores_8_utilization_0.5.txt)  
- [scheduling_cores_8_utilization_0.75.txt](./results/scheduling_cores_8_utilization_0.75.txt)  
- [scheduling_cores_8_utilization_1.0.txt](./results/scheduling_cores_8_utilization_1.0.txt)  

### 🔹 **16 Cores**  
- [scheduling_cores_16_utilization_0.25.txt](./results/scheduling_cores_16_utilization_0.25.txt)  
- [scheduling_cores_16_utilization_0.5.txt](./results/scheduling_cores_16_utilization_0.5.txt)  
- [scheduling_cores_16_utilization_0.75.txt](./results/scheduling_cores_16_utilization_0.75.txt)  
- [scheduling_cores_16_utilization_1.0.txt](./results/scheduling_cores_16_utilization_1.0.txt)  

### 🔹 **32 Cores**  
- [scheduling_cores_32_utilization_0.25.txt](./results/scheduling_cores_32_utilization_0.25.txt)  
- [scheduling_cores_32_utilization_0.5.txt](./results/scheduling_cores_32_utilization_0.5.txt)  
- [scheduling_cores_32_utilization_0.75.txt](./results/scheduling_cores_32_utilization_0.75.txt)  
- [scheduling_cores_32_utilization_1.0.txt](./results/scheduling_cores_32_utilization_1.0.txt)  

---

## 👥 **Members**  
We were a team of two members consisting of the following individuals:  
- [AmirReza Azari](https://github.com/Amirreza81)  
- [Bozorgmehr Zia](https://github.com/BozorgmehrZia)  

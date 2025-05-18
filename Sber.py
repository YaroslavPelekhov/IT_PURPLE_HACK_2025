import json         
import re          
import gymnasium as gym  
import numpy as np  
from gymnasium import spaces  
from stable_baselines3 import PPO  
from datetime import datetime  

# Функция для "выравнивания" списка задач: извлекает все задачи, включая дочерние, в один список.
def flatten_tasks(task_list):
    flat = []
    for task in task_list:
        flat.append(task)
        if "children" in task and isinstance(task["children"], list):
            flat.extend(flatten_tasks(task["children"]))
    return flat

# Функция для парсинга информации о сотруднике из строки. Извлекает роль и зарплату (руб/час).
def parse_employee_info(info_str):
    pattern = r"^(.*?) №\d+ \((\d+)руб/час\)"
    match = re.match(pattern, info_str)
    if match:
        role = match.group(1).strip()
        salary = int(match.group(2))
        return role, salary
    return None, None

# Функция для определения требуемой роли сотрудника по названию задачи.
def get_required_role(task):
    name = task.get("name", "").lower()
    if "аналитика" in name:
        return "Аналитик"
    elif "разработка" in name:
        return "Разработчик"
    elif "тестирование" in name:
        return "Тестировщик"

# Определение среды для планирования проекта, наследуемой от gym.Env.
class ProjectSchedulingEnv(gym.Env):
    def __init__(self, tasks, weights, average_salary):
        super(ProjectSchedulingEnv, self).__init__()
        self.tasks = tasks              
        self.weights = weights
        self.average_salary = average_salary
        self.num_tasks = len(self.tasks) 

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_tasks,), dtype=np.float32)

        self.action_space = spaces.Discrete(self.num_tasks)
        self.reset()

    # Метод сброса среды в начальное состояние
    def reset(self, seed=None, options=None):
        self.scheduled_tasks = []
        self.done = False        
        return self._get_obs(), {}

    # Вспомогательный метод для получения текущего наблюдения
    def _get_obs(self):
        obs = np.zeros(self.num_tasks, dtype=np.float32)
        for idx in self.scheduled_tasks:
            obs[idx] = 1.0
        return obs

    # Метод шага среды: выполняется действие агента, обновляется состояние и вычисляется награда
    def step(self, action):

        if action in self.scheduled_tasks:
            reward = -1.0
        else:
      
            self.scheduled_tasks.append(action)
   
            duration_days = self.tasks[action].get("duration", 1)
            duration_hours = duration_days * 8
        
            req_role = get_required_role(self.tasks[action])

            rate = self.average_salary.get(req_role, 2000)

            cost = duration_hours * rate
   
            reward = - (duration_days * self.weights.get("duration", 7) + cost * self.weights.get("cost", 5))

        if len(self.scheduled_tasks) == self.num_tasks:
            self.done = True
        terminated = self.done
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    # Метод отображения текущего состояния (вывод запланированных задач)
    def render(self, mode="human"):
        print("Назначенные задачи (индексы):", self.scheduled_tasks)

# Функция для оптимизации расписания проекта с использованием алгоритма PPO
def optimize_schedule(flat_tasks, weights, average_salary):

    env = ProjectSchedulingEnv(flat_tasks, weights, average_salary)

    model = PPO("MlpPolicy", env, verbose=0)

    model.learn(total_timesteps=10000)

    obs, _ = env.reset()
    schedule = []
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        schedule.append(action)
        if terminated or truncated:
            break
    return schedule

# Функция для вычисления глобальных метрик проекта: общая продолжительность и общая стоимость
def compute_global_metrics(tasks):
    start_dates = []
    end_dates = []
    total_cost = 0
    for task in tasks:
        start_str = task.get("startDate")
        end_str = task.get("endDate")

        if start_str:
            try:
                start_dates.append(datetime.fromisoformat(start_str))
            except Exception:
                pass
        if end_str:
            try:
                end_dates.append(datetime.fromisoformat(end_str))
            except Exception:
                pass

        total_cost += task.get("salaryExpense", 0)
    if start_dates and end_dates:

        duration_hours = (max(end_dates) - min(start_dates)).total_seconds() / 3600
    else:
        duration_hours = 0
    return duration_hours, total_cost


def main():

    global worker_salary, employee_str

    with open("dataset.json", 'r', encoding='utf-8') as f:
        project_data = json.load(f)

    weights = project_data.get("weights", {"duration": 7, "resources": 3, "cost": 5})

    flat_tasks = flatten_tasks(project_data.get("tasks", {}).get("rows", []))

    # Обработка ресурсов проекта (сотрудников)
    resources = project_data.get("resources", {}).get("rows", [])
    available_workers = {}
    if resources and isinstance(resources, list):
        for res in resources:
            name = res.get("name", "")
            if name:
                role, salary = parse_employee_info(name)
                if role:
                    if role not in available_workers:
                        available_workers[role] = []
                    available_workers[role].append({"id": res.get("id"), "salary": salary, "name": name})

    # Вычисляем среднюю зарплату для каждой роли
    average_salary = {}
    for role, workers in available_workers.items():
        avg = sum([w["salary"] for w in workers]) / len(workers)
        average_salary[role] = avg

    # Оптимизируем расписание задач с помощью PPO
    schedule = optimize_schedule(flat_tasks, weights, average_salary)
    # Добавляем оптимальный порядок задач в каждую задачу
    for order, idx in enumerate(schedule):
        flat_tasks[idx]["optimizedOrder"] = order

    optimized_tasks = []
    worker_hours = {}
    role_counter = {}
    worker_salary = 0
    employee_str = 0
    # Формируем окончательное расписание, сортируя задачи по оптимальному порядку
    for task in sorted(flat_tasks, key=lambda t: t.get("optimizedOrder", 0)):
        days = task.get("duration", 0)
        hours = days * 8
        req_role = get_required_role(task)
        # Назначаем сотрудника на задачу, чередуя их, если имеется несколько кандидатов
        if req_role in available_workers and available_workers[req_role]:
            workers_list = available_workers[req_role]
            workers_count = len(workers_list)
            cnt = role_counter.get(req_role, 0)
            worker_idx = cnt % workers_count
            role_counter[req_role] = cnt + 1
            worker_salary = workers_list[worker_idx]["salary"]
            employee_str = f"{req_role} №{worker_idx + 1} ({worker_salary}руб/час)"
        # Рассчитываем зарплатные расходы для задачи
        salary_expense = hours * worker_salary
        worker_hours[employee_str] = worker_hours.get(employee_str, 0) + hours

        optimized_tasks.append({
            "id": task.get("id"),
            "name": task.get("name", ""),
            "startDate": task.get("startDate", ""),
            "endDate": task.get("endDate", ""),
            "employee": employee_str,
            "salaryExpense": salary_expense,
            "hours": hours
        })
        task["salaryExpense"] = salary_expense

    total_project_duration_hours, total_cost = compute_global_metrics(optimized_tasks)

    # Формируем итоговый JSON-вывод
    output = {
        "optimized_tasks": optimized_tasks,
        "total_project_duration_hours": total_project_duration_hours,
        "total_cost": total_cost,
        "worker_hours": worker_hours
    }
    # Сохраняем оптимизированный план в файл
    with open("optimized_tasks.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print("Оптимизированный план сохранён в файле optimized_tasks.json")

if __name__ == '__main__':
    main()




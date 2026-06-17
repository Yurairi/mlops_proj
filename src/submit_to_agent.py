from clearml import Task
import os


def submit_task_to_agent(script_path, args, queue_name="students"):
    """
    Отправляет задачу в очередь ClearML Agent
    """
    # Создаем Task
    task = Task.init(
        project_name="MLOps_Course",
        task_name=f"Agent_Training_{os.path.basename(script_path).replace('.py', '')}",
        task_type=Task.TaskTypes.training,
    )

    # Добавляем скрипт
    task.set_script(script=script_path)

    # Добавляем аргументы как параметры
    for key, value in args.items():
        task.set_parameter(key, str(value))

    # Отправляем в очередь
    task.execute_remotely(queue_name=queue_name)

    print(f"✅ Task отправлен в очередь '{queue_name}'")
    print(f"🔗 Task ID: {task.id}")
    print(f"📊 UI: http://localhost:9091/projects/MLOps_Course/experiments/{task.id}")

    return task.id


if __name__ == "__main__":
    # Проверяем наличие dataset_id
    if not os.path.exists("dataset_id.txt"):
        print("❌ dataset_id.txt не найден. Сначала создайте Dataset.")
        exit(1)

    with open("dataset_id.txt", "r") as f:
        dataset_id = f.read().strip()

    print(f"📂 Используем Dataset ID: {dataset_id}")

    # Отправляем задачу в Agent
    task_id = submit_task_to_agent(
        script_path="src/train.py",
        args={
            "dataset_id": dataset_id,
            "max_features": 5000,
            "C": 0.8,
            "ngram_max": 2,
            "use_stop_words": True,
            "max_df": 0.8,
            "min_df": 2,
            "max_iter": 1000,
            "task_name": "Agent_Training_Experiment",
            "task_tags": ["agent", "production"],
        },
        queue_name="students",
    )

    print("\n💡 Чтобы посмотреть выполнение в реальном времени:")
    print(f"   http://localhost:9091/projects/MLOps_Course/experiments/{task_id}")

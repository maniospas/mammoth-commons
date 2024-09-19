from flask import render_template, redirect, url_for
from demonstrator.backend.loaders import dataset_loaders, name_to_runnable
import traceback
from datetime import datetime


def handle_new_task_get(database):
    return render_template(
        "new_task.html",
        dataset_loaders=dataset_loaders,
        base_task=dict(),
        default_task_name="Unnamed task"
    )


def handle_new_task_post(request, database):
    dataset_loader_name = request.form["dataset_loader"]
    dataset_parameters = {
        key: request.form[key] for key in request.form if key != "dataset_loader" and key != "task_name"
    }
    call_parameters = {
        key: value
        if not isinstance(value, str) or "," not in value or len(value.strip()) == 1
        else [val.strip() for val in value.split(",")]
        for key, value in dataset_parameters.items()
    }

    task = {
        "dataset_loader": dataset_loader_name,
        "dataset_parameters": dataset_parameters,
        "name": request.form["task_name"],
        "status": "created",
        "modified": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    task_id = database.register(task)  # automatically assigns id

    try:
        task["dataset_loaded"] = name_to_runnable[dataset_loader_name](
            **call_parameters
        )
        task["modified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    except (Exception, RuntimeError) as e:
        task["modified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        task["status"] = "failed"
        traceback.print_exception(e)
        return (
            render_template(
                "500.html",
                title="Error loading dataset",
                message=str(e),
                task_id=task_id,
            ),
            500,
        )

    return redirect(url_for("select_model", task_id=task_id))

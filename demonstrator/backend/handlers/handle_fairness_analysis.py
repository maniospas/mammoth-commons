from flask import render_template, redirect, url_for
from demonstrator.backend.loaders import (
    name_to_runnable,
    analysis_methods,
    parameters_to_class,
)
import traceback
from datetime import datetime


def handle_fairness_analysis_get(database, task_id):
    task = database.get(task_id)
    if not task:
        return redirect(url_for("index"))

    compatible_methods = {
        method: {
            "parameters": entries["parameters"][3:],
            "description": entries["description"],
        }
        for method, entries in analysis_methods.items()
        if issubclass(
            parameters_to_class[task["dataset_loader"]]["return"],
            parameters_to_class[method][entries["parameters"][0][0]],
        )
        and issubclass(
            parameters_to_class[task["model_loader"]]["return"],
            parameters_to_class[method][entries["parameters"][1][0]],
        )
    }

    # Extract existing information from the task (if available)
    preselected_method = task.get("analysis_method", None)
    prefilled_parameters = task.get("analysis_parameters", {})
    prefilled_sensitive_attributes = task.get("sensitive_attributes", [])

    return render_template(
        "fairness_analysis.html",
        analysis_methods=compatible_methods,
        task=task,
        preselected_method=preselected_method,
        prefilled_parameters=prefilled_parameters,
        prefilled_sensitive_attributes=prefilled_sensitive_attributes,
        sensitive_attributes=task["dataset_loaded"].cols,
        default_task_name=task.get("name", "Task "+task["id"])
    )


def handle_fairness_analysis_post(request, database, task_id):
    task = database.get(task_id)
    if not task:
        return redirect(url_for("index"))

    selected_method = request.form["analysis_method"]
    sensitive_attributes = request.form.getlist("sensitive_attributes")
    analysis_parameters = {
        key: request.form[key]
        for key in request.form
        if key != "analysis_method" and key != "sensitive_attributes" and key!="task_name"
    }

    task["analysis_method"] = selected_method
    task["analysis_parameters"] = analysis_parameters
    task["sensitive_attributes"] = sensitive_attributes
    task["status"] = "running"
    task["name"] = request.form["task_name"]
    task["modified"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        task["result"] = name_to_runnable[selected_method](
            task["dataset_loaded"],
            task["model_loaded"],
            sensitive_attributes,
            **analysis_parameters
        ).text()
        task["status"] = "completed"
        task["modified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    except (Exception, RuntimeError) as e:
        task["modified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        task["status"] = "failed"
        traceback.print_exception(e)
        return (
            render_template(
                "500.html",
                title="Error during fairness analysis",
                message=str(e),
                task_id=task_id,
            ),
            500,
        )

    return redirect(url_for("index"))

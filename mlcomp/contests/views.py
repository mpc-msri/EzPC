from os import access
import os
import onnx
from time import timezone
from urllib.error import HTTPError
from django.conf import settings
from django.http import (
    FileResponse,
    HttpResponseForbidden,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseRedirect,
    JsonResponse,
)
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from markdown import markdown
from contests.config import CONTEST_NAME, NUM_TESTCASES
from contests.utils import validate_config, check_model_valid

from mlcomp.settings import BASE_DIR, WEBSITE_URL
from .forms import SubmissionForm, UploadErrorForm, UploadResultForm
from django.utils import timezone
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from .models import Submission
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods


# These variables will be available in all templates
# This is done using `context_processors` in `settings.py`
def default_template_variables(request: HttpRequest):
    return {
        "CONTEST_NAME": CONTEST_NAME,
        "NUM_TESTCASES": NUM_TESTCASES,
        "WEBSITE_URL": WEBSITE_URL,
    }


def home(request: HttpRequest):
    submissions = Submission.objects.all()
    submissionForm = None
    return render(
        request,
        "contests/home.html",
        {
            "submissions": submissions,
            "form": submissionForm or SubmissionForm(),
        },
    )


def participate(request: HttpRequest):
    form = None

    if request.method == "POST":
        form = SubmissionForm(request.POST, request.FILES)
        config_str = request.FILES["model_config"].read()

        # Check if the config.json is valid
        if not validate_config(config_str):
            form.add_error(field="model_config", error="Invalid model config")

        # Check if the model is valid
        model_validity = check_model_valid(request.FILES["stripped_model"])

        if model_validity != True:
            form.add_error(field="stripped_model", error=model_validity)

        if form.is_valid():
            submission: Submission = form.save(commit=False)
            submission.submission_timestamp = timezone.now()
            submission.save()
            return HttpResponseRedirect(
                f"{reverse('contests:post_submission_instructions', args=(submission.id,))}"
            )

    return render(
        request,
        "contests/participate.html",
        {"form": form or SubmissionForm()},
    )


def post_submission_instructions(request: HttpRequest, submission_id: int):
    submission = get_object_or_404(Submission, pk=submission_id)

    port_command_list = ""
    for i in range(submission.num_threads):
        port_command_list += f"-p {submission.server_port+i} "

    port_list = ""
    for i in range(submission.num_threads):
        port_list += f"{submission.server_port+i} "

    return render(
        request,
        "contests/post_submission_instructions.html",
        {
            "submission": submission,
            "port_command_list": port_command_list,
            "port_list": port_list,
        },
    )


# @csrf_exempt
# def start_eval_next_submission(request: HttpRequest, contest_id: int):
#     contest = get_authorized_contest(request, contest_id)
#     next_submission: Submission | None = (
#         contest.submission_set.filter(evaluation_started=False)
#         .order_by("-submission_timestamp")
#         .first()
#     )

#     if next_submission is None:
#         return HttpResponseBadRequest("No submissions in queue")

#     response = {
#         "submission_id": next_submission.id,
#         "model_path": WEBSITE_URL + next_submission.stripped_model.url,
#         "server_ip": next_submission.server_ip,
#         "server_port": next_submission.server_port,
#         "config_path": WEBSITE_URL + next_submission.model_config.url,
#         "pre_process_script_path": WEBSITE_URL + next_submission.pre_process_script.url,
#     }

#     next_submission.evaluation_started = True
#     next_submission.save()
#     return JsonResponse(response)


# @require_http_methods(["POST"])
# @csrf_exempt
# def upload_result(request: HttpRequest, submission_id: int):
#     submission = get_object_or_404(Submission, pk=submission_id)
#     form = UploadResultForm(request.POST)
#     if form.is_valid():
#         data: Submission = form.save(commit=False)
#         submission.score = data.score
#         submission.output = data.output
#         submission.evaluation_finished = True
#         submission.evaluation_finish_timestamp = timezone.now()
#         submission.save()
#         print(submission.score)
#         return HttpResponse("OK")
#     return HttpResponseBadRequest(form.errors.as_json())


# @require_http_methods(["POST"])
# @csrf_exempt
# def upload_error(request: HttpRequest, submission_id: int):
#     submission = get_object_or_404(Submission, pk=submission_id)
#     form = UploadErrorForm(request.POST)
#     if form.is_valid():
#         data: Submission = form.save(commit=False)
#         submission.evaluation_finished = True
#         submission.evaluation_finish_timestamp = timezone.now()
#         submission.error = data.error
#         submission.save()
#         return HttpResponse("OK")

#     return HttpResponseBadRequest(form.errors.as_json())


# def view_submission_error(request: HttpRequest, submission_id: int):
#     submission = get_object_or_404(Submission, pk=submission_id)
#     return HttpResponse(f"<pre>{submission.error}</pre>")

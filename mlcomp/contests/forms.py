from dataclasses import field
from django import forms
from django.contrib.auth.models import User

from contests.models import Submission


class SubmissionForm(forms.ModelForm):
    class Meta:
        model = Submission
        fields = [
            "team_name",
            "stripped_model",
            "model_config",
            "pre_process_script",
            "server_ip",
            "server_port",
            "num_threads",
        ]


class UploadResultForm(forms.ModelForm):
    class Meta:
        model = Submission
        fields = ["score"]


class UploadErrorForm(forms.ModelForm):
    class Meta:
        model = Submission
        fields = ["error"]

from uuid import uuid4
from django.db import models
from contests.formatChecker import MyFileField
from contests.utils import mb_to_bytes


def model_file_name(instance, filename):
    return ".".join(["user_uploads/stripped_models/", str(uuid4()), "onnx"])


def model_config_file_name(instance, filename):
    return ".".join(["user_uploads/model_configs/", str(uuid4()), "json"])


def pre_process_script_file_name(instance, filename):
    return ".".join(["user_uploads/pre_process_scripts/", str(uuid4()), "py"])


class Submission(models.Model):
    team_name = models.CharField(
        max_length=15,
        help_text="Unique name for your submission",
        unique=True,
        blank=False,
    )

    stripped_model = MyFileField(
        upload_to=model_file_name,
        help_text="ONNX model without model weights ( stripped model )",
        max_upload_size=mb_to_bytes(10),
        blank=False,
    )

    model_config = MyFileField(
        upload_to=model_config_file_name,
        content_types=["application/json"],
        max_upload_size=mb_to_bytes(1),
        help_text="JSON file where Target must be SCI and backend must be OT",
    )

    pre_process_script = MyFileField(
        upload_to=pre_process_script_file_name,
        content_types=["text/x-python"],
        max_upload_size=mb_to_bytes(1),
        help_text="Python script that converts input to a numpy array of desired shape",
        blank=False,
    )

    submission_timestamp = models.DateTimeField(auto_now_add=True, blank=False)
    evaluation_finish_timestamp = models.DateTimeField(blank=True, null=True)
    participant_ready = models.BooleanField(default=False)
    evaluation_started = models.BooleanField(default=False)
    evaluation_finished = models.BooleanField(default=False)

    score = models.IntegerField(default=0)
    error = models.TextField(blank=True)

    server_ip = models.GenericIPAddressField(
        blank=False,
        default="127.0.0.1",
        help_text="IP of the server where you will run the participant script",
    )

    server_port = models.IntegerField(
        blank=False,
        default=32000,
        help_text="If you choose port P, N consecutive ports must be open after it, where N is the number of threads in your CPU.",
    )

    num_threads = models.IntegerField(
        blank=False, default=4, help_text="Number of threads in the CPU of your server"
    )

    def start_evaluation(self):
        self.evaluation_started = True
        self.save()

    def make_participant_ready(self):
        self.participant_ready = True
        self.save()

    def __str__(self):
        return self.team_name + " " + str(self.submission_timestamp)

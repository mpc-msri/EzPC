from django.urls import path
from . import views


app_name = "contests"

urlpatterns = [
    path("", views.home, name="home"),
    path("participate/", views.participate, name="participate"),
    path(
        "post_submission_instructions/<int:submission_id>",
        views.post_submission_instructions,
        name="post_submission_instructions",
    ),
    # path('get_next_submission/', views.get_next_submission, name='get_next_submission'),
    # path('make_participant_ready/<int:submission_id>', views.make_participant_ready, name='make_participant_ready'),
    # path('upload_result/<int:submission_id>',
    #      views.upload_result, name='upload_result'),
    # path('upload_error/<int:submission_id>',
    #      views.upload_error, name='upload_error')
]

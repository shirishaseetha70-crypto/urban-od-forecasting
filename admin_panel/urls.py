from django.urls import path
from . import views

urlpatterns = [
    path('data/',         views.data_management,   name='data_management'),
    path('train/',        views.model_training,     name='model_training'),
    path('metrics/',      views.performance_metrics, name='metrics'),
    path('graph/',        views.graph_config,        name='graph_config'),

    # User management
    path('users/',                        views.user_management,    name='user_management'),
    path('users/toggle/<int:user_id>/',   views.toggle_user_status, name='toggle_user'),
    path('users/delete/<int:user_id>/',   views.delete_user,        name='delete_user'),
    path('users/role/<int:user_id>/',     views.change_user_role,   name='change_user_role'),
]

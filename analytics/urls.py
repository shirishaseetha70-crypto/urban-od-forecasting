from django.urls import path
from . import views
urlpatterns = [path('trends/', views.temporal_trends, name='trends')]

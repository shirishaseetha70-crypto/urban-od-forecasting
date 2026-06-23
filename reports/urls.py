from django.urls import path
from . import views
urlpatterns = [
    path('', views.export_view, name='reports'),
    path('csv/', views.download_csv, name='download_csv'),
    path('pdf/', views.download_pdf, name='download_pdf'),
]

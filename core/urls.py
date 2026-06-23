from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect


urlpatterns = [
    path('django-admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('visualization/', include('visualization.urls')),
    path('prediction/', include('prediction.urls')),
    path('analytics/', include('analytics.urls')),
    path('admin-panel/', include('admin_panel.urls')),
    path('reports/', include('reports.urls')),
    path('', lambda request: redirect('login'), name='home'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

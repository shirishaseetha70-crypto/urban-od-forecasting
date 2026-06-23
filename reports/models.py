from django.db import models
from django.contrib.auth.models import User

class ExportedReport(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    report_type = models.CharField(max_length=50)
    file = models.FileField(upload_to='reports/')
    created_at = models.DateTimeField(auto_now_add=True)

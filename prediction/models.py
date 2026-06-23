from django.db import models
from django.contrib.auth.models import User

class ODPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    origin_zone = models.CharField(max_length=50)
    dest_zone = models.CharField(max_length=50)
    predicted_flow = models.FloatField()
    prediction_time = models.DateTimeField(auto_now_add=True)
    target_interval = models.CharField(max_length=50)
    confidence_score = models.FloatField(default=0.0)

class PredictionRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    region = models.CharField(max_length=100)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    status = models.CharField(max_length=20, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    result_file = models.FileField(upload_to='predictions/', blank=True, null=True)

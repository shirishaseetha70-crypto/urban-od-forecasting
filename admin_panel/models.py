from django.db import models

class UploadedDataset(models.Model):
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    cleaned = models.BooleanField(default=False)
    record_count = models.IntegerField(default=0)

class ModelTrainingLog(models.Model):
    epoch = models.IntegerField()
    train_loss = models.FloatField()
    val_loss = models.FloatField()
    mse = models.FloatField(default=0)
    mae = models.FloatField(default=0)
    rmse = models.FloatField(default=0)
    trained_at = models.DateTimeField(auto_now_add=True)

class SystemConfig(models.Model):
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    updated_at = models.DateTimeField(auto_now=True)

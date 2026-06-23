from django.db import models

class MobilityPattern(models.Model):
    zone = models.CharField(max_length=50)
    date = models.DateField()
    hour = models.IntegerField()
    avg_outflow = models.FloatField()
    avg_inflow = models.FloatField()
    day_type = models.CharField(max_length=20)  # weekday/weekend
    season = models.CharField(max_length=20)

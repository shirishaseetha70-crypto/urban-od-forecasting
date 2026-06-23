from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import pandas as pd, json

@login_required
def temporal_trends(request):
    df = pd.read_csv("dataset/od_cleaned.csv")
    
    # Daily pattern
    daily = df.groupby('hour')['od_flow'].mean().reset_index()
    
    # Weekly pattern
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekly = df.groupby('day_of_week')['od_flow'].mean().reindex(day_order).reset_index()
    
    # Seasonal pattern
    seasonal = df.groupby('season')['od_flow'].mean().reset_index()

    chart = {
        'daily': {'labels': daily['hour'].tolist(), 'data': daily['od_flow'].round(2).tolist()},
        'weekly': {'labels': weekly['day_of_week'].tolist(), 'data': weekly['od_flow'].round(2).tolist()},
        'seasonal': {'labels': seasonal['season'].tolist(), 'data': seasonal['od_flow'].round(2).tolist()},
    }
    return render(request, 'analytics/trends.html', {'chart': json.dumps(chart)})

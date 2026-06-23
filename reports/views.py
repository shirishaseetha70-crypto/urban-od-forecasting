from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from .models import ExportedReport
from prediction.models import ODPrediction
import csv, json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

@login_required
def export_view(request):
    predictions = ODPrediction.objects.filter(user=request.user).order_by('-prediction_time')[:50]
    return render(request, 'reports/export.html', {'predictions': predictions})

@login_required
def download_csv(request):
    predictions = ODPrediction.objects.filter(user=request.user)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="od_predictions.csv"'
    writer = csv.writer(response)
    writer.writerow(['Origin Zone','Dest Zone','Predicted Flow','Target Interval','Prediction Time'])
    for p in predictions:
        writer.writerow([p.origin_zone, p.dest_zone, p.predicted_flow, p.target_interval, p.prediction_time])
    return response

@login_required
def download_pdf(request):
    predictions = ODPrediction.objects.filter(user=request.user)[:20]
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="od_report.pdf"'
    p = canvas.Canvas(response, pagesize=A4)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, "Urban OD Prediction Report")
    p.setFont("Helvetica", 11)
    y = 760
    p.drawString(50, y, f"Total Predictions: {predictions.count()}")
    y -= 30
    p.drawString(50, y, "Zone | Predicted Flow | Time")
    y -= 20
    for pred in predictions:
        p.drawString(50, y, f"{pred.origin_zone} → {pred.dest_zone} | {pred.predicted_flow:.2f} | {pred.prediction_time.strftime('%Y-%m-%d %H:%M')}")
        y -= 18
        if y < 50:
            p.showPage()
            y = 800
    p.save()
    return response

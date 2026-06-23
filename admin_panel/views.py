from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import User
from django.contrib import messages
from accounts.models import UserProfile
from .models import UploadedDataset, ModelTrainingLog, SystemConfig
import json


@staff_member_required
def user_management(request):
    users       = User.objects.all().order_by('-date_joined').select_related('userprofile')
    total       = users.count()
    active      = users.filter(is_active=True).count()
    admin_count = users.filter(is_staff=True).count()
    user_count  = users.filter(is_staff=False).count()

    return render(request, 'admin_panel/user_management.html', {
        'users':       users,
        'total':       total,
        'active':      active,
        'admin_count': admin_count,
        'user_count':  user_count,
    })


@staff_member_required
def toggle_user_status(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user == request.user:
        messages.error(request, "You cannot deactivate your own account.")
        return redirect('user_management')
    user.is_active = not user.is_active
    user.save()
    status = "activated" if user.is_active else "deactivated"
    messages.success(request, f"User '{user.username}' has been {status}.")
    return redirect('user_management')


@staff_member_required
def delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user == request.user:
        messages.error(request, "You cannot delete your own account.")
        return redirect('user_management')
    username = user.username
    user.delete()
    messages.success(request, f"User '{username}' has been deleted.")
    return redirect('user_management')


@staff_member_required
def change_user_role(request, user_id):
    if request.method == 'POST':
        user = get_object_or_404(User, id=user_id)
        new_role = request.POST.get('role')
        if user == request.user:
            messages.error(request, "You cannot change your own role.")
            return redirect('user_management')
        profile, created = UserProfile.objects.get_or_create(user=user)
        profile.role = new_role
        profile.save()
        user.is_staff = (new_role == 'admin')
        user.save()
        messages.success(request, f"Role updated to '{new_role}' for '{user.username}'.")
    return redirect('user_management')


# Keep your existing views below (data_management, model_training, etc.)
@staff_member_required
def data_management(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        import pandas as pd
        f  = request.FILES['dataset']
        ds = UploadedDataset.objects.create(name=f.name, file=f)
        df = pd.read_csv(ds.file.path)
        ds.record_count = len(df)
        ds.save()
        messages.success(request, f"Dataset '{f.name}' uploaded successfully.")
    datasets = UploadedDataset.objects.all().order_by('-uploaded_at')
    return render(request, 'admin_panel/data_management.html', {'datasets': datasets})


@staff_member_required
def model_training(request):
    logs = ModelTrainingLog.objects.all().order_by('epoch')
    return render(request, 'admin_panel/model_training.html', {'logs': logs})


@staff_member_required
def performance_metrics(request):
    logs     = ModelTrainingLog.objects.all().order_by('epoch')
    last_log = logs.last()
    loss_data = {
        'epochs': list(logs.values_list('epoch', flat=True)),
        'train':  list(logs.values_list('train_loss', flat=True)),
        'val':    list(logs.values_list('val_loss', flat=True)),
    }
    metrics = {
        'mse':  round(last_log.mse,  6) if last_log else None,
        'mae':  round(last_log.mae,  6) if last_log else None,
        'rmse': round(last_log.rmse, 6) if last_log else None,
    }
    return render(request, 'admin_panel/metrics.html', {
        'loss_data': json.dumps(loss_data),
        'logs':      logs,
        'metrics':   metrics,
        'last_log':  last_log,
    })


@staff_member_required
def graph_config(request):
    import pickle
    import os
    graph_path = os.path.join('dataset', 'road_graph.pkl')
    graph_info = {'nodes': 0, 'edges': 0, 'node_list': []}
    if os.path.exists(graph_path):
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        graph_info = {
            'nodes':     G.number_of_nodes(),
            'edges':     G.number_of_edges(),
            'node_list': list(G.nodes(data=True))[:10]
        }
    return render(request, 'admin_panel/graph_config.html', {'graph_info': graph_info})

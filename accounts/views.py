from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import UserProfile


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'accounts/login.html')

@login_required
def user_management(request):
    from django.contrib.auth.models import User
    users=User.objects.all()
    return render(request,'accounts/user_management.html',{'users':users})

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        username   = request.POST.get('username')
        email      = request.POST.get('email')
        password1  = request.POST.get('password1')
        password2  = request.POST.get('password2')
        role       = request.POST.get('role', 'user')

        if password1 != password2:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'accounts/register.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
            return render(request, 'accounts/register.html')

        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered.')
            return render(request, 'accounts/register.html')

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1
        )
        # If admin role selected, set is_staff = True
        if role == 'admin':
            user.is_staff = True
            user.save()

        UserProfile.objects.create(user=user, role=role)
        messages.success(request, 'Account created! Please log in.')
        return redirect('login')

    return render(request, 'accounts/register.html')


@login_required
def dashboard_view(request):
    from prediction.models import ODPrediction, PredictionRequest
    from admin_panel.models import ModelTrainingLog, UploadedDataset

    stats = {
        'total_predictions':  ODPrediction.objects.count(),
        'pending_requests':   PredictionRequest.objects.filter(status='pending').count(),
        'datasets_uploaded':  UploadedDataset.objects.count(),
        'last_training':      ModelTrainingLog.objects.last(),
    }
    return render(request, 'accounts/dashboard.html', {'stats': stats})


def logout_view(request):
    logout(request)
    return redirect('login')

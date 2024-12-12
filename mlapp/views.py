from django.shortcuts import render, redirect
from .ml_model import train_model  # Correctly importing the train_model function
import pickle
import os
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()

    return render(request, 'mlapp/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

def home(request):
    return render(request, 'mlapp/home.html')

def train_model_view(request):
    if request.method == 'POST':
        try:
            train_model()
            return render(request, 'mlapp/train_model.html', {'message': 'Model trained and saved successfully!'})
        except Exception as e:
            return render(request, 'mlapp/train_model.html', {'message': f"Error during model training: {e}"})
    return render(request, 'mlapp/train_model.html')

def predict_view(request):
    prediction = None
    if request.method == 'POST':
        try:
            model_path = r'D:\EggProduction_PredictApp\eggprediction\mlapp\models\egg_model.sav'
            scaler_path = r'D:\EggProduction_PredictApp\eggprediction\mlapp\models\scaler.sav'

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                prediction = "Model or scaler file not found. Please train the model first."
            else:
                model = pickle.load(open(model_path, 'rb'))
                scaler = pickle.load(open(scaler_path, 'rb'))

                try:
                    inputs = [
                        float(request.POST['age_week']),
                        float(request.POST['feed']),
                        float(request.POST['water']),
                        float(request.POST['bodyweight']),
                        float(request.POST['lighting']),
                    ]
                except ValueError:
                    return render(request, 'mlapp/predict.html', {'prediction': "Please enter valid numeric values for all inputs."})

                scaled_inputs = scaler.transform([inputs])
                prediction = model.predict(scaled_inputs)[0]
        except Exception as e:
            prediction = "An error occurred during prediction."

    return render(request, 'mlapp/predict.html', {'prediction': prediction})

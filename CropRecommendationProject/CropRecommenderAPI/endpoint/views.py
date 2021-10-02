from django.shortcuts import render
from rest_framework import views, status
from rest_framework.response import Response
import json

from CropRecommenderAPI.wsgi import knn_crop_classifier

# Create your views here.

class PredictCrop(views.APIView):
    def post(self, request,  format=None):
        print(request.data)
        predicted_crops = knn_crop_classifier.Predict_Class(request.data)
        return Response(predicted_crops)
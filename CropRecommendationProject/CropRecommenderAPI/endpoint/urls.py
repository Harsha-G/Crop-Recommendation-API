from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from django.urls.resolvers import URLPattern
from rest_framework.routers import DefaultRouter

from endpoint.views import PredictCrop

router = DefaultRouter(trailing_slash=False)

urlpatterns = [
    url(
        r"predictcrop", PredictCrop.as_view()
    )
]
"""
WSGI config for CropRecommenderAPI project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

import pandas as pd
import numpy as np
import os

from ml.knnclassifier import KnnClassifier


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CropRecommenderAPI.settings')

application = get_wsgi_application()

knn_crop_classifier = KnnClassifier()

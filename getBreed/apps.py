from django.apps import AppConfig

from .predictor import FinalModel




class GetbreedConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'getBreed'

    predictor = FinalModel()




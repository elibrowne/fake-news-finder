from django.urls import path

from . import views

urlpatterns = [
    path('', views.index.as_view(), name='index'),
    path('info', views.info.as_view(), name='info'),
    path('tips', views.tips.as_view(), name='tips')
]
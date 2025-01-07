from django.urls import path
from . import views

urlpatterns = [
    # Route for the form page
    path('', views.select_month_time, name='select_month_time'),
    # Route for the map generation with the selected month and time
    path('generate-map/', views.generate_bus_stop_map, name='generate_bus_stop_map'),
]

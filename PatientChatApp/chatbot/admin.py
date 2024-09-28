from django.contrib import admin
from .models import Chat, Patient, Doctor, Reservation, ReservationRequest

# Register your models here.
admin.site.register(Chat)
admin.site.register(Patient)
admin.site.register(Doctor)
admin.site.register(Reservation)
admin.site.register(ReservationRequest)

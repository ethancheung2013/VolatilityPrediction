from django.db import models

class NewsContent(models.Model):

    url                = models.CharField(max_length=1000)
    title              = models.CharField(max_length=4000)
    content            = models.CharField(max_length=50000)
    date               = models.DateField()

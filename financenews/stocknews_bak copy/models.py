from django.db import models

class NewsContent(models.Model):

    url                = models.CharField(max_length=1000)
    title              = models.CharField(max_length=4000)
    content            = models.CharField(max_length=50000)
    date               = models.DateField()

    def __str__(self):
        return self.title

class yahooCalendar(models.Model):
    datetime            = models.DateField()
    statistic           = models.CharField(max_length=100)
    for_period          = models.CharField(max_length=50)
    actual              = models.FloatField()
    briefing_forecast   = models.FloatField()
    market_expects      = models.FloatField()
    prior               = models.FloatField()
    revised_from        = models.FloatField()

from django.conf.urls import patterns, include, url
from django.conf import settings
from django.contrib import admin
# from dh5bp.urls import urlpatterns as dh5bp_urls

admin.autodiscover()

# handler404 = 'dh5bp.views.page_not_found'
# handler500 = 'dh5bp.views.server_error'

urlpatterns = patterns('',
    # Examples:
    # maps code to the path after the domain
    # django does this by making you write regex to match a path coming in to the view the will be run
    # i didn't get anything ^$
    #          this is the view to run (that takes a request obj and returns a response obj)
    # https://docs.djangoproject.com/en/1.5/topics/http/urls/
    # url(r'^$', 'stocknews.views.index', name='home'),
    url(r'^home/$', 'stocknews.views.index', name='home'),
    # url(r'^news/', 'stocknews.views.index', name='news'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
)

if settings.DEBUG:
    import debug_toolbar
    urlpatterns += patterns('',
        url(r'^__debug__/', include(debug_toolbar.urls)),
    )

# urlpatterns += dh5bp_urls
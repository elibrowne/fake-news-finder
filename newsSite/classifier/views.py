from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views import View

# Create your views here.
class index(View):
	# Only using GET data, no POST data
	def get(self, request):
		# Load the template
		template = loader.get_template('classifier/homepage.html')
		articleContent = "" # placeholder for future data 
		realNews = False

		# See if the user had submitted an article to be judged or not.
		# Try/catch is used to ignore null errors
		try:
			if request.GET['article']:
				articleContent = request.GET['article']
		except: 
			# Adding code to fill the except block
			print("No article, just loading the webpage.")

		# Create context: article content to determine if something was entered
		# and boolean to output real or fake
		context = {
			'article': articleContent,
			'real': realNews
		}
		
		# Return HTTP response with context
		return HttpResponse(template.render(context, request))
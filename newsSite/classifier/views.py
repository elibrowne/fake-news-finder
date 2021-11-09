from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views import View
import pickle

# Create your views here.
class index(View):
	# Only using GET data, no POST data
	def get(self, request):
		# Load the template
		template = loader.get_template("classifier/homepage.html")
		articleContent = "" # placeholder for future data 
		realNews = False

		# Use a try/catch block to see if there's an article
		try: 
			articleContent = request.GET['article']
			# Use pickle to load saved vectorizer and classifier (save time!)
			vectorizer = pickle.load(open("classifier/vectorizer.pkl", 'rb'))
			loaded_model = pickle.load(open("classifier/newsClassifier.pkl", 'rb'))
			# The first element on the prediction is 0 for fake and 1 for real
			result = loaded_model.predict(vectorizer.transform([articleContent]))[0]
			if result == 1:
				realNews = True # signal to page that it's real
		except: # no article sent, just load the page with a blank one
			articleContent = "" # empty article = no result

		# Create context: article content to determine if something was entered
		# and boolean to output real or fake
		context = {
			'article': articleContent,
			'real': realNews
		}
		
		# Return HTTP response with context
		return HttpResponse(template.render(context, request))

class info(View):
	def get(self, request):
		# Load template
		template = loader.get_template("classifier/info.html")
		context = {}
		
		# Return HTTP response with context
		return HttpResponse(template.render(context, request))

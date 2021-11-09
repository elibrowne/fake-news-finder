from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views import View
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Create your views here.
class index(View):
	# Only using GET data, no POST data
	def get(self, request):
		# Load the template
		template = loader.get_template("classifier/homepage.html")
		articleContent = "" # placeholder for future data 
		realNews = False

		vectorizer = pickle.load(open("classifier/vectorizer.pkl", 'rb'))
		articleContent = request.GET['article']
		print(articleContent)
		loaded_model = pickle.load(open("classifier/newsClassifier.pkl", 'rb'))
		print("TRYING")
		result = loaded_model.predict(vectorizer.transform([articleContent]))[0]
		print(result)
		if result == 1:
			realNews = True

		# Create context: article content to determine if something was entered
		# and boolean to output real or fake
		print(realNews)
		context = {
			'article': articleContent,
			'real': realNews
		}
		
		# Return HTTP response with context
		return HttpResponse(template.render(context, request))
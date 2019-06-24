from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 

# NLP Packages
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag #tokenization # Pos Tag
from nltk.stem import WordNetLemmatizer #Lemmatization
from nltk.corpus import wordnet,stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
tfidf = TfidfVectorizer()
stop = stopwords.words('english')
stop.remove('what')
stop.remove('which')
#print(stop)
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')

chatbot = pd.read_excel('./data.xlsx')
#chatbot.head()

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('bot.html')

def postag(pos):
    if pos.startswith('N'):
        wp = wordnet.NOUN
    elif pos.startswith('V'):
        wp = wordnet.VERB
    elif pos.startswith('R'):
        wp = wordnet.ADV
    elif pos.startswith('J'):
        wp = wordnet.ADJ
    else:
        wp = wordnet.NOUN
    
    return wp

wnl = WordNetLemmatizer()

def textprocess(doc):
    
    # Step1 : Converting into lower case
    doc = doc.lower()
    # Step2 : Remove special characters
    doc = re.sub('[^a-z]',' ',doc)
    #Step3 : pos tagging (parts of speech)
    token = word_tokenize(doc) # tokenization -get the words
    token_pos = pos_tag(token)
    # step4 : Lemma and remove stopwords
    lemma = [wnl.lemmatize(word,pos=postag(pos)) for word,pos in token_pos]
    #lemma = [wnl.lemmatize(word,pos=postag(pos)) for word,pos in token_pos if word not in stop]
    clean = ' '.join(lemma)
    return clean

def cosine(a,b):
    moda = np.linalg.norm(a) # magnitude of a
    modb = np.linalg.norm(b) # magnitude of b
    dotprod = np.dot(a,b) # dot product of vector a and vector b
    # a[0], b[0] -> remove shape in it, we don't want vector to have some shape
    # i.e., neither column matrix nor row matrix
    cos = dotprod/(moda*modb)
    return cos

Total_Questions1 = {}
for i,val in enumerate(list(chatbot['Questions'])):
  Total_Questions1.update({i:textprocess(val)})

def tfidf_vec(text): 
  X = tfidf.fit_transform(text).toarray()
  return X

def Question_mapping(query):   
 
  query = textprocess(query)
  subject = query.split()
  documents = []
  for i in range(len(subject)):
    if i == len(subject)-1:
      pass
    else:
      documents.append(chatbot['Questions'][chatbot['Questions'].str.contains((subject[i]),case=False)])
  corpus1 = []
  for docs in documents:
    doc = list(docs)
    for dc in doc:
      corpus1.append(textprocess(dc))
      #print(corpus1)
  return corpus1

def chatanswers(query):
    #step-1: text processing
    clean = textprocess(query)
    #step-2: Chunking
    corpus = Question_mapping(query) 
    Y = tfidf_vec(corpus)
    b = tfidf.transform([query]).toarray() # query in list
    cosvalue = {}
    for i,vector in enumerate(Y):
      cos = cosine(vector,b[0]) # b[0] -> remove shape in it
      if cos>0.5:
        cosvalue.update({i:cos}) # append values in dictionary
    sort = sorted(cosvalue.items(),key=operator.itemgetter(1),reverse=True)
    #print(sort)
    ind = [index for index,cosv in sort[:1]]
    #print(ind)
    var = 0
    for i in ind:
      var = i
    question_number = 0
    for key,value in Total_Questions1.items():
      if corpus[var] in value:
        question_number = key
    return ind,list(chatbot.loc[[question_number]]['Answers'])

@app.route('/bot',methods=['POST'])
def bot():
	greetings = ['hi','hii','hello','helo','hey','hey!','wassup','what is up',"what's up",'whatsup', 'whatszap','greetings','dude','what is your name?',"what is your name","what's your name?",
               "what's your name",'your name',]
	sendoff = ['bye','see you','see u','good bye','byee','thank you','tc','bye!','quit','exit','ok thanks','thanku','thanks']
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		while True:
			
			rawtext = textprocess(rawtext)
			if  len(rawtext.split()) <= 1 and (rawtext not in greetings and rawtext not in sendoff):
				ans = "I’m sorry, I don’t understand."
				print(ans)
			elif rawtext in greetings:
				ans = "Hello! This is KASPER how may I help you"
				print(ans)
			elif rawtext in sendoff:
				ans = "KASPER : ""Have a nice day"
				print(ans)
				break
			else:
				index,ans=chatanswers(rawtext)
				print(ans)
			return render_template('bot.html',received_text=rawtext,response_text=ans)

	#return render_template('index.html',received_text = rawtext,number_of_tokens=number_of_tokens,response_text=ans,len_of_words=number_of_tokens)

if __name__ == '__main__':
	app.run(debug=True)
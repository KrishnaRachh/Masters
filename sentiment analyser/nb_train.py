
from sklearn.feature_extraction.text import CountVectorizer #does tokenizatoiin,stop words-all NLP stuff
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB 
import pandas as pd
import pickle


data = pd.read_csv(r".\CrudeOil_News_Articles.csv", encoding = "ISO-8859-1") 

X = data.iloc[:,1] 

# tokenize the news text and convert data in matrix format
vectorizer = CountVectorizer(stop_words='english')

X_vec = vectorizer.fit_transform(X) # fit_transform func creates feature matrix 

print(X_vec) # Scipy sparse matrix # token assigned to words by sklearn
print(vectorizer.vocabulary_)
pickle.dump(vectorizer, open("vectorizer_crude_oil", 'wb')) 

X_vec = X_vec.todense() 

# Transform data by applying term frequency inverse document frequency (TF-IDF) 
tfidf = TfidfTransformer() #by default applies "l2" normalization
X_tfidf = tfidf.fit_transform(X_vec)
X_tfidf = X_tfidf.todense()


#Apply Naive Bayes algorithm to train data

# Extract the news body and labels for training the classifier
X_train = X_tfidf[:40,:] 
Y_train = data.iloc[:40,2]  

# Train the NB classifier
clf = GaussianNB().fit(X_train, Y_train)
pickle.dump(clf, open("nb_clf_crude_oil", 'wb')) 

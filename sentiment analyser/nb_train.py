
# import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer #does tokenizatoiin,stop words-all NLP stuff
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB #Gaussian is used since we have 3 variables -pos,neg,neutral in our sentiment dataset. If you have 2 variables can use Multinomial
import pandas as pd
import pickle


data = pd.read_csv(r"C:\Users\krkoo\OneDrive\Desktop\Udemy\Mayank Rasu lects\Section 10-Sentiment Analysis\CrudeOil_News_Articles.csv", encoding = "ISO-8859-1") #encoding is required by sklearn library

X = data.iloc[:,1] # extract column with news article body

# tokenize the news text and convert data in matrix format
vectorizer = CountVectorizer(stop_words='english')

X_vec = vectorizer.fit_transform(X) # fit_transform func creates feature matrix #X_vec required for tf-idf

print(X_vec) # Scipy sparse matrix # X_vec displays all the words(token assigned to words by sklearn) which are repeatative
print(vectorizer.vocabulary_)
pickle.dump(vectorizer, open("vectorizer_crude_oil", 'wb'))  # Save vectorizer for reuse #pickle lets you save any python object on your local drive for later use

X_vec = X_vec.todense() # convert sparse matrix into dense matrix

# Transform data by applying term frequency inverse document frequency (TF-IDF) 
tfidf = TfidfTransformer() #by default applies "l2" normalization
X_tfidf = tfidf.fit_transform(X_vec)
X_tfidf = X_tfidf.todense()


##################Apply Naive Bayes algorithm to train data####################

# Extract the news body and labels for training the classifier
X_train = X_tfidf[:40,:] #40 is the number of articles in our csv file
Y_train = data.iloc[:40,2]  # 2 is the column of excel where we have to manually write Neutral, Pos or Neg and then run GaussianNB -(as coded below)

# Train the NB classifier
clf = GaussianNB().fit(X_train, Y_train)
pickle.dump(clf, open("nb_clf_crude_oil", 'wb')) # Save classifier for reuse


#after trainging data using above codes- run nb_predict to predict any future news aticles to be pos/neg/neutral
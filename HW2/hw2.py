#
# to run the programm 
# python hw2.py --lr --MinF 10 --MaxF 15 --paramC 1.0
#
#

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import argparse


def load_iris_data() :

    # load the iris dataset from the sklearn module
    iris = datasets.load_iris()

    # extract the elements of the data that are used in this exercise
    return (iris.data, iris.target, iris.target_names)



def knn(X_train, y_train, k_neighbors = 3 ) :
    # function returns a kNN object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --> fit the model using a training set
    #   predict(X_classify) --> to predict a result using the trained model
    #   score(X_test, y_test) --> to score the model using a test set

    clf = KNeighborsClassifier(k_neighbors)
    clf.fit(X_train, y_train)

    return clf


def nb(X_train, y_train) :
    # this function returns a Naive Bayes object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --> fit the model using a training set
    #   predict(X_classify) --> to predict a result using the trained model
    #   score(X_test, y_test) --> to score the model using a test set

    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)

    return clf

# generic cross validation function
def cross_validate(XX, yy, classifier, k_fold) :

    # derive a set of (random) training and testing indices
    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0)

    k_score_total = 0
    # for each training and testing slices run the classifier, and score the results
    for train_slice, test_slice in k_fold_indices :

        model=classifier(XX[[ train_slice  ]],
                         yy[[ train_slice  ]])

        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        k_score_total += k_score

    # return the average accuracy
    return k_score_total/k_fold

c=2.0
#def lr(X_train, y_train, c=0.1):
def lr(X_train, y_train):
   logreg = LogisticRegression(C=c)
   clf = logreg.fit(X_train, y_train)
   return clf    


(XX,yy,y)=load_iris_data()    

parser = argparse.ArgumentParser(description='Process the input')
parser.add_argument('--knn', action='store_true', help='KNN output, input knn')
parser.add_argument('--nb', action='store_true', help='Naive Bayes output, input nb')
parser.add_argument('--lr', action='store_true', help='Logistic Regression output, input lr')
parser.add_argument('--MinF', help='Min Fold, input -- number ', type=int)
parser.add_argument('--MaxF', help='Max Fold, input -- number ', type=int)
parser.add_argument('--paramC', help='paramC, input -- number ', type=float)
args = parser.parse_args()


if args.knn==True:
    classfiers_to_cv=[("kNN",knn)]
    print
    print ' ---> KNN classifier <---'

    for (c_label, classifer) in classfiers_to_cv :

        print
        print "---> %s <---" % c_label

        best_k=0
        best_cv_a=0
    for k_f in [2,3,5,10,15,30,50,75,150] :
       cv_a = cross_validate(XX, yy, classifer, k_fold=k_f)
       if cv_a >  best_cv_a :
            best_cv_a=cv_a
            best_k=k_f

       print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)


    print "Highest Accuracy: fold <<%s>> :: <<%s>>" % (best_k, best_cv_a)

elif args.nb==True:
    classfiers_to_cv=[("Naive Bayes",nb)]
    print
    print ' ---> Naive Bayes classifier <---'

    for (c_label, classifer) in classfiers_to_cv :

        print
        print "---> %s <---" % c_label

    best_k=0
    best_cv_a=0
    for k_f in [2,3,5,10,15,30,50,75,150] :

       cv_a = cross_validate(XX, yy, classifer, k_fold=k_f)
       if cv_a >  best_cv_a :
            best_cv_a=cv_a
            best_k=k_f

       print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)


    print "Highest Accuracy: fold <<%s>> :: <<%s>>" % (best_k, best_cv_a)

elif args.lr==True:
    
        classfiers_to_cv=[("Logistic Regression",lr)]
        

        for (c_label, classifer) in classfiers_to_cv :

            print
            print "---> %s <---" % c_label

            best_k=0
            best_cv_a=0
            c=args.paramC
            for k_f in range(args.MinF,args.MaxF):
        

                cv_a = cross_validate(XX, yy, classifer, k_fold=k_f)
                if cv_a >  best_cv_a :
                    best_cv_a=cv_a
                    best_k=k_f

                print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)

        

            print "Highest Accuracy: fold <<%s>> :: <<%s>>" % (best_k, best_cv_a)   

else: 
    print 
    print '---> kNN <--- and ---> Naive Bayes <--- ---> LogisticRegression <---'
    classfiers_to_cv=[("kNN",knn),("Naive Bayes",nb),("Logistic Regression",lr)]

    for (c_label, classifer) in classfiers_to_cv :

        print
        print "---> %s <---" % c_label

        best_k=0
        best_cv_a=0
        for k_f in [2,3,5,10,15,30,50,75,150] :
            cv_a = cross_validate(XX, yy, classifer, k_fold=k_f)
            if cv_a >  best_cv_a :
                best_cv_a=cv_a
                best_k=k_f

            print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)


        print "Highest Accuracy: fold <<%s>> :: <<%s>>" % (best_k, best_cv_a)

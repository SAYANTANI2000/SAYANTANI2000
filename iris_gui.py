#################################dataset#######################################
#load iris dataset
from sklearn.datasets import load_iris
iris=load_iris()

##############################
X=iris.data   ##input
Y=iris.target ##output

###########
##split the dataset for training and testing
##for training 80% and for testing 20%

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=11)
###############################################################################

#################################FUNCTIONS#####################################

def KNN():
    ##here we use a model KNN##
    #K Nearest Neighbors algorithm
    global acc_knn
    from sklearn.neighbors import KNeighborsClassifier
    K=KNeighborsClassifier(n_neighbors=5)
    
    ##train the model by using training data set
    K.fit(X_train,Y_train)
    
    ##test the model by testing dataset
    Y_pred_knn=K.predict(X_test)
    
    ##find accuracy
    from sklearn.metrics import accuracy_score
    acc_knn=accuracy_score(Y_test,Y_pred_knn)
    acc_knn=round(acc_knn*100,2)
    m.showinfo(title='KNN',message="Accuracy is "+str(acc_knn)+" %")

def LG():
    ##here we use a model LinearRegression##
    global acc_lg
    from sklearn.linear_model import LogisticRegression
    L=LogisticRegression(solver='liblinear',multi_class='auto')
    
    ##train the model
    L.fit(X_train,Y_train)
    
    ##test the model
    Y_pred_lg=L.predict(X_test)
    
    ##find accuracy of logisitic regression
    from sklearn.metrics import accuracy_score
    acc_lg=accuracy_score(Y_test,Y_pred_lg)
    acc_lg=round(acc_lg*100,2)
    m.showinfo(title='LG',message="Accuracy is "+str(acc_lg)+" %")


def NB():
    #2.Implement Naive bayes
    global acc_nb
    from sklearn.naive_bayes import GaussianNB
    N=GaussianNB()
    ##train the model
    N.fit(X_train,Y_train)
    
    ##test the model
    Y_pred_nb=N.predict(X_test)
    
    ##find accuracy of Naive Bayes
    from sklearn.metrics import accuracy_score
    acc_nb=accuracy_score(Y_test,Y_pred_nb)
    acc_nb=round(acc_nb*100,2)
    m.showinfo(title='NB',message="Accuracy is "+str(acc_nb)+" %")

def DT():
    #3. Decision Tree
    global acc_dt
    from sklearn.tree import DecisionTreeClassifier
    D=DecisionTreeClassifier()
    ##train the model
    D.fit(X_train,Y_train)
    
    ##test the model
    Y_pred_dt=D.predict(X_test)
    
    ##find accuracy of Decisison Tree
    from sklearn.metrics import accuracy_score
    acc_dt=accuracy_score(Y_test,Y_pred_dt)
    acc_dt=round(acc_dt*100,2)
    m.showinfo(title='DT',message="Accuracy is "+str(acc_dt)+" %")

def COMPARE():
    import matplotlib.pyplot as plt
    model=['knn','lg','dt','nb']
    accuracy=[acc_knn,acc_lg,acc_dt,acc_nb]
    plt.bar(model,accuracy,color=['green','yellow','red','blue'])
    plt.show()
    
###############################################################################
    
######################GUI######################################################
from tkinter import*
import tkinter.messagebox as m
w=Tk()
L=Label(w,text="IRIS Flower Prediction",fg="green")
Bknn=Button(w,text="KNN",command=KNN)
Blg=Button(w,text="LG",command=LG)
Bdt=Button(w,text="DT",command=DT)
Bnb=Button(w,text="NB",command=NB)
Bcmp=Button(w,text="Compare",command=COMPARE)

L.grid(row=1,column=1,columnspan=4)
Bknn.grid(row=2,column=1)
Blg.grid(row=2,column=2)
Bdt.grid(row=2,column=3)
Bnb.grid(row=2,column=4)
Bcmp.grid(row=3,column=2,columnspan=2)

##Bknn=Button(w,text"KNN")
##Bknn=Button(w,text"KNN")




w.mainloop()
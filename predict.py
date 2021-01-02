import pandas as pd # data processing
import numpy as np # working with arrays
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.linear_model import LogisticRegression # model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import jaccard_score as jss # evaluation metric
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import classification_report # evaluation metric
from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import log_loss # evaluation metric
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import cgi, cgitb # Import modules for CGI handling 

def predict(req_data):
    print(req_data)
    result = runJob(req_data)
    return result


def runJob(req_data):
    df= pd.read_csv('two.csv')
    # Importing and cleaning the data
     # Create instance of FieldStorage 
    form = cgi.FieldStorage() 

    # Get data from fields
    q1 = form.getvalue('q1')
    q2 = form.getvalue('q2')
    q3 = form.getvalue('q3')
    q4 = form.getvalue('q4')
    q5 = form.getvalue('q5')
    q6 = form.getvalue('q6')
    q7 = form.getvalue('q7')
    q8 = form.getvalue('q8')
    q9 = form.getvalue('q9')
    q10 = form.getvalue('q10')
    q11 = form.getvalue('q11')
    q12 = form.getvalue('q12')
    q13 = form.getvalue('q13')
    q14 = form.getvalue('q14')
    q15 = form.getvalue('q15')
    q16 = form.getvalue('q16')
    q17 = form.getvalue('q17')
    q18 = form.getvalue('q18')
    q19 = form.getvalue('q19')
    q20 = form.getvalue('q20')
    q21 = form.getvalue('q21')
    q22 = form.getvalue('q22')
    q23 = form.getvalue('q23')
    q24 = form.getvalue('q24')
    q25 = form.getvalue('q25')
    q26 = form.getvalue('q26')
    q27 = form.getvalue('q27')
    q28 = form.getvalue('q28')
    q29 = form.getvalue('q29')
    q30 = form.getvalue('q30')
    q31 = form.getvalue('q31')
    q32 = form.getvalue('q32')
    q33 = form.getvalue('q33')
    q34 = form.getvalue('q34')
    q35 = form.getvalue('q35')
    q36 = form.getvalue('q36')
    q37 = form.getvalue('q37')
    q38 = form.getvalue('q38')

    inputs = [1.,2.,3.,4.,5.,6]
    outputs = [['exposed'], ['diseased'],['susceptible'], ['recovred possible injury again'] ,['recovred'], ['infectious']]
    X = np.asarray(inputs)
    X = X.reshape(-1, 1)
    y = np.asarray(outputs)

    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size = 0.2)
    ##print('X_train samples : ', X_train[:5])
    ##print('X_test samples : ', X_test[:5])
    ##print('y_train samples : ', y_train[:10])
    ##print('y_test samples : ', y_test[:10])
    # Modelling
    
    X_scaled = preprocessing.scale(X_train)
    ##print("X_scaled=", X_scaled [:3])
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    ##print(lr)
    # Predictions
    yhat = lr.predict(X_test)
    yhat_prob = lr.predict_proba(X_test)
    #print('yhat samples : ', yhat[:10])
    #print('yhat_prob samples : ',yhat_prob[:10])
    
    # 1. Jaccard Index
    #print('similerity index = ',jss(y_test, yhat, average='macro'))
    # 2. Precision Score
    #print('precision_score = ',precision_score(y_test, yhat, average='micro'))
    # 3. Log loss
    #print('log_loss = ', log_loss(["spam", "ham", "ham", "spam"],[[.1, .9], [.9, .1], [.8, .2], [.35, .65]]))
    # 4. Classificaton report
    #===target_names = ['exposed', 'diseased', 'susceptible' ,'recovred possible injury again','recovred', 'infectious' ]
    #print(classification_report(y_test, yhat, labels=[1, 2, 3,4,5,6], target_names=target_names))
    # 5. Confusion matrix
    predictions = lr.predict(X_test)
    cm = metrics.confusion_matrix(y_test, predictions)
    #print(cm)
    
    Classification1 = (classification_report(y_test, yhat, labels=[1], target_names = ['exposed'], zero_division=0))
    #print(Classification1)
    
    Classification2 = (classification_report(y_test, yhat, labels=[2], target_names = ['diseased'], zero_division=0))
    #print(Classification2)
    
    Classification3 = (classification_report(y_test, yhat, labels=[3], target_names = ['susceptible'], zero_division=0))
    #print(Classification3)
    
    Classification4 = (classification_report(y_test, yhat, labels=[4], target_names = ['recovred possible injury again'], zero_division=0))
    #print(Classification4)
    
    Classification5 = (classification_report(y_test, yhat, labels=[5], target_names = ['recovred'], zero_division=0))
    #print(Classification5)
    
    Classification6 = (classification_report(y_test, yhat, labels=[6], target_names = ['infectious'], zero_division=0))
    #print(Classification6)
    
    #print('lable [1] metric :')
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0)
    #print('precion exposed',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[1], zero_division=0)
    #print('recall exposed', recall_metric)
    
    f1 = f1_score (y_test, yhat,average='weighted', labels=[1], zero_division=0)
    #print ('f1 score =', f1)
    
    #print('---------------------------------------------------------------')
    #print('lable [2] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0)
    #print('precion diseased',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[2], zero_division=0)
    #print('recall diseased', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[2], zero_division=0)
    #print ('f2 score =', f2)
    
    #print('---------------------------------------------------------------')
    #print('lable [3] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[3], zero_division=0)
    #print('precion susceptible',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[3], zero_division=0)
    #print('recall susceptible', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[3], zero_division=0)
    #print ('f2 score =', f2)
    
    #print('---------------------------------------------------------------')
    #print('lable [4] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0)
    #print('precion recovred possible injury again',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[4], zero_division=0)
    #print('recall recovred possible injury again', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[4], zero_division=0)
    #print ('f2 score =', f2)
    
    #print('---------------------------------------------------------------')
    #print('lable [5] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0)
    #print('precion recovred',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[5], zero_division=0)
    #print('recall recovred', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[5], zero_division=0)
    #print ('f2 score =', f2)
    
    #print('---------------------------------------------------------------')
    #print('lable [6] metric :')
    
    precision_metric = precision_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0)
    ##print('precion infectious',precision_metric)
    
    recall_metric = recall_score(y_test, yhat, average = "weighted", labels=[6], zero_division=0)
    ##print('recall infectious', recall_metric)
    
    f2 = f1_score (y_test, yhat,average='weighted', labels=[6], zero_division=0)
    ##print ('f2 score =', f2)
    
    
    #print('---------------------------------------------------------------')

    result = ''
    form.getvalue('q37')== q37_value
    if q37_value  = form.getvalue('5'):
        result = 'نتيجة الفحص: مصاب                             ويترتب على ذلك :  1. إجراء الفحص المخبري للتأكد من الإصابة بنسبة أكبر                                2. القيام بالعزل المنزلي، وفي حال شعرت بأي تعب يحتاج لتدخل طبي اتصل بالجهات المعنية كالطب الوقائي                                3. المحافظة على تناول المكملات الغذائية والفيتاميان إما على شكل أدوية بوصفة طبية، وإما عن طريق الخضار كالفلفل الحلو الغني بفيتامين د، والفواكه الحمضية'
        print (result)



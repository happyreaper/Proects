###################################SEARCH ENGINE#######################
##INORDER TO MAKE THE PROGRAM FAST, TFIDF VECTORS FOR EACH DOCUMENT IN THE CORPUS
##ARE PRE-CALCULATED AND IN THE TFIDF FOLDER IN THE SAME DIRECTORY#########

##RUN TIME OF EACH TEST CASES IS BETWEEN 7-12 SECONDS
import ast
import time
start_time = time.time()
import numpy
import re
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
corpusroot = './presidential_debates'
tfidfvec="./tfidf"

##1. PREPROCESSING
##THIS METHOD PERFORMS PREPROCESSING UPON THE CORPUS. TEXT IN THE FILES IS
##CONVERTED TO LOWERCASE, TOKENIZED AND STEMMED. IT CREATES A TEMPORARY GLOBAL DICTIONARY
def preproc():
    dic={}
    tokenizer = RegexpTokenizer(r'[a-z]+')
    stemmer=PorterStemmer()
    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read().lower()
        doc=doc.replace("\n", " ")
        tokens=list(tokenizer.tokenize(doc))
        newtok=[]
        stop=list(stopwords.words('english'))
        for i in tokens:
            if i not in stop:
                newtok.append(stemmer.stem(i))
        dic[filename]=newtok
    return dic


dic=preproc()

##THIS LOOP PREPROCESSES ALL THE DOCUMENTS IN THE CORPUS.
##AND DECLARES THE DICTIONARY IN GLOBAL SPACE
##dic={}
##for filename in os.listdir(corpusroot):
##    dic[filename]=preproc(filename)


##2.GETIDF(): THIS METHOD CALCULATES THE IDF OF ANY TOKEN IN THE PREPROCESSED CORPUS.
##IT TAKES A STEMMED TOKEN AS AN ARGUMENT AND RETURNS THE INVERSE DOCUMENT FREQUENCY
def getidf(token):
    df=0
    for i in dic:
        if token in dic[i]:
            df+=1    
    if df==0:
        return -1
    else:
        return math.log10(30/df)
##3. GETWEIGHT() THIS METHOD CALCULATES THE NORMALIZED TF-IDF VALUE OF A TOKEN, IN A FIILENAME
def getweight(filename,token):
    x=0
    l=[]
    norm=0
    check=[]
    toktf=0
    for i in dic[filename]:
        if i not in check:
            tf=dic[filename].count(i)
            l.append((1+math.log10(tf))*getidf(i))
        if i==token:
            toktf=dic[filename].count(token)
        
        check.append(i)
        
    norm =numpy.linalg.norm(l)
    if toktf==0:
        return 0.0
    else:
        return (((1+math.log10(toktf))*getidf(token))/norm)


##THIS ADDITIONAL METHHOD CALCULATES THE TFIDF VECTOR FOR EACH DOCUMENT
##AND STORES THEM IN A SET OF FILES
def writeweight():
    dic2={}
    m=[] 
    check=[]
    for filename in dic:
        for i in dic[filename]:
            if i not in check:
                tf=dic[filename].count(i)
                tfidf=(1+math.log10(tf))*getidf(i)
                m.append(tfidf)
                dic2[i]=tfidf 
            check.append(i)
        norm =numpy.linalg.norm(m)
        for i in dic2:
            dic2[i]=dic2[i]/norm
        with open(os.path.join(tfidfvec, filename), "w", encoding='UTF-8') as f:
            f.write(str(dic2))
        dic2={}
        check=[]
        m=[]
        print(filename+" done!")
        print("--- %s seconds ---" % (time.time() - start_time))
        
    
## THIS ADDITIONAL METHOD PRE PROCESSES THE INPUT QUERY AND RETURNS ITS TOKEN      
## LIST AND ITS WEIGHT VECTOR
def qvector(query):
    check=[]
    v=[]
    stem=""
    new=[]
    temp=[]
    tokenizer = RegexpTokenizer(r'[a-z]+')
    stemmer=PorterStemmer()
    query=query.lower()
    tokens = tokenizer.tokenize(query)
    tokens=list(tokens)
    stop=list(stopwords.words('english'))
    for i in tokens:
        if i not in stop:
            new.append(i)
    for i in new:
        stem=stem+" "+stemmer.stem(i)
    stem=stem[1:]
    temp=stem.split()
    for i in temp:
        if i not in check:
            tf=len(re.findall(r'\b'+i+r'\b', stem))
            v.append((1+math.log10(tf)))
        check.append(i)
    
    norm =numpy.linalg.norm(v)
    v=list(numpy.array(v)/norm)
    return ([stem,v])

##4. QUERY(): THIS METHOD TAKES A STRING AS INPUT AND RETURNS A TUPLE IN THE FORM:
##['DOC_NAME', SCORE]
def query(string):
    metric=qvector(string)
    vector=metric[1]
    line=metric[0]
    a=line.split()
    tes={}
    tes2={}
    temp=[]
    postings=[]
    final=[]
    count=0
    for i in a:
        tes[i]={}
        for filename in os.listdir(tfidfvec):
            file=open(os.path.join(tfidfvec, filename), "r", encoding='UTF-8')
            doc=file.read()
            dic=ast.literal_eval(doc)
            
            if i in dic:
                tes[i][filename]=dic[i]
            else:
                tes[i][filename]=0
    ##THIS IS THE PART WHERE THE POSTING LIST IS CALCULATED FOR EACH TOKEN
    for i in tes:
        for j in tes[i]:
            temp.append(tes[i][j])
        temp.sort(reverse=True)
        temp=temp[:10]
        for k in temp:
            for j in tes[i]:
                if tes[i][j]==k:
                    postings.append(j)
        tes2[i]={}
        tes2[i]=postings
        temp=[]
        postings=[]
    ##AT THE END OF THE LOOP, WE GET THE DICTIONARY 'TES2', WHICH CONTAINS THE
    ##POSTING LISTS OF EACH TOKEN IN THE QUERY
    bound={}
    var=[]
    ##HERE WE CHECK FOR DOCUMENTS WHICH ARE IN ALL TOKEN POSTING LISTS AND
    ##APPEND THEM TO A FINAL LIST FOR COSINE SIMILARITY CALCULATIONS
    ##IF DOCUMENT IS PRESENT IN ALL POSTING LISTS EXCEPT IN ONE TOKEN,
    ##THOSE DOCUMENTS ARE ALSO ADDED TO A BOUND LIST, FOR UPPER BOUND CALCULATIONS
    for i in tes2:
        for j in tes2[i]:
            count=0
            for k in tes2:
                if j in tes2[k]:
                    count+=1
                if j not in tes2[k] and k not in var:
                    var.append(k)   
            if count==len(a) and j not in final:
                final.append(j)
            elif count<len(a) and j not in final:
                bound[j]=[]
                bound[j]=var
                var=[]
    r={}
    t={}
    s=0
    b=[]
    residual=[]
    
    if final==[]:
        for i in bound:
           
            if len(bound[i])==len(a)-1:
                residual.append(i)
        
        for i in residual:
            file=open(os.path.join(tfidfvec, i), "r", encoding='UTF-8')
            read=file.read()
            r=ast.literal_eval(read) 
            for j in a:
                if j in bound[i] and j in r:
                    b.append(r[j])
                if j not in bound[i]:
                    file=open(os.path.join(tfidfvec, tes2[j][-1]), "r", encoding='UTF-8')
                    read=file.read()
                    r=ast.literal_eval(read) 
                    b.append(r[j])
            s=sum([k*l for k,l in zip(vector,b)])
            t[i]=s
            s=0
            b=[]
        score=0
        document=""
        for i in t:
            if t[i]>score:
                score=t[i]
                document=i
        return [document,score]        
    ##IF FINAL LIST HAS VALUES i.e THERE ARE DOCUMENTS PRESENT IN POSTING LISTS
    ##OF ALL QUERY TOKENS, THEN COSINE SIMILARITY SCORES ARE CALCULATED AND COMPARED
    ##WE RETURN THE DOCUMENT WITH THE HIGHEST COSINE SIMILARITY SCORE
    else:
        for i in final:
            file=open(os.path.join(tfidfvec, i), "r", encoding='UTF-8')
            read=file.read()
            r=ast.literal_eval(read)
            for j in a:
                if j in r:
                    b.append(r[j])
            s=sum([k*l for k,l in zip(vector,b)])
            t[i]=s
            s=0
            b=[]
        score=0
        document=""
    
        for i in t:
            if t[i]>score:
                score=t[i]
                document=i
        return [document,score]


#######################    TEST CASES    #######################
#print(qvector("health health insurance wall street"))                    

print(query("health insurance wall street"))

#(2012-10-03.txt, 0.033877975254)

print(query("particular constitutional amendment"))

#(fetch more, 0.000000000000)

print(query("terror attack"))

#(2004-09-30.txt, 0.026893338131)

print(query("vector entropy"))

#(None, 0.000000000000)

print(getweight("2012-10-03.txt","health"))

#0.008528366190

print(getweight("1960-10-21.txt","reason"))

#0.000000000000

print(getweight("1976-10-22.txt","agenda"))

#0.012683891289

print(getweight("2012-10-16.txt","hispan"))

#0.023489163449

print(getweight("2012-10-16.txt","hispanic"))

#0.000000000000

print(getidf("health"))

#0.079181246048

print(getidf("agenda"))

#0.363177902413

print(getidf("vector"))

#-1.000000000000

print(getidf("reason"))

#0.000000000000

print(getidf("hispan"))

#0.632023214705

print(getidf("hispanic"))

#-1.000000000000

print("--- %s seconds ---" % (time.time() - start_time))

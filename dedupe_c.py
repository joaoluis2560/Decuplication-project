import nltk
import ssl
import numpy as np
from nltk.tokenize import sent_tokenize, RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import pymysql
import itertools
from itertools import chain 
from struct import *
from scipy.sparse import csr_matrix
import os
import sparse_dot_topn.sparse_dot_topn as ct
import string
from pandas import DataFrame
from nltk.corpus import stopwords as sw
import json
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
from fuzzywuzzy  import fuzz, process
import time
import difflib  
from functools import partial
import pandas as pd
from sqlalchemy import create_engine
import os
from pyjarowinkler.distance import get_jaro_distance
#from settings import 

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('rslp')



class DedupeModel (object):
    
    def __init__(self, file= False ):
        self.file = file
        
    
    def open_a (self, js):
    
        lista = list(js.values())
        nw=[]
        for i in lista[0]: 
             val = i.values() 
             nw.append(list(val))
        return (nw)
    
    def tokenizer (self, nw):
        nova_lista = []
        lista_ids = []
        for i in nw:
            popped_element = i.pop(1)
            nova_lista.append(popped_element)
            lista_ids.append(i)
        
        return nova_lista
    
    def remove_acento(self, lista):
        text = [x.replace('á', 'a') for x in lista]
        text = [x.replace('é', 'e') for x in text]
        text = [x.replace('í', 'i') for x in text]
        text = [x.replace('ó', 'o') for x in text]
        text = [x.replace('ú', 'u') for x in text]
        text = [x.replace('à', 'a') for x in text]
        text = [x.replace('ã', 'a') for x in text]
        text = [x.replace('õ', 'o') for x in text]
        text = [x.replace('ç', 'c') for x in text]
        text = [x.replace('â', 'a') for x in text]
        text = [x.replace('ê', 'e') for x in text]
        text = [x.replace('ô', 'o') for x in text]

        return text
    
    def remove_sinais (self, lista):
        irrelevant_regex = re.compile(r'[^A-Za-z0-9\s]')
        text = [irrelevant_regex.sub(' ', c) for c in lista]
        return text
    
    def remove_space(self,lista):
        #tv
        s = [re.sub(r'\s+(?=\d\d\d\d)','', i) for i in lista]
        s = [re.sub(r'(?<=[A-Z])\s+(?=\d+)','', i) for i in lista]
        s = [re.sub(r'(?<=\d) +(?=\d)','', i) for i in s]
        
        return s
    
    
    def busca_padrao_p1 (self, lista): 
        regex_tv =  r'(\S*\d\d+\S*)|(\d\S\S\S+\d*)|(\S*\d+\S*)|(\w[A-Z]+)'
        tokens2 ={}
        tokens ={}
        for value, i in enumerate(lista):
            if re.search(regex_tv, i):
                tk = re.findall(regex_tv,i)
                a = [tuple(j for j in i if j)[-1] for i in tk]
                #token = tk.group()
                tokens.setdefault(value, []).append(a)
                tokens.setdefault(value, []).append(i)
            else:
                tokens2.setdefault(value, []).append([])
                tokens2.setdefault(value, []).append(i)
        z = {**tokens, **tokens2}

        return z
    
    def id_to_dict(self, x):
        corpusb = sum(x, [])
        tokens_id ={}
        for value, i in enumerate(corpusb):
            tokens_id.setdefault(value, []).append(i)
        return tokens_id
    
    def merge (self,tokens, tokens_id):
        join_d = { key:tokens.get(key,[])+tokens_id.get(key,[]) for key in set(list(tokens.keys())+list(tokens_id.keys())) }
        return(join_d)
    
    def define_tokens(self, tokens):
        df = pd.DataFrame.from_dict(tokens, orient='index')
        df.columns = ['id_familia', 'descricao', 'internalId']
        cols = list(df.columns)
        cols = [cols[-1]] + cols[:-1]
        df = df[cols]
        return df
    
    def stop_words(self,df):
        stop = stopwords.words('portuguese')
        df['produto'] = df['descricao'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df_s = df.drop(['descricao'], axis=1)
        return df_s
    
    def stemmer_(self,df):
        stemmer = nltk.stem.RSLPStemmer()
        df['produto'] = df['produto'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))
        return df
    
    def abbreviation (self, df_):
        abb_dictionary = {"Masc":"masculino" , 'Fem': "feminino"}
        d2 = {r'(\b){}(\b)'.format(k):r'\1{}\2'.format(v) for k,v in abb_dictionary.items()}
        df_['produto'] = df_['produto'].replace(d2, regex=True)
        df_['id_familia'] = [''.join(map(str, l)) for l in df_['id_familia']]
        return df_

    def read_dedupe(self):
        connection = pymysql.connect('35.198.44.64','ms_campaing','zHL95Wr5vEPjVhdt','occVerifyDeduplicationWorker');
        cursor = connection.cursor();
        engine = create_engine("mysql+pymysql://{user}:{pw}@35.198.44.64/{db}"
                .format(user="ms_campaing",
                        pw="zHL95Wr5vEPjVhdt",
                        db="occVerifyDeduplicationWorker",
                        charset='utf8',
                        cursorclass=pymysql.cursors.DictCursor))
        
        try:
            with connection.cursor() as cursor:
                sql = 'SELECT * FROM verifyDeduplication'
                dedupe = pd.read_sql(sql, con=connection)
            return dedupe 
            connection.commit()
        finally:
            connection.close()
            
    
    def read_pickle(self):
        connection = pymysql.connect('35.198.44.64','ms_campaing','zHL95Wr5vEPjVhdt','occVerifyDeduplicationWorker');
        cursor = connection.cursor();
        engine = create_engine("mysql+pymysql://{user}:{pw}@35.198.44.64/{db}"
                .format(user="ms_campaing",
                        pw="zHL95Wr5vEPjVhdt",
                        db="occVerifyDeduplicationWorker",
                        charset='utf8',
                        cursorclass=pymysql.cursors.DictCursor))
        
        try:
            with connection.cursor() as cursor:
                sql = 'SELECT * FROM trainDeduplication'
                train = pd.read_sql(sql, con=connection)
            return train
            connection.commit()
        finally:
            connection.close()

    def deduplication_learning (self, stop, dedupe):
            
        dataframe_resultado = pd.merge(stop, dedupe, on=['internalId', 'id_familia', 'produto'], how='left', indicator='Exist')
        dataframe_resultado['Exist'] = np.where(dataframe_resultado.Exist =='both', True, False)
        indexNames = dataframe_resultado[dataframe_resultado['Exist'] == True].index
        dataframe_resultado.drop(indexNames , inplace=True)
        dataframe_resultado.drop(['Exist'], 1,  inplace=True)
        d = dataframe_resultado.reset_index(drop=True)
        return d

    def verify_deduplication (self, abr, train):
        
        dataframe_resultado = pd.merge(abr, train, on=['internalId', 'id_familia', 'produto'], how='left', indicator='Exist')
        dataframe_resultado['Exist'] = np.where(dataframe_resultado.Exist =='both', True, False)
        indexNames = dataframe_resultado[dataframe_resultado['Exist'] == True].index
        dataframe_resultado.drop(indexNames , inplace=True)
        dataframe_resultado.drop(['Exist'], 1,  inplace=True)
        d = dataframe_resultado.reset_index(drop=True)
        return d
    
    def empty_df (self, d): 
        isempty =  d.empty
        return True if isempty == True else False
         
    
    def ngrams(self,string, n=3):
        string = re.sub(r'[,-./]|\sBD',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        ngrams =  [''.join(ngram) for ngram in ngrams]
        return ngrams
    
    def vectorizer(self, train,tk):
        
        tfidf_vectorizer = TfidfVectorizer(min_df=0, analyzer=lambda x: self.ngrams(x))
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(train['produto'])
        tfidf_matrix_test_id = tfidf_vectorizer.transform(tk['produto'])
        return tfidf_matrix_train,tfidf_matrix_test_id
    
    def awesome_cossim_top(self,tfidf_matrix_train, tfidf_matrix_test_id, ntop, lower_bound=0):
        # force A and B as a CSR matrix.
        # If they have already been CSR, there is no overhead
        
        A = tfidf_matrix_train.tocsr()
        B = tfidf_matrix_test_id.tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M*ntop

        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
                M, N, np.asarray(A.indptr, dtype=idx_dtype),
                np.asarray(A.indices, dtype=idx_dtype),
                A.data,
                np.asarray(B.indptr, dtype=idx_dtype),
                np.asarray(B.indices, dtype=idx_dtype),
                B.data,
                ntop,
                lower_bound,
                indptr, indices, data)

        x = csr_matrix((data,indices,indptr),shape=(M,N)) 
        return  0 if x.get_shape()==0 else x
    
    def get_matches(self, matches, A, B, top=100):
        non_zeros = matches.nonzero()
        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]
        
        if top:
            nr_matches = top

        else:
            nr_matches = sparsecols.size

        id_familia_base = np.empty([nr_matches], dtype=object)
        id_familia = np.empty([nr_matches], dtype=object)
        id_base = np.empty([nr_matches], dtype=object)
        id_produto= np.empty([nr_matches], dtype=object)
        displayName_base = np.empty([nr_matches], dtype=object)
        displayName = np.empty([nr_matches], dtype=object)
        similairity = np.zeros(nr_matches)
        
        for index in range(0, nr_matches):

            id_base[index] = A.iloc[sparserows[index]]['internalId']
            id_familia_base[index] = A.iloc[sparserows[index]]['id_familia']
            displayName_base[index] = A['produto'].iloc[sparserows[index]]
            id_produto[index] = B.loc[sparsecols[index]]['internalId']
            id_familia[index] = B.iloc[sparsecols[index]]['id_familia']
            displayName[index] = B['produto'].iloc[sparsecols[index]]
            similairity[index] = matches.data[index]

        return pd.DataFrame({'id_base': id_base,
                                     'id_familia_base': id_familia_base,
                                     'displayName_base': displayName_base,
                                     'internalId': id_produto, 
                                     'id_familia': id_familia,
                                     'displayName': displayName,
                                     'Similairity': similairity})
    
    def test_df (self, df): 
        isempty = df.empty
        return True if isempty == True else False
        
    def verify_df (self,x): 
        products = {}
        products['products'] = []
        for index, row in x.iterrows():
            q = row['internalId'], row['produto'],row['id_familia']
            products['products'].append({
            'index': index,
            'id': row['internalId'],
            'displayname': row['produto'],
            'token': row['id_familia'], 
            'cosine':  '0',
            'fuzzy':'0', 
            'humanvalidation': False,
            'newproduct': True,
            'deduplicated':False

            })
        json_formatted_str1 = json.dumps(products, indent=1, sort_keys=True)

        output1 = json.loads(json_formatted_str1)


        # retrain new product
        d = list(products.values())
        y = list(filter(lambda x: x['newproduct'] in [True], d[0]))
        id = [i["id"] for i in y]
        name = [i["displayname"] for i in y]
        token =[i["token"] for i in y]

        connection = pymysql.connect('35.198.44.64','ms_campaing','zHL95Wr5vEPjVhdt','occVerifyDeduplicationWorker');
        cursor = connection.cursor();
        engine = create_engine("mysql+pymysql://{user}:{pw}@35.198.44.64/{db}"
                .format(user="ms_campaing",
                        pw="zHL95Wr5vEPjVhdt",
                        db="occVerifyDeduplicationWorker",
                        charset='utf8',
                        cursorclass=pymysql.cursors.DictCursor))
       
       
        with connection.cursor() as cursor:
            df = pd.DataFrame(np.column_stack([id, token, name]),
                        columns=['internalId', 'id_familia', 'produto'])
            cols = "`,`".join([str(i) for i in df.columns.tolist()])
            for i,row in df.iterrows():
                sql = "INSERT INTO `trainDeduplication` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
                cursor.execute(sql, tuple(row))
        connection.commit()
        connection.close()
        return output1
        
    def jaro(self,df):
          df['Fuzzy'] = [get_jaro_distance(x, y) for x, y in zip(df['displayName_base'], df['displayName'])]
          return df              
        
    def fuzzy(self,df):
          df['Fuzzy'] = df.apply(lambda x: fuzz.token_sort_ratio(x['displayName_base'], x['displayName']), axis=1)
          return df
    
    def create_shingles(self,doc):
        k=1
        shingled_set = set() # create an empty set
        doc_length = len(doc) 
        # iterate through the string and slice it up by k-chars at a time
        for idx in range(doc_length - k + 1):
            doc_slice = doc[idx:idx + k]
            shingled_set.add(doc_slice)
        return shingled_set
    
    def compute_jaccard_sim(self, set1, set2):
        intersection = (set1 & set2)# using Python's symbol for intersection
        union = set1 | set2 # using Python's symbol for union
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity
    

    def hashing(self,df):
        df['str1'] = df.apply(lambda x: self.create_shingles(x['displayName_base']),axis=1)
        df['str2'] = df.apply(lambda x: self.create_shingles(x['displayName']),axis=1)
        df['Fuzzy'] = df.apply(lambda x: self.compute_jaccard_sim(x['str1'], x['str2']), axis=1)
        df = df.drop(columns=['str1', 'str2'], axis=1)
        return df
        
    def apply_sm(self, s, c1, c2): 
        c1 = c1.replace(' ', '')
        c2 = c2.replace(' ', '')
        return difflib.SequenceMatcher(None, s[c1], s[c2]).ratio()
    
    def validate_tokens(self, df): 
        df['str1'] = df.apply(lambda x: self.create_shingles(x['id_familia_base']),axis=1)
        df['str2'] = df.apply(lambda x: self.create_shingles(x['id_familia']),axis=1)
        df['Similarity_tokens'] = np.where((df['str1'] == "") & (df['str2'] == ""),
         's/t',df.apply(lambda x: self.compute_jaccard_sim(x['str1'], x['str2']), axis=1))
        df = df.drop(columns=['str1', 'str2'], axis=1)
        return df
    
    def threashold (self, df): 
       df.loc[(df['Fuzzy'] >= 0.9)  & (df['Similarity_tokens'] >='0.9'), 'Label'] = 'dedupe' 
       df.loc[(df['Fuzzy'] >=0.9)  & (df['Similarity_tokens'] =='s/t'), 'Label'] = 'dedupe' 
       df.loc[(df['Fuzzy'] >=0.9)  & (df['Similarity_tokens'] <'0.9'), 'Label'] = 'Human Validation'
       df.loc[((df['Fuzzy'] >= 0.42) & (df['Fuzzy'] <0.9))  & (df['Similarity_tokens'] >'0.5'), 'Label'] = 'Human Validation'
       df.loc[((df['Fuzzy'] >= 0.42) & (df['Fuzzy'] <0.9))  & (df['Similarity_tokens'] =='s/t'), 'Label'] = 'Human Validation'
       df.loc[((df['Fuzzy'] >= 0.42) & (df['Fuzzy'] <0.9))  & (df['Similarity_tokens'] <'0.5'), 'Label'] = 'Human Validation'
       df.loc[(df['Fuzzy'] < 0.42)  & (df['Similarity_tokens'] >='0.5'), 'Label'] = 'Human Validation'
       df.loc[(df['Fuzzy'] < 0.42)  & (df['Similarity_tokens'] =='s/t'), 'Label'] = 'New_product' 
       df.loc[(df['Fuzzy'] < 0.42)  & (df['Similarity_tokens'] <'0.5'), 'Label'] = 'New_product'
       return df

    def parse_df (self, df_, testes):
        df_compare = df_.loc[(df_['Label'] != 'New_product')]
        df_compare = df_compare.groupby('internalId', group_keys=False).apply(lambda x: x.loc[x.Fuzzy.idxmax()])
        df_compare = df_compare[['internalId','Label']]
        testes_compare = testes[['internalId']]
        df = pd.concat([df_compare, testes_compare]) 
        df = df['internalId'].drop_duplicates(keep=False)
        df_new_product = df.to_frame()
        df_new_product['Label'] = 'New_product'
        df_final  = df_compare.append(df_new_product)
        x= df_final.set_index('internalId')
        df1 = x.reindex(index=testes['internalId'])
        df_final = df1.reset_index()
        df_final= df_final.merge(testes, on='internalId', how='inner')
        df_final=  df_final.rename(columns={"produto": "displayName"})
        x = df_.append(df_final, ignore_index = False)
        df_classe = x.drop_duplicates(subset=['internalId'])
        df_classe = df_classe.reset_index()
        df1 = df_classe.replace(np.nan, 0.0, regex=True)
        return  df1


    
    def output(self,df):
        products = {}
        products['products'] = []
        for index, row in df.iterrows():
            q = row['internalId'], row['displayName'],row['id_familia'], row['Similairity'],row['Similarity_tokens'], row['id_base'], row['displayName_base']
            products['products'].append({
            'index': index,
            'id': row['internalId'],
            'displayname': row['displayName'],
            'token': row['id_familia'], 
            'cosine':  row['Similairity'],
            'fuzzy':row['Fuzzy'], 
            'humanvalidation': True if row['Label'] == "Human Validation" else False,
            'newproduct': True if row['Label'] == "New_product" else False,
            'deduplicated': True if row['Label'] == "dedupe" else False,
            'suggestionproduct': [{'id': row['id_base'],'displayname':row['displayName_base']}]

            })
        json_formatted_str = json.dumps(products, indent=1, sort_keys=True)

        output = json.loads(json_formatted_str)
        
        #retrain deduplicated
        d_treino = list(products.values())
        y_treino = list(filter(lambda x: x['deduplicated'] in [True],  d_treino[0]))
        
        id_treino = [i["id"] for i in y_treino]
        name_treino = [i["displayname"] for i in y_treino]
        token_treino =[i["token"] for i in y_treino]

        connection = pymysql.connect('35.198.44.64','ms_campaing','zHL95Wr5vEPjVhdt','occVerifyDeduplicationWorker');
        cursor = connection.cursor();
        engine = create_engine("mysql+pymysql://{user}:{pw}@35.198.44.64/{db}"
                .format(user="ms_campaing",
                        pw="zHL95Wr5vEPjVhdt",
                        db="occVerifyDeduplicationWorker",
                        charset='utf8',
                        cursorclass=pymysql.cursors.DictCursor))

        
        with connection.cursor() as cursor:
            df = pd.DataFrame(np.column_stack([id_treino, token_treino, name_treino]),
                        columns=['internalId', 'id_familia', 'produto'])
            cols = "`,`".join([str(i) for i in df.columns.tolist()])
            for i,row in df.iterrows():
                sql = "INSERT INTO `verifyDeduplication` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
                cursor.execute(sql, tuple(row))
            d = list(products.values())
            y = list(filter(lambda x: x['newproduct'] in [True], d[0]))
            id = [i["id"] for i in y]
            name = [i["displayname"] for i in y]
            token =[i["token"] for i in y]
        with connection.cursor() as cursor:
            df = pd.DataFrame(np.column_stack([id, token, name]),
                        columns=['internalId', 'id_familia', 'produto'])
            
            cols = "`,`".join([str(i) for i in df.columns.tolist()])
            for i,row in df.iterrows():
                sql = "INSERT INTO `trainDeduplication` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
                cursor.execute(sql, tuple(row))
        connection.commit()
        connection.close()
        return output
            
        

    def predict(self, file=False):
        open_data =  DedupeModel.open_a(self, file)
        tokenizer = DedupeModel. tokenizer(self, open_data)
        acento = DedupeModel. remove_acento(self, tokenizer)
        sinais = DedupeModel. remove_sinais(self, acento)
        espaco = DedupeModel. remove_space(self, sinais)
        tokens = DedupeModel.busca_padrao_p1(self, espaco)
        ids = DedupeModel.id_to_dict(self, open_data)
        merge = DedupeModel.merge(self, tokens, ids)
        query = DedupeModel.define_tokens(self, merge) #token teste
        stop = DedupeModel.stop_words(self, query)
        stem = DedupeModel.stemmer_(self, stop)
        abr = DedupeModel.abbreviation(self, stem) 
        learning_df = DedupeModel.read_dedupe(self) 
        df_b = DedupeModel.deduplication_learning (self, abr, learning_df) 
        dfa = DedupeModel.read_pickle(self)
        dfc = DedupeModel.verify_deduplication(self, df_b, dfa)
        verify = DedupeModel.empty_df(self, dfc)
        if verify == True:
            return 'O(s) produto(s) já existe(m) na base'
        else: 
            tfidf_matrix_train,tfidf_matrix_test_id = DedupeModel.vectorizer(self,dfa,dfc)
            matches = DedupeModel.awesome_cossim_top(self,tfidf_matrix_train, tfidf_matrix_test_id.transpose(),1,0.4) 
        if matches.nnz == 0:
            verify_df = DedupeModel.verify_df(self,dfc)
            return verify_df
        else:
            get = DedupeModel.get_matches(self,matches, dfa, dfc, top=0)
            teste = DedupeModel.test_df(self,get)
        if teste == True:
            verify_df = DedupeModel.verify_df(self,dfc)
            return  verify_df
        else:
            hashi = DedupeModel.hashing(self, get)
            #fuzzy = DedupeModel.fuzzy(self, get)  
            validation = DedupeModel.validate_tokens(self,hashi)
            threashold = DedupeModel.threashold(self,validation)
            df_output = DedupeModel.parse_df(self, threashold, dfc) 
            out =  DedupeModel.output(self,df_output)
            return out

        
    
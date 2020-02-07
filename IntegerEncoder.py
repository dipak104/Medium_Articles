from sklearn.pipeline import TransformerMixin, make_pipeline
from sklearn import preprocessing

class IntegerEncoder(TransformerMixin):
    def __init__(self,columns = None):
        self.columns = list(columns)
        
    def fit(self,X,y=None,**kwargs):
        col_categories = {}
        for col in self.columns:
            le = preprocessing.LabelEncoder()
            col_categories[col] = le.fit(X[col])
        self._col_categories = col_categories
        return self
    
    def transform(self,X,y=None,**kwargs):
        for col,le in self._col_categories.items():
            X[col] = le.transform(X[col])
        return X

cat_cols = ["RepID","Area","Detail","category"]
pipeline = make_pipeline(IntegerEncoder(columns=cat_cols))

train = pipeline.fit(df_1).transform(df_1.iloc[:10,:])
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn import preprocessing
from flask import Flask,abort,jsonify,request
from sklearn.pipeline import TransformerMixin, make_pipeline

class DummyEncoder(TransformerMixin):
    def __init__(self,columns = None,drop_first = True):
        self.columns = list(columns)
        self.drop_first = drop_first
        
    def fit(self,X,y=None,**kwargs):
        col_categories = {}
        for col in self.columns:
            col_categories[col] = X[col].astype('category').dtype.categories
        self._col_categories = col_categories
        return self
    def transform(self,X,y=None,**kwargs):
        df = X.apply(lambda col: (col if col.name not in self._col_categories
                                 else pd.Categorical(col,categories = self._col_categories[col.name])))
        return pd.get_dummies(df,columns = self.columns, drop_first = self.drop_first)

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

        
from flask import request
from flask_api import FlaskAPI
app = FlaskAPI(__name__)

@app.route('/call_router', methods=['POST'])        
def call_router():
    
    data = request.json 
    """data = { "data":[  {  
         "RepID":"0GTE",
         "TaskOriginator":"TSHD",
         "Detail":"Password Reset",
         "ErrorSource":"Other",
         "SubStatus":"Unassigned",
         "Priority":"Standard",
         "ReceivedVia":"OutBound Call",
         "ReOpenFlag":"N",
         "SLADays":"0.0"}]}"""
    print(data)
    df = pd.DataFrame(data["data"])
    
    """df = df.drop(['CompletedBy','CompletedDate','RepCallBack','ForOpsUseOnly','FollowUpDate','Method','Origination','ErrTradeDt','ErrCuspid','ErrNumShares','ErrPricePerShare','ErrProductClass','CorrAcctNum','CorrCUSIP','CorrShares',
				  'CorrFundName','TradeDateCT','CUSIPCT','SharesCT','PrincipalCT','TradeType','ContactId','AccountType','PIPS','NoMarketRisk','SPAD','InstantKBCheck','POICheck','SupCheck',
				  'SMECheck','OpsCheck','OtherCheck','ClosedIndicator','ContactedBy','SRFeeAmt','Who','ClubLevel','CutoffTime','NPS_SurveyDate','Overall_NPS','RelatedID','ServiceSupport_NPS','ActivityLastUpdated',
				  'OpenActivities','TotalActivities','ReceivedDate','TradeLoss','Verified','TicketCharge','BilledLoss','CorrectionDayPrice','GLPerShare','MarketShares','Err_ClientAccountNumber','Err_PrincipalAmount','Err_SettlementDate',
				  'Err_FundAccountNumber','Err_FundName','Err_Type','Err_ControlNumber','Err_Security','Err_SPAD','Err_PIPS','Err_NoMarketRisk','Err_PAGCorrections','Corr_PriceShare','OriginalClosedDate','Department','Service360Team','Tier1Why','ClosedOnFirstContact','Status','RowId','Created','CreatedId','LastUpdated','SRNumber','LastName','FirstName','CorrSecNum','CommittedDate','CorrPrincipal','RootCause','DefectNum','CallType','Owner','CreatedBy','Closed','ClosedBy','ContactedByLastName','ContactedByFirstName','LastUpdatedby','SubArea','AccountNumber'],axis=1)"""
	# Dataframe Creation			  
    df = df[['RepID', 'TaskOriginator','Detail', 'ErrorSource', 'SubStatus','Priority', 'ReceivedVia', 'ReOpenFlag', 'SLADays']]
    
    df.RepID = df.RepID.astype('object')
    df.TaskOriginator = df.TaskOriginator.astype('object')
    #df.Area  = df.Area.astype('object')
    df.Detail  = df.Detail.astype('object')
    df.ErrorSource  = df.ErrorSource.astype('object')
    df.SubStatus  = df.SubStatus.astype('object')
    df.Priority  = df.Priority.astype('object')
    df.ReceivedVia = df.ReceivedVia.astype('object')
    df.ReOpenFlag  = df.ReOpenFlag.astype('object')
    df.SLADays  = df.SLADays.astype('float64')
    
   
    #Handling Categorical Columns (One Hot Encoding) 
    df['TaskOriginator'] = pd.Categorical(df['TaskOriginator'],ordered=True,categories=['Tier 1 Service Center', 'TSHD', 'Service Center',
       'Service Team Processing', 'Investor/Client Escalation Team',
       'Registration Advisor Items', 'Cash Management',
       'Specialized Team-Account Transfers',
       'Specialized Team-New Accounts', 'Mgr Escalations',
       'Client Technology', 'BranchLink LPL Express',
       'Retirement Accounts'])
    
    df['ErrorSource'] = pd.Categorical(df['ErrorSource'],ordered=True,categories=['No Error', 'Other', 'Incorrect/Incomplete Processin',
       'Advisor/Customer Error', 'Warm Transfer', 'Misinformation',
       'Technology Issue', 'TSHD CW Specialized', 'Senior Escalation',
       'Management Escalation']) 
    
    df['SubStatus'] = pd.Categorical(df['SubStatus'],ordered=True,categories=['Unassigned', 'Escalated', 'Resolved', 'Assigned', 'Developers',
       'Production Issue', 'Settlement', 'For Review',
       'Multiple Dept Task', 'Waiting Response - Advisor'])

    df['Priority'] = pd.Categorical(df['Priority'],ordered=True,categories=['Standard', 'Escalated'])
    
    df['ReOpenFlag'] = pd.Categorical(df['ReOpenFlag'],ordered=True,categories=['N', 'Y'])
    
    df['ReceivedVia'] = pd.Categorical(df['ReceivedVia'],ordered=True,categories=['Phone', 'OutBound Call', 'Ask Anything', 'Email', 'POI Line',
       'Client Line', 'Service Center', 'SME Line', 'Proactive Callout',
       'Proactive Research'])
    
    cat_cols1 = ['TaskOriginator','ErrorSource','SubStatus','Priority','ReOpenFlag','ReceivedVia']
    cat_cols2 = ['RepID','Detail']
    pipeline = make_pipeline(DummyEncoder(columns=cat_cols1), IntegerEncoder(columns=cat_cols2))   
    train = pipeline.fit(df).transform(df.iloc[:10,:])
    print("======================1")
    print(train)
    print("======================2")
    #Handling Categorical Columns (Integer Encoding) 
    #df["RepID"] = pd.Categorical
    model_2 = joblib.load('C:/Users/ryadav/Desktop/LPL Recommendation Engine/4.CallRouterPrediction/DevProd/model_RF_CallRouterPrediction.pkl')
	#predictions = model_2.predict(train)
    output = model_2.predict_proba(train)
    final_predictions = pd.DataFrame(list(output),columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']).to_dict(orient="records")
    final_predictions = jsonify(final_predictions)
    return (final_predictions)

	

@app.route('/test', methods=['GET'])        
def test():
	return ("Test - #{Octopus.Project.Name} - #{Octopus.Release.Number} - Octopus Deployed")
	
app.run(debug=False,host = '10.52.17.60', port = 5002)
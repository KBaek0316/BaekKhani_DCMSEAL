# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:12:12 2024

@author: baek0040@umn.edu Kwangho Baek
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
if __name__=="__main__":
    if os.environ['USERPROFILE']=='C:\\Users\\baek0040':
        WPATH=r'C:\Users\baek0040\Documents\GitHub\DCM_SEAL'
    else:
        WPATH=os.path.abspath(r'C:\git\DCM_SEAL')
    pd.set_option('future.no_silent_downcasting', True)
    os.chdir(WPATH)

#%% Path Data
def pathDataGen(surFile,pathFile,convFile=None,TRB=False,tivdomcut=0,minxferpen=1,abscut=15,propcut=2,depcut=None,strict=True):
    '''for debugging
    surFile='survey2022.csv'
    pathFile='paths2022.csv'
    convFile='dfConv.csv'
    tivdomcut=0.2
    minxferpen=3
    abscut=20
    propcut=1.5
    strict=True
    depcut=15
    TRB=False
    '''
    ver=int(os.path.splitext(surFile)[0][-4:]) #either 2016 or 2022
    dfSurveyRaw=pd.read_csv(surFile,low_memory=False, encoding='ISO 8859-1')
    match ver:
        case 2016:
            pass
        case 2022:
            dfSurvey=dfSurveyRaw.loc[:,['id','collection_type','date_type','origin_place_type','destin_place_type',
                                     'plan_for_trip','realtime_info','do_you_drive', 'used_veh_trip','hh_size',
                                     'hh_member_travel','origin_transport','destin_transport','trip_in_oppo_dir',
                                     'oppo_dir_trip_time','gender_male', 'gender_female','race_white','resident_visitor',
                                     'work_location','student_status','english_ability', 'your_age','income', 'have_disability']]
            print(f'Deleted cols: {np.setdiff1d(dfSurveyRaw.columns,dfSurvey.columns)}')
            dfSurvey.columns=['id','summer','dayofweek','typeO','typeD','plan','realtime','candrive','cdhvusdveh','hhsize',
                              'HHcomp','access','egress','oppo','oppotime','male','female','white','visitor','worktype','stu',
                              'engflu','age','income','disability']
    try: # To refactor some categorical variables from the survey format to model-able
        dfConversion=pd.read_csv(convFile)
        dfConversion=dfConversion.loc[(dfConversion.version==ver) & (dfConversion.step=='post'),:]
    except NameError: #use keygen to generate dfConv.csv
        keygen=dict()
        for col in dfSurvey.columns[1:]:
            elems=dfSurvey.loc[:,col].unique()
            if len(elems)<50:
                keygen[col]=elems.tolist()
        keygen=pd.Series(keygen, name='orilevel').rename_axis('field').explode().reset_index()
        keygen.to_clipboard(index=False,header=False)
        raise Exception('Variable factors conversion file (dfConv.csv) not found: use clipboard to generate this and run again')
    #Initialize minor, systematic, or evident missing values
    dfSurvey.fillna(value={'plan':'others','HHcomp':'0','worktype':'unemp','engflu':'1'},inplace=True)
    #refactoring some categorical variables from the survey format to model-able
    dfConversion=dfConversion.loc[(dfConversion.version==ver) & (dfConversion.step=='post'),:]
    for fld in dfConversion.loc[:,'field'].unique():
        dfSlice=dfConversion.loc[dfConversion.field==fld,['orilevel','newlevel']]
        dfSlice=pd.Series(dfSlice['newlevel'].values,index=dfSlice['orilevel'])
        dfSurvey=dfSurvey.replace({fld:dfSlice})
        try:
            dfSurvey[fld]=dfSurvey[fld].astype(float)
        except ValueError:
            pass
    # Reorganizing some vars
    removecols=['oppotime','sid','realDep','male','female'] #predefine cols to be dropped after newvars gen
    dfSurvey['gender']='others'
    dfSurvey.loc[dfSurvey['male']==1,'gender']='male'
    dfSurvey.loc[dfSurvey['female']==1,'gender']='female'
    if TRB: #if you want to use HTS-like travel purpose, 1he genders and not using origin/destination types
        #some variables are need to be defined using multiple survey responses
        dfSurvey['choicerider']='dependent'
        dfSurvey.loc[dfSurvey.candrive=='Yes','choicerider']='potentially'
        dfSurvey.loc[dfSurvey.cdhvusdveh=='Yes','choicerider']='choicerider'
        dfSurvey['nonbinary']=1
        dfSurvey.loc[(dfSurvey['male']+dfSurvey['female'])>0,'nonbinary']=0
        dfSurvey['purpose']='HB'
        dfSurvey.loc[(dfSurvey['typeO']!='Home') & (dfSurvey['typeD']!='Home'),'purpose']='NHB'
        dfSurvey.loc[dfSurvey.purpose=='HB','purpose']+=(dfSurvey.loc[dfSurvey.purpose=='HB','typeO']+dfSurvey.loc[dfSurvey.purpose=='HB','typeD']).str.replace('Home','')#.str[0]
        dfSurvey.loc[dfSurvey.purpose=='HB','purpose']='HBO' #there is only one instance whose O and D are both Home
        removecols+=['typeO','typeD','candrive','cdhvusdveh','duration','nonbinary']
    #move on to the path preprocessing; paths retrieved from the repository SchBasedSPwithTE_Pandas
    dfPathRaw=pd.read_csv(pathFile,low_memory=False, encoding='ISO 8859-1')
    print('Among '+str(len(dfPathRaw.sid.unique()))+' survey respondents examined,')
    dfPath=dfPathRaw.drop(columns=['detail','nodes','snap','elapsed','TE','hr'],errors='ignore').dropna(subset='realDep')
    print(str(len(dfPath.sid.unique()))+' respondents have at least one path identified from V-SBSP')
    dfPath['ntiv']=dfPath['iv']-dfPath['tiv'] #non-transitway IVT
    dfPath['aux']=dfPath['wk']+dfPath['nwk'] #access and egress time combined
    dfPath['ov']=dfPath['aux']+dfPath['wt'] #out-of-vehicle time
    dfPath['tt']=dfPath.iv+dfPath.ov
    dfPath['cost']=dfPath.tt+minxferpen*dfPath.nTrans #tiebreaker
    dfPath['tway']=0
    dfPath.loc[dfPath.tiv>dfPath.iv*tivdomcut,'tway']=1
    ## Added After TRBAM 2025 Submission: discard 'pairing' and allow up to 5 alts per agent
    dfPath=dfPath.loc[dfPath.iv>0,:]
    dfPath=dfPath.loc[dfPath.sid.isin(dfPath.loc[dfPath.match==1,'sid'].unique())]
    if depcut is not None:#Departure time restriction
        dfPath['matchDep']=dfPath.groupby('sid')['realDep'].transform(lambda x: x[dfPath['match'] == 1].values[0])
        dfPath=dfPath.loc[(dfPath.matchDep-dfPath.realDep).abs()<=depcut,:]
    #add reasonable choice set assumption for shorter-than-matching-paths: exclude dominant non-chosen paths
    dfPath['matchiv']=dfPath.groupby('sid')['iv'].transform(lambda x: x[dfPath['match'] == 1].values[0])
    dfPath['matchtt']=dfPath.groupby('sid')['tt'].transform(lambda x: x[dfPath['match'] == 1].values[0])
    dfPath['matchxf']=dfPath.groupby('sid')['nTrans'].transform(lambda x: x[dfPath['match'] == 1].values[0])
    dfPath['matchprop']=dfPath.matchiv/dfPath.matchtt
    dfPath=dfPath.loc[~( (dfPath.tt<=dfPath.matchtt)&(dfPath.iv/dfPath.tt<dfPath.matchprop) ),:]
    dfPath['spcost']=dfPath.groupby(['sid'])['cost'].transform('min')
    dfPath['compDiff']=dfPath.cost-dfPath.spcost
    dfPath['compProp']=dfPath.cost/dfPath.spcost
    dfPath=dfPath.loc[(dfPath.compDiff<abscut)|(dfPath.compProp<propcut),:]
    if strict:
        dfPath=dfPath.loc[(dfPath.compDiff<abscut)&(dfPath.compProp<propcut),:]
    pathfilter=dfPath.groupby('sid').agg({'tway':'sum','match':'sum','cost':['count','min','max']}).reset_index()
    pathfilter.columns=['sid','tway','match','counts','mint','maxt']
    pathfilter2=pathfilter.loc[(pathfilter.counts>1)&(pathfilter.match>0)  ,:] #   &(pathfilter.tway>0)
    dfPath=dfPath.loc[dfPath.sid.isin(pathfilter2.sid.unique()),:]#.drop(columns=['spcost','compDiff','compProp'])
    print(f'{len(dfPath.sid.unique())} choice sets generated ({100*(1-len(pathfilter2)/len(pathfilter)):.2f}% filtered)')
    dfSurvey=pd.merge(dfSurvey,dfPathRaw.loc[dfPathRaw.match==1,['sid','realDep']],left_on='id',right_on='sid')
    dfSurvey['hr']=dfSurvey.realDep//60
    #if TRB: #imputing
    #    from sklearn.impute import KNNImputer
    #    dfSurvey['duration']=dfSurvey.oppotime-dfSurvey.realDep/60
    #    dfSurvey['duration']=KNNImputer(n_neighbors=10,weights='distance').fit_transform(dfSurvey.drop(columns='id'))[:,np.where(dfSurvey.columns=='duration')[0][0]-1]
    dfSurvey=dfSurvey.drop_duplicates('id').drop(columns=removecols,errors='ignore').reset_index(drop=True)
    #final organization
    dfSurvey=dfSurvey.loc[dfSurvey.id.isin(pathfilter2.sid.unique()),:].reset_index(drop=True)
    dfPath=dfPath.drop(columns=['ind','label_t','label_c','spcost','compDiff','matchDep'],errors='ignore').rename(columns={"sid": "id"})
    dfPath['aux']=dfPath['wk']+dfPath['nwk']
    dfPath['ov']=dfPath['aux']+dfPath['wt']
    dfPath=dfPath.sort_values(['id','match']).reset_index(drop=True)
    return dfSurvey, dfPath

def attachPS(df):
    import geopandas as gpd
    from shapely import wkt
    #df=dfPath.loc[dfPath.id==78,:].copy() #for debugging
    df['geometry'] = df['line'].apply(wkt.loads)
    df['PS']=0.0
    def calcPS(geometry, node_frequency):
        PSin = 0
        nodes = set(list(geometry.coords)[1:-1])
        for node in nodes:
            PSin += np.log(node_frequency[node])
            PSin /= len(nodes)
        return PSin
    print('Calculating Path Size Factors')
    for sid in df.id.unique():
        dfi=df.loc[df.id==sid,['geometry','PS']].copy()
        node_frequency = {}
        for geom in dfi['geometry']:
            nodes = set(list(geom.coords)[1:-1]) #remove O and D points
            for node in nodes:
                if node in node_frequency:
                    node_frequency[node] += 1
                else:
                    node_frequency[node] = 1
        dfi['PS'] = dfi['geometry'].apply(lambda geom: calcPS(geom, node_frequency))
        df.update(np.round(dfi.PS,6))
    df=df.drop(columns=['line','geometry'])
    print('Path Size Correction Term has been calculated')
    return df

#%% Synthetic Data
def synthDataGen(nobs=1000):
    '''
    nobs=10000
    Class1: likely to be female (s1=='s12'), more concerned about in-vehicle safety and crowding (buses ('a') are small)
    U1=(0.8+0.2)*(mode=='b')+(-0.1-0.025)*x1+(-0.15)*x2+(-0.05)*x3
    Class2: likely to be male (s1=='s11') and non-white (s2!='s21') more concerned about out-of-vehicle safety except high-incomed
    U2=0.8*(mode=='b')+(-0.1)*x1+(-0.15-0.1)*x2+(-0.05)*x3+1*e23
    Class3: likely to be other demographic groups, baseline preference
    U3=0.8*(mode=='b')+(-0.1)*x1+(-0.15)*x2+(-0.05)*x3
    Embedding utilities are added uniformly to each class for mode b:
    Uadd=(mode=='b')*(1*e11+0.2*e12-0.2*e13+0.1*e21+0.2*e22+0.4*e23)
    '''
    o1 = np.random.choice(['o11', 'o12', 'o13','o14'], size=nobs, p=[0.2,0.3,0.4,0.1]) #e.g., favorite colors
    s1 = np.random.choice(['s11', 's12', 's13'], size=nobs, p=[0.47, 0.48, 0.05]) # e.g., male female others
    s2 = np.random.choice(['s21', 's22', 's23'], size=nobs, p=[0.78, 0.07, 0.15]) # e.g., white black others
    e1 = np.random.choice(['e11', 'e12', 'e13'], size=nobs, p=[0.47,0.40,0.13]) #e.g., HBW/S HBO NHB
    
    P_e2_given_s2 = {#e.g., linc minc hinc (col) for white black others (row) 
        's21': [0.2, 0.6, 0.2],
        's22': [0.4, 0.55, 0.05],
        's23': [0.4, 0.5, 0.1],}
    e2 = []
    for s in s2:
        e2.append(np.random.choice(['e21', 'e22','e23'], p=P_e2_given_s2[s]))
    e2 = np.array(e2)
    
    dist = np.round(2 + np.random.exponential(scale=4, size=nobs),2)
    x1a = np.round(2 * dist + 0.4 * dist * np.random.randn(nobs))  #ivt for mode a
    x1b = np.round(1.5 * dist + 0.1 * dist * np.random.randn(nobs)) #ivt for mode b
    x2a = np.round(5 + 1 * np.random.randn(nobs)) #ovt for mode a
    x2b = np.round(15 + 2 * np.random.randn(nobs))  #ovt for mode b
    x3a = np.round(1.5 + 0.5 * 1*(np.random.uniform(0,1,nobs)>0.8),2)
    x3b = np.round(2 + 0.75*np.floor(dist/5),2)
    
    #np.exp(3)/(2+np.exp(3))~0.045
    lc1=np.exp(3*(s1=='s12'))
    lc2=np.exp(3*(s1=='s11')*(s2!='s21'))
    lc3=np.exp(3*((lc1+lc2)==2))
    pr1=lc1/(lc1+lc2+lc3)
    pr2=lc2/(lc1+lc2+lc3)
    pr3=lc3/(lc1+lc2+lc3)
    prs=np.transpose(np.array([pr1,pr2,pr3]))
    LC = np.array([np.random.choice(len(row), p=row) for row in prs])+1
    
    dfC=pd.concat([pd.DataFrame({'id':(1+np.arange(nobs)),'alt':'a','x1':x1a,'x2':x2a,'x3':x3a}),
                   pd.DataFrame({'id':(1+np.arange(nobs)),'alt':'b','x1':x1b,'x2':x2b,'x3':x3b})]).sort_values(['id','alt'])
    dfN=pd.DataFrame({'id':(1+np.arange(nobs)),'s1':s1,'s2':s2,'e1':e1,'e2':e2,'o1':o1,'dist':dist,'LC':LC})
    
    ASC=0.6 #identified for mode b (train)
    bx1=-0.1 #ivt
    bx2=-0.15 #ovt
    bx3=-0.05 #fare
    be11=1 #HBW/S on mode b
    be12=0.2 #HBO on mode b
    be13=-0.2 #NHB on mode b
    be21=0.1 #linc on mode b
    be22=0.2 #minc on mode b
    be23=0.4 #hinc on mode b
    L1ASC=0.15 #ASC add-on for LC1
    L1x1=-0.025 #ivt add-on for LC1
    L2x2 = -0.1 #ovt add-on for LC2
    L2e23 = -0.5 #hinc utility penalty revert
    eps=0.15
    #delt=0.05
    
    #endogeneity defined for LC1 for interaction with x1 or iv, LC2 for interaction with ovt but except highincome LC2
    df=pd.merge(dfC,dfN,on='id')
    df['utility']=(df['alt']=='b')*(ASC+L1ASC*(df['LC']==1))+df['x1']*(bx1+L1x1*(df['LC']==1))+df['x2']*(bx2+L2x2*(df['LC']==2))+df['x3']*bx3
    df['utility']+=(df['alt']=='b')*(be11*(df['e1']=='e11')+be12*(df['e1']=='e12')+be13*(df['e1']=='e13')+be21*(df['e2']=='e21')+be22*(df['e2']=='e22')+be23*(df['e2']=='e23'))
    df['utility']+=L2e23*(df['alt']=='b')*(df['LC']==2)*(df['e2']=='e23')
    df['utility']+=eps*np.random.randn(len(df)) #random error
    #df['utility']+=delt*np.random.randn(len(df))*(df['LC']!=3) #saftey concern error; the LV concern has coef 1 (to def error simply) and have L params to be estimated by model
    
    dfa=df.loc[df.alt=='a','utility']
    dfb=df.loc[df.alt=='b','utility']
    dfchoice=pd.DataFrame({'id':(1+np.arange(nobs)),'choseA':dfa.values>=dfb.values})
    
    df=pd.merge(df,dfchoice,on='id')
    df['match']=0
    
    df.loc[(df.alt=='a') & (df.choseA),'match']=1
    df.loc[(df.alt=='b') & (~(df.choseA)),'match']=1
    df=df.drop(columns=['choseA'])
    _, testID = train_test_split(df.id.unique(), test_size=0.2, random_state=5723588)
    df['flag']='tr'
    df.loc[df.id.isin(testID),'flag']='ts'
    
    df['alt']=1*(df.alt=='b') #revert to 0 1 alts
    return df
#%% output
def preProcess(dataOption,nobs): #'path' or 'synth'
    if dataOption=='path': #path data
        doPathPreprocess=('dfPath.csv' not in os.listdir())
        if doPathPreprocess:
            print ('Generating Path Data and Store it in the Directory')
            dfSurvey, dfPath= pathDataGen('survey2022.csv','paths2022.csv','dfConv.csv',TRB=False,
                                              tivdomcut=0,minxferpen=1,abscut=15,propcut=2,depcut=45,strict=True)
            dfPath=attachPS(dfPath)
            dfPath['alt'] = dfPath.groupby('id').cumcount()
            dfPP=pd.merge(dfPath,dfSurvey,how='left',on='id')
            _, testID = train_test_split(dfPP.id.unique(), test_size=0.2, random_state=5723588)
            dfPP['flag']='tr'
            dfPP.loc[dfPP.id.isin(testID),'flag']='ts'
            dfPP.to_csv('dfPath.csv',index=False)
        else:
            print ('Loading Path Data')
            dfPP=pd.read_csv('dfPath.csv')
    else: #synthetic data
        doSynth=('dfSynth.csv' not in os.listdir())
        if doSynth or __name__=="__main__":
            print ('Generating Synthetic Data and Store it in the Directory')
            dfPP=synthDataGen(nobs=12500)
            dfPP.to_csv('dfSynth.csv',index=False)
        else:
            print ('Loading Synthetic Data')
            dfPP=pd.read_csv('dfSynth.csv')
    return dfPP

if __name__=="__main__":
    dataGen='Synth'
    dfSynth=preProcess(dataOption=dataGen,nobs=12500)
    dfPerson=dfSynth.loc[dfSynth.alt==0,:].copy().reset_index()
    print(dfPerson.groupby(['LC']).size()/len(dfPerson))
    print(dfSynth.groupby(['alt'])['match'].sum()/len(dfPerson))
    print(dfSynth.groupby(['LC','alt'])['match'].sum()/dfPerson.groupby(['LC']).size().repeat(2).values)
    enc=OneHotEncoder(sparse_output=False)
    dfPerson2=pd.DataFrame(enc.fit_transform(dfPerson[['s1','s2','e1','e2','o1']]),columns=enc.get_feature_names_out())
    dfPerson3=pd.concat([dfPerson[['LC']],dfPerson2],axis=1)
    dfPerson4=pd.concat([dfPerson3.groupby('LC').sum(),(dfPerson.groupby(['LC']).size())],axis=1)
    dfPerson5=round(dfPerson4.div(dfPerson4[0],axis=0),3).iloc[:,:-1]
    print(dfPerson5.transpose())

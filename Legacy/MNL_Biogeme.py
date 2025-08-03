# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:46:36 2024

@author: Kwangho Baek baek0040@umn.edu
"""
#%% Setup
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable #, DefineVariable, bioMultSum
from biogeme.models import loglogit
from biogeme.database import Database


# directory setting
if os.environ['USERPROFILE']=='C:\\Users\\baek0040':
    WPATH=r'C:\Users\baek0040\Documents\GitHub\DCM_SEAL'
else:
    WPATH=os.path.abspath(r'C:\git\DCM_SEAL')
pd.set_option('future.no_silent_downcasting', True)
os.chdir(WPATH)
del WPATH


# Hyper-hyper parameters
dataUsed='Synth' #'synth' or 'path'
if dataUsed=='path': #available pathattrs: 'nwk','wk','wt','ntiv','tiv','nTrans' ,'aux', 'ov', 'iv','tt','PS'
    dfMNL=pd.read_csv('dfPath.csv')
    attrsUsed=['tway','ov','nTrans','iv'] #TRB 
    dfMNL['tway']=0
    dfMNL.loc[dfMNL.tiv>0,'tway']=1
    attrcols=['iv','ov','nTrans','PS','tway','wt','aux','wk','nwk']
    if dfMNL.iv.min()<=0:
        raise Exception('remove iv<=0 paths')
    cate=[]
else:
    dfMNL=pd.read_csv('dfSynth.csv')
    attrsUsed=['x1','x2','x3']
    cate=['s1','s2','e1','e2','o1']


dfMNL['alt'] = dfMNL.groupby('id').cumcount() + 1 #redefine from 1 to 5 (in SEAL_DCM, 0 to 4, where 0 is deleted as reference alt)
maxalt=dfMNL.alt.max()

def ppForBiogeme(dfM,attrcols,dataUsed,cate):
    #dfM=dfMNL.copy()
    alts=np.arange(maxalt)+1
    comb = pd.MultiIndex.from_product([dfM['id'].unique(), alts], names=['id', 'alt'])
    dfLong = dfM.set_index(['id', 'alt']).reindex(comb, fill_value=0).reset_index()
    
    dfWide= dfLong.pivot(index='id', columns='alt',values=attrsUsed)
    dfWide.columns = [f'{col[0]}_{int(col[1])}' for col in dfWide.columns]
    dfWide=dfWide.reset_index()
    if len(cate)>0:
        enc=OneHotEncoder(sparse_output=False)
        dfEnc=pd.DataFrame(enc.fit_transform(dfM.loc[dfM.match==1,cate]))
        colnames=[x.split('_')[1] for x in enc.get_feature_names_out()]
        dfEnc.columns=colnames
        dfWide=pd.concat([dfWide,dfEnc],axis=1,ignore_index=False)
    if dataUsed=='path':
        for alt in alts:
            dfWide[f'avail_{alt}'] = (dfWide[f'iv_{alt}'] != 0).astype(int)
    else:
        for alt in alts:
            dfWide[f'avail_{alt}'] = 1
    choice=dfLong.loc[dfLong.match==1,['id','alt']]
    choice.columns=['id','choice']
    dfWide = pd.merge(dfWide, choice, on='id')
    dfTr=dfWide.loc[dfWide.id.isin(dfM.loc[dfM.flag=='tr','id']),:]
    dfTs=dfWide.loc[~dfWide.id.isin(dfM.loc[dfM.flag=='tr','id']),:]
    return dfTr, dfTs

dfTr, dfTs=ppForBiogeme(dfMNL,attrsUsed,dataUsed,cate)


database = Database('mymodel',dfTr)

# Define parameters (Betas) that will be estimated
BETA_IV = Beta('BETA_IV', 0, None, None, 0)  # Initial value is 0, no bounds
BETA_WT = Beta('BETA_WT', 0, None, None, 0)
BETA_AUX = Beta('BETA_AUX', 0, None, None, 0)
BETA_OV = Beta('BETA_OV', 0, None, None, 0)
BETA_NTRANS = Beta('BETA_NTRANS', 0, None,None, 0)
BETA_PS = Beta('BETA_PS', 0, None, None, 0)
BETA_TWAY = Beta('BETA_TWAY', 0, None, None, 0)
BETA_WK = Beta('BETA_WK', 0, None, None, 0)
BETA_NWK = Beta('BETA_NWK', 0, None, None, 0)
BETA_X1 = Beta('BETA_X1', 0, None, None, 0)
BETA_X2 = Beta('BETA_X2', 0, None, None, 0)
BETA_X3 = Beta('BETA_X3', 0, None, None, 0)
BETA_ASC = Beta('BETA_ASC', 0, None, None, 0)
BETA_S11 = Beta('BETA_S11', 0, None, None, 0)
BETA_S12 = Beta('BETA_S12', 0, None, None, 0)
BETA_S13 = Beta('BETA_S13', 0, None, None, 0)
BETA_S21 = Beta('BETA_S21', 0, None, None, 0)
BETA_S22 = Beta('BETA_S22', 0, None, None, 0)
BETA_S23 = Beta('BETA_S23', 0, None, None, 0)
BETA_E11 = Beta('BETA_E11', 0, None, None, 0)
BETA_E12 = Beta('BETA_E12', 0, None, None, 0)
BETA_E13 = Beta('BETA_E13', 0, None, None, 0)
BETA_E21 = Beta('BETA_E21', 0, None, None, 0)
BETA_E22 = Beta('BETA_E22', 0, None, None, 0)
BETA_E23 = Beta('BETA_E23', 0, None, None, 0)
BETA_O11 = Beta('BETA_O11', 0, None, None, 0)
BETA_O12 = Beta('BETA_O12', 0, None, None, 0)
BETA_O13 = Beta('BETA_O13', 0, None, None, 0)

s11=Variable('s11')
s12=Variable('s12')
s13=Variable('s13')
s21=Variable('s21')
s22=Variable('s22')
s23=Variable('s23')
e11=Variable('e11')
e12=Variable('e12')
e13=Variable('e13')
e21=Variable('e21')
e22=Variable('e22')
e23=Variable('e23')
o11=Variable('o11')
o12=Variable('o12')
o13=Variable('o13')

V1 = {}
V2 = {}
av = {}



for i in range(1, maxalt + 1):# Define variables for each alternative dynamically
    iv = Variable(f'iv_{i}')
    ov = Variable(f'ov_{i}')
    wt = Variable(f'wt_{i}')
    aux = Variable(f'aux_{i}')
    nTrans = Variable(f'nTrans_{i}')
    PS = Variable(f'PS_{i}')
    tway=  Variable(f'tway_{i}')
    wk=  Variable(f'wk_{i}')
    nwk=  Variable(f'nwk_{i}')
    x1= Variable(f'x1_{i}')
    x2= Variable(f'x2_{i}')
    x3= Variable(f'x3_{i}')
    # Define utility function for alternative i
    if dataUsed=='path':
        V1[i] = BETA_IV * iv + BETA_AUX * aux   + BETA_WT * wt + BETA_NTRANS * nTrans + BETA_PS * PS + BETA_TWAY * tway #full
        #V1[i] = BETA_IV * iv + BETA_WK * wk  + BETA_NWK * nwk  + BETA_WT * wt + BETA_NTRANS * nTrans + BETA_PS * PS + BETA_TWAY * tway #full
        V2[i] = BETA_IV * iv + BETA_OV * ov + BETA_NTRANS * nTrans + BETA_TWAY * tway # consolidate to ov
    else:
        V1[i] = BETA_X1*x1+BETA_X2*x2+BETA_X3*x3
        V2[i] = BETA_X1*x1+BETA_X2*x2+BETA_X3*x3
        if i==2:
            V1[i]+=BETA_ASC
            V2[i]+=BETA_ASC+BETA_S11*s11+BETA_S12*s12+BETA_S13*s13+BETA_S21*s21+BETA_S22*s22+BETA_S23*s23
            V2[i]+=BETA_E11*e11+BETA_E12*e12+BETA_E13*e13+BETA_E21*e21+BETA_E22*e22+BETA_E23*e23
            #V2[i]+=BETA_O11*o11+BETA_O12*o12+BETA_O13*o13
    # Define availability for alternative i (e.g., avail_1, avail_2, ..., avail_5)
    av[i] = Variable(f'avail_{i}')


choice = Variable('choice')


# Logit model: loglogit(V, av, chosen)
logprob1 = loglogit(V1, av, choice)
logprob2 = loglogit(V2, av, choice)
# Define the likelihood function
biogeme_model1 = bio.BIOGEME(database, logprob1)
biogeme_model2 = bio.BIOGEME(database, logprob2)

# Estimate the model
results = biogeme_model1.estimate()
print(results.get_estimated_parameters())

# Validation
new_database = Database('mymodel',dfTs)
biogeme_model1.database = new_database
betas = biogeme_model1.get_beta_values()
choice_probabilities = loglogit(V1, av, choice) 
biogeme_test = bio.BIOGEME(new_database, choice_probabilities)
simulated_LL = biogeme_test.simulate(betas)
simulated_LL['nalts']=dfTs.loc[:,dfTs.columns.str.contains('avail')].apply(sum,axis=1)
LLB=simulated_LL.log_like.sum()
LL0=sum(np.log(1/simulated_LL['nalts']))
rhosq=1-(LLB/LL0)
print(f' ******* Validation McFadden rho-sq value: {rhosq:.4f} *******')


# Estimate the model
results = biogeme_model2.estimate()
print(results.get_estimated_parameters())

# Validation
new_database = Database('mymodel',dfTs)
biogeme_model2.database = new_database
betas = biogeme_model2.get_beta_values()
choice_probabilities = loglogit(V2, av, choice) 
biogeme_test = bio.BIOGEME(new_database, choice_probabilities)
simulated_LL = biogeme_test.simulate(betas)
simulated_LL['nalts']=dfTs.loc[:,dfTs.columns.str.contains('avail')].apply(sum,axis=1)
LLB=simulated_LL.log_like.sum()
LL0=sum(np.log(1/simulated_LL['nalts']))
rhosq=1-(LLB/LL0)
print(f' ******* Validation McFadden rho-sq value: {rhosq:.4f} *******')





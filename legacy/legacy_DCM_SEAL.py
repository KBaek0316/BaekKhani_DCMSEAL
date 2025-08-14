# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:06:50 2024

@author: Kwangho Baek baek0040@umn.edu
"""
#%% Setup
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# directory setting
if os.environ['USERPROFILE']=='C:\\Users\\baek0040':
    WPATH=r'C:\Users\baek0040\Documents\GitHub\DCM_SEAL'
else:
    WPATH=os.path.abspath(r'C:\git\DCM_SEAL')
pd.set_option('future.no_silent_downcasting', True)
os.chdir(WPATH)
del WPATH

# initialize torch
np.random.seed(5723588)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Hyper-hyper parameters
dataUsed='synth' #'synth' or 'path'
from preProcessor import preProcess
dfPP=preProcess(dataOption=dataUsed,nobs=12500)
if dataUsed=='path': #available pathattrs: 'nwk','wk','wt','ntiv','tiv','nTrans' ,'aux', 'ov', 'iv','tt','PS'
    attrsUsed=['tway','ov','nTrans','iv'] #TRB
    #attrsUsed=['tway','PS','wk','nwk','wt','nTrans','iv']
else:
    attrsUsed=['x1','x2','x3']
#%% Before models
def varsPreprocessor(df, embcols=[],stdcols=[],segcols=[]):
    ''' 
    #path debugging
    segcols=['dayofweek','plan','worktype','stu','choicerider','purpose','age','income','hhsize','gender']
    embcols=['access','egress','typeO','typeD'] #'hr'
    stdcols=['realDep']
    df=dfPP.copy()
    #synth debugging
    segcols=['s1','s2']
    embcols=['e1','e2']
    stdcols=[]
    df=dfPP.copy()
    '''
    dfData=df.copy()
    keepcols=['id','match','alt','flag']+attrsUsed
    #input integrity check
    segcols=np.intersect1d(segcols,dfData.columns)
    embcols=np.intersect1d(embcols,dfData.columns)
    stdcols=np.intersect1d(stdcols,dfData.columns)
    unionLen=len(np.union1d(np.union1d(segcols,embcols),stdcols))
    emb_mappings = {}
    if unionLen<len(segcols)+len(embcols)+len(stdcols):
        raise Exception ('Columns should be mutually exclusive')
    dfS=dfData.loc[dfData.alt==0,:]
    if len(stdcols)>0: #standardization for stdcols, but after squeezing
        scaler=StandardScaler()
        stdized=pd.DataFrame(scaler.fit_transform(dfS[stdcols]),columns=stdcols)
        stdized['id']=dfS.id.values
        dfData=pd.merge(dfData.drop(columns=stdcols),stdized,on='id')
    if len(segcols)>0: #one-hot-encoding for segcols
        enc=OneHotEncoder(sparse_output=False)
        dfOnehot=pd.DataFrame(enc.fit_transform(dfData[segcols]),columns=enc.get_feature_names_out())
        dfData=pd.concat([dfData.drop(columns=segcols),dfOnehot],axis=1,ignore_index=False)
        segcolsfinal=enc.get_feature_names_out().tolist()+stdcols.tolist()
    else:
        segcolsfinal=[]
    if len(embcols)>0: #label-encoding for embcols
        for col in embcols:
            le = LabelEncoder() # Initialize LabelEncoder for each column
            dfData[col] = le.fit_transform(dfData[col])
            emb_mappings[col] = dict(enumerate(le.classes_))
    keepcols+=embcols.tolist()+segcolsfinal
    dfConverted=dfData.loc[:,keepcols]
    settings={}
    settings['nobs']=len(dfS)
    settings['nalts']=len(dfData.alt.unique())
    settings['numcols']=attrsUsed
    settings['embcols']=embcols.tolist()
    settings['segcols']=segcolsfinal
    settings['embmap']=emb_mappings
    return dfConverted, settings


def genTensors(dfConv,settings:dict,dimDown=False):
    '''
    settings=settings
    dimDown=False #deprecated
    '''
    maxalt=dfConv.alt.max() #settings['nalts'] may not be useful since it counts unique vals, not starting from 0
    #respondent-specific tensors : seg + emb
    dfR=dfConv.loc[dfConv.alt==0,['id','flag']+settings['segcols']+settings['embcols']].copy()
    ids=dfR.id.values
    tr=np.where(dfR.flag=='tr')[0]
    ts=np.where(dfR.flag=='ts')[0]
    dfR=dfR.drop(columns=['flag']).set_index('id')
    emb=torch.tensor(dfR.loc[:,settings['embcols']].to_numpy(),dtype=torch.long).to(device)
    seg=torch.tensor(dfR.loc[:,settings['segcols']].to_numpy(),dtype=torch.float32).to(device)
    #path numerical attributes; try to achieve diff-based MNL estimation by restructuring
    dfX=dfConv.loc[:,['id','alt','match']+settings['numcols']].copy()
    dfX2=dfX[dfX.alt==0].drop(columns=['alt','match'])
    dfXD=pd.merge(dfX, dfX2, on='id', suffixes=('_main', '_aux'))
    for attr in settings['numcols']:  #calc difference
        dfXD[attr] = dfXD[f'{attr}_main'] - dfXD[f'{attr}_aux']
    if dimDown: #deprecated; alt0 should be preprocessed as nonmatching path; we did in dfPath.sort_values(['id','match'])
        dfXD=dfXD.loc[dfXD.alt!=0,dfX.columns]
        maxalt=maxalt-1 #dim downed
    grouped=dfXD.groupby('id')
    numlist=grouped[settings['numcols']].apply(lambda x: x.values.tolist()).tolist()
    nums=torch.nn.utils.rnn.pad_sequence([torch.tensor(a,dtype=torch.float32) for a in numlist], batch_first=True, padding_value=0).to(device)
    choices = (grouped.apply(lambda x: np.argmax(x['match'].values),include_groups=False)).tolist() #pytorch accepts 0-starting indices
    choices=torch.tensor(choices, dtype=torch.long).to(device)
    validAlt = grouped['alt'].apply(lambda x: [1] * len(x) + [0] * (maxalt+1 - len(x)), include_groups=False).tolist()
    validAlt = torch.tensor(validAlt, dtype=torch.float32).to(device)
    return emb, seg, nums, choices, validAlt, tr, ts, ids


if dataUsed=='path':
    dfConv, settings = varsPreprocessor(dfPP,embcols=[],stdcols=['realDep'],segcols=dfPP.columns[(np.where(dfPP.columns=='summer')[0][0]):-1]) #TRB
    #dfConv, settings = varsPreprocessor(dfPP,embcols=['access','egress','typeO','typeD'],stdcols=['realDep'],segcols=['dayofweek','plan','worktype','stu','choicerider','age','income','hhsize','gender'])
    dfConv.loc[dfConv.match==1,:].drop(columns=['match','alt','flag','realDep'],errors='ignore').to_csv('dfInP.csv',index=False)
else:
    dfConv, settings = varsPreprocessor(dfPP,embcols=['e1','e2'],stdcols=[],segcols=['s1','s2'])
del dataUsed

emb, seg, nums, y, mask, tr, ts, ids=genTensors(dfConv,settings)
#%% Models
class MembershipModel(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_classes:int):
        super(MembershipModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layero = nn.Linear(hidden_size, num_classes)
    def forward(self, x): #input: segmentation bases (N*S), output: class membership probabilities (N*C)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layero(x), dim=1)
        return x


class EmbeddingMatrix(nn.Module): 
    def __init__(self, unique_cats_num: int, choices_num: int, homogeneous=False, dropout_rate=0.2):
        super(EmbeddingMatrix, self).__init__()
        self.choices_num = choices_num
        self.homogeneous = homogeneous
        if not self.homogeneous and choices_num<2:
            raise Exception('Make inputs either homogeneous=True or choices_num>1')
        embedding_dim = 1 if self.homogeneous else choices_num #K=1 for homogeneous in intermediate steps
        self.embedding = nn.Embedding(num_embeddings=unique_cats_num, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_rate) # may not need this if we directly enforce l2norm in the training, how about when normalized?
        if not homogeneous: # normalizing will cause undesired effect for homogeneous choices
            self.normalize_rows() # Normalize rows initially
    def normalize_rows(self): # Normalize each row of the emb matrix to have an L2 norm of 1
        with torch.no_grad():  # Disable gradient tracking for norm; good because interp- and identifiability between these and linear betas
            norm = self.embedding.weight.norm(p=2, dim=1, keepdim=True)
            self.embedding.weight.data = self.embedding.weight / norm  # Avoid in-place modification
    def forward(self,x): #input x: [batch_size, emb_vars_num] where values are indices of mapping emb_Vars and unique_cats
        emb_out = self.embedding(x)  # Resulting shape: [batch_size, emb_vars_num, choices_num or 1]
        if self.homogeneous: # Expand the dimension to choices_num when homogeneous
            emb_out = emb_out.repeat(1, 1, self.choices_num)  # Shape: [batch_size, emb_vars_num, choices_num]
        emb_out = self.dropout(emb_out) # Apply dropout after expansion
        emb_out = emb_out.permute(0, 2, 1)  # Reshape to [batch_size, choices_num, emb_vars_num]
        #self.normalize_rows() # Rather than this, let's call this periodically in training
        return emb_out


class SEAL_DCM(nn.Module):
    def __init__(self, settings: dict, nnodes: int, nClasses: int, homogeneous: bool = True, dropout: float=0.2, negBeta: int = 0, embByClass=True):
        super(SEAL_DCM, self).__init__()
        self.nClasses = nClasses
        self.embByClass = embByClass #looks like we should always do embedding by class?
        self.embcats = sum(len(v) for v in settings['embmap'].values())
        self.nAlts = settings['nalts']
        self.numSize = len(settings['numcols'])
        self.nEVars = len(settings['embcols'])
        self.intcpt = False
        self.latent_class_nn = MembershipModel(input_size=len(settings['segcols']), hidden_size=nnodes, num_classes=nClasses)
        if embByClass and self.embcats>0:
            self.emb_mats = nn.ModuleList([EmbeddingMatrix(unique_cats_num=self.embcats, choices_num=self.nAlts, homogeneous=homogeneous, dropout_rate = dropout) for _ in range(nClasses)])
        elif self.embcats>0:
            self.emb_mats = EmbeddingMatrix(unique_cats_num=self.embcats,choices_num=self.nAlts,homogeneous=homogeneous,dropout_rate=dropout)
        self.negBetas = negBeta + self.nEVars
        if not homogeneous:# Define intercepts with one per class set to zero
            self.intcpt = True
            self.beta = nn.Parameter(torch.randn(nClasses, self.nAlts - 1 + self.numSize + self.nEVars))
        else:
            self.beta = nn.Parameter(torch.randn(nClasses, self.numSize + self.nEVars))
    def forward(self, seg, nums, mask, emb): #note: expand() for the shared paremeter and repeat() for individual params estimated across an axis
        '''
        Dimensions (no intercept case; note that num_vars_size + emb_vars_num=num_features):
            inputs:
                seg (2D): [batch_size, seg_vars_num] -> latent_classes (2D): [batch_size, nClasses] -> [batch,1,nclass] after unsqueeze in final_prob
                nums (3D): [batch_size, choices_num, num_vars_size] -> num_expanded (4D): [batch_size, nClasses, choices_num, num_vars_size] via unsqueeze(1) and expand
                mask (2D): [batch_size, choices_num] -> mask.unsqueeze(1) (3D): [batch_size, 1, choices_num]
                emb (2D): [batch_size, emb_vars_num] -> embs (4D): [batch_size, nClasses, choices_num, emb_vars_num] after stacking along nClasses
            intermediates:
                covars (4D): [batch_size, nClasses, choices_num, num_features] after torch.cat([nums_expanded, embs], dim=3)->
                    final covars (3D) after view???: [batch_size * nClasses, choices_num, num_features]
                self.beta (2D): [nClasses, num_features] -> beta_expanded (3D): [batch_size, nClasses, num_features] by .unsqueeze(0).expand()->
                    negBetas separation/concat without dim changes-> final beta_expanded with reshape: [batch_size * nClasses, num_features, 1]
            output:
                logit: bmm([batch_size * nClasses, choices_num, num_features], [batch_size * nClasses, num_features, 1])->
                    [batch_size * nClasses, choices_num, 1]; see dims 1 and 2 for the both input and they are mat-mult-ed->
                    viewed to finally reshape the dimension by [batch_size, nClasses, choices_num], where alt_class_probs has the same dim
                mask.unsqueeze(1): [batch_size, 1, choices_num] will be broadcasted on dim 1 to mask associated unavailable alts for all classes
                final_prob: sum([batch_size, nClasses, 1] * [batch_size, nClasses, choices_num],dim=1)=[batch_size, choices_num]=wavg prob by nClasses
        '''
        batch_size = nums.size(0)
        latent_classes = self.latent_class_nn(seg)
        if self.embByClass and self.embcats>0: # Use individual embedding matrices for each class
            embs = [self.emb_mats[c](emb) for c in range(len(self.emb_mats))]  # List of [batch_size, choices_num, emb_vars_num]
            embs = torch.stack(embs, dim=1)  # Shape: [batch_size, nClasses, choices_num, emb_vars_num]
        elif self.embcats>0:  # Use a single embedding matrix and expand along the nClasses dimension
            embs = self.emb_mats(emb)  # Shape: [batch_size, choices_num, emb_vars_num]
            embs = embs.unsqueeze(1).expand(-1, self.nClasses, -1, -1)  # Shape: [batch_size, nClasses, choices_num, emb_vars_num]
        if self.intcpt:# Add intercept if specified
            identity_matrix = torch.eye(self.nAlts).to(device).unsqueeze(0).expand(batch_size, -1, -1) # Shape: [batch_size, choices_num, choices_num]
            nums = torch.cat([identity_matrix, nums], dim=2)  # Shape: [batch_size, choices_num, choices_num+numSize]
            intercepts = torch.cat([torch.zeros(self.nClasses, 1).to(device), self.beta[:, :self.nAlts - 1]], dim=1) #add uninitialized reference ASC (0): 
            beta_expanded = torch.cat([intercepts, self.beta[:,self.nAlts - 1:]], dim=1).unsqueeze(0).expand(batch_size, -1, -1) #[batch_size, nClases, choices_num+num_features]
        else:
            beta_expanded = self.beta.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [*batch_size*, nClasses, num_features]
        # Expand nums to include the nClasses dimension and make covars
        nums_expanded = nums.unsqueeze(1).expand(-1, self.nClasses, -1, -1)  # Shape: [batch_size,*nClasses*, choices_num, (choices_num+) numSize]
        if self.embcats>0:
            covars = torch.cat([nums_expanded, embs], dim=3)  # Shape: [batch_size, nClasses, choices_num, (choices_num+) num_features]
        else:
            covars = nums_expanded
        if self.negBetas > 0: # Apply constraints on last negBetas amount of betas; beta_expand dimension is unchanged after this clause
            beta_free = beta_expanded[..., : -self.negBetas]
            beta_const = beta_expanded[..., -self.negBetas:]
            beta_const = -F.relu(-1 * beta_const)
            beta_expanded = torch.cat([beta_free, beta_const], dim=-1)
        beta_expanded = beta_expanded.unsqueeze(-1) # Reshape beta for matmul to [batch_size, nClasses, (choices_num+) num_features, 1]
        logits = torch.matmul(covars, beta_expanded).squeeze(-1)  # matmul: [batch_size, nClasses, choices_num,1]-> remove last singleton
        #print("Logits (before masking):", logits[0])
        logits_masked = logits.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))  # Normalize probabilities to -float('inf')?
        #print("Logits (after masking):", logits_masked[0])
        alt_class_probs = torch.softmax(logits_masked, dim=2) #does not change dimension
        #print("Alternative Class Probabilities:", alt_class_probs[0])
        # Aggregate class probabilities for the final output probability: sum([batch_size, nClasses, 1] * [batch_size, nClasses, choices_num],dim=1)
        final_prob = torch.sum(latent_classes.unsqueeze(2) * alt_class_probs, dim=1)  # Sum along the class dimension
        final_prob = torch.clamp(final_prob, 1e-7, 1 - 1e-7)  # Avoid log(0)
        return final_prob, latent_classes


# Training function
def train_model(seg=seg,nums=nums,emb=emb,y=y,mask=mask,tr=tr,ts=ts,settings=settings,testVal=True,nclass=2,homogeneous=True,
                embByClass=True,negBetaNum=2,max_norm=5,nnodes=64,nepoch=300,lrate=0.03,GammaMem=0.02,GammaEmb=0.05,dropout=0.1):
    '''
    testVal=True
    nclass=3
    homogeneous=False
    embByClass=True
    negBetaNum=0
    max_norm=5
    nnodes=32
    nepoch=300
    lrate=0.03
    GammaMem=0.005
    GammaEmb=0.002
    dropout=0.1
    normFreq=20 #deprecated
    '''
    #Setup
    if testVal:
        seg_ts, nums_ts, emb_ts, y_ts, mask_ts = seg[ts], nums[ts], emb[ts], y[ts], mask[ts]
        ts_losses = []
    else:
        tr=range(len(y))
    seg_tr, nums_tr, emb_tr, y_tr, mask_tr = seg[tr], nums[tr], emb[tr], y[tr], mask[tr]
    tr_losses = []
    model = SEAL_DCM(settings,nnodes,nclass,homogeneous=homogeneous,dropout=dropout,negBeta=negBetaNum,embByClass=embByClass).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    embedded= (emb.size()[1]>0) #boolean
    # Training loop
    for epoch in range(nepoch):  # number of epochs
        model.train()
        #Forward
        y_hat, latent_classes = model(seg_tr, nums_tr, mask_tr, emb_tr) #dims: [batch_size, choices_num], [batch_size, nClasses]
        #CELoss with Masking
        y_hat_ml = torch.log(y_hat)  # Apply mask to ignore invalid (mask==0) alternatives and avoid log(0)
        #y_hat_ml = torch.log(y_hat * mask_tr + 1e-7)  # Apply mask to ignore invalid (mask==0) alternatives and avoid log(0)
        # Gather the predicted log-probabilities for the chosen alternatives
        chosen_logprobs = torch.gather(y_hat_ml, 1, y_tr.unsqueeze(1)).squeeze(1)  # log(y_n^i_hat)s only for chosen alternatives indexed by y
        loss_raw = -chosen_logprobs.mean() #: -(1/N)sum(i){sum(j){y_n^i*log(y_n^i_hat)}}; sum(i) redundant because of the above line
        # L2 regularization for membership NN and embedding
        l2_nn = sum(torch.norm(param) for name, param in model.named_parameters() if 'latent_class_nn' in name)
        if embedded:
            if embByClass:
                l2_emb = sum(torch.norm(embedding_matrix.embedding.weight) for embedding_matrix in model.emb_mats)
            else:
                l2_emb = torch.norm(model.emb_mats.embedding.weight)
            if (not homogeneous): # and ((epoch+1) % normFreq == 0) (deprecated):
                if embByClass: #list of nClass embedding matrices
                    for embedding_matrix in model.emb_mats:
                        embedding_matrix.normalize_rows()
                else: #single embedding matrix
                    model.emb_mats.normalize_rows()
        else:
            l2_emb=0
        loss = loss_raw + GammaMem * l2_nn + GammaEmb * l2_emb
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if max_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm) #prevent gradient explosion
        optimizer.step()
        #Bookkeeping
        tr_losses.append(loss.item())  # Store the loss value
        if testVal:
            model.eval()
            with torch.no_grad():
                y_hat_ts, _ = model(seg_ts, nums_ts, mask_ts,emb_ts)
                y_hat_ml_ts = torch.log(y_hat_ts * mask_ts + 1e-7)
                chosen_logprobs_ts = torch.gather(y_hat_ml_ts, 1, y_ts.unsqueeze(1)).squeeze(1)
                ts_loss = -chosen_logprobs_ts.mean().item()
                ts_losses.append(ts_loss)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{nepoch}], Loss: {loss.item():.4f}')
    # Summary; torch.sum(torch.log(1/(1+mask_tr.sum(dim=1)))).item() when diff used for LL0
    LL0=torch.sum(torch.log(1/mask_tr.sum(dim=1))).item()
    LLB=torch.sum(chosen_logprobs).item()
    rhosq=1-(LLB/LL0)
    print(f' ** Training McFadden rho-sq value: {rhosq:.4f} **')
    if testVal:
        LL0_ts=torch.sum(torch.log(1/mask_ts.sum(dim=1))).item()
        LLB_ts=torch.sum(chosen_logprobs_ts).item()
        rhosq_ts=1-(LLB_ts/LL0_ts)
        print(f' ** Test or validation McFadden rhosq: {rhosq_ts:.4f} **')
        plt.plot(range(1, len(ts_losses) + 1), ts_losses, label='Testing Loss')
    else:
        rhosq_ts=np.nan
    plt.plot(range(1, len(tr_losses) + 1), tr_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses (nnodes: {nnodes} lrate: {lrate:.3f})')
    plt.legend()
    plt.show()
    if homogeneous:
        rescols=settings['numcols']+settings['embcols']
    else:
        rescols=['ASC'+str(i+2) for i in range(settings['nalts']-1)]+settings['numcols']+settings['embcols']
    resdf=pd.DataFrame(model.beta.cpu().detach().numpy(),columns=rescols)
    resdf.index=np.array('Class')+(resdf.index+1).astype(str)
    
    membinsp=pd.DataFrame(latent_classes.detach().cpu().numpy())
    membershipTest=membinsp.apply(min).max()<1/2 #encompasses std()->0 and abs(mean()-0.5) ->0.5
    memPropRange=membinsp.loc[:,0].max()-membinsp.loc[:,0].min()
    memPropRange=membinsp.loc[:,0].max()-membinsp.loc[:,0].min()
    print(f' ** Membership test result (False: fail): {membershipTest} **')
    print(f' ** Class 1 Assignment Probabilities Range: {memPropRange:.4f} **')
    
    #attaching embedded terms and calculating MRS
    if embedded:
        if embByClass:
            resultDict = {f'emb_{i+1}': model.emb_mats[i].embedding.weight.detach().cpu().numpy() for i in range(len(model.emb_mats))}
        else:
            resultDict ={'emb':model.emb_mats.embedding.weight.detach().cpu().numpy()}
        embRepeat=np.array([len(values) for values in settings['embmap'].values()])
        for i, embmat in enumerate(resultDict.values()):
            embEstimates=resdf.iloc[i,-model.nEVars:].values
            embEstimates=np.repeat(embEstimates,embRepeat) #expand by num of levels
            embFinal=embmat*embEstimates[:,np.newaxis]
            if i==0:#currently only works for binary alternative choice
                emblevels=[level for embvar in settings['embmap'].values() for level in embvar.values()]
                embAddition=pd.DataFrame([list((embFinal[:,1]-embFinal[:,0]))],columns=emblevels)
            else:
                embAddition.loc[i,:]=(embFinal[:,1]-embFinal[:,0])
        if not embByClass:
            embAddition=pd.concat([embAddition]*model.nClasses,ignore_index=True)
        embAddition.index=resdf.index
        MRS=pd.concat([resdf.iloc[:,:-1*model.nEVars],embAddition],axis=1)
    else:
        MRS=resdf.copy()
        resultDict={}
    MRSvar=MRS.columns[MRS.columns.isin(['x1','iv'])][0]
    MRS=MRS.div(resdf[MRSvar],axis=0)
    print(' ** MRS with respect to in-vehicle time (IVT) **')
    print(round(MRS.iloc[:,:10],2))

    resultDict.update({'membershipTest':membershipTest,'memPropRange':memPropRange,'rho_tr':rhosq,'rho_ts':rhosq_ts,'rescols':rescols})
    return model, resdf, MRS, resultDict, membinsp


''' Tuning ID 214
out1, out2, out3, out4, out5 = train_model(
    nclass=3,homogeneous=False,embByClass=True,negBetaNum=0,max_norm=7,
    nnodes=64,nepoch=500,lrate=0.03,GammaMem=0.003,GammaEmb=0.001,dropout=0)
'''

def desiredModel(resultDict,MRS,maxMRS=30):
    #resultDict=out4.copy()
    #MRS=out3.copy()
    firstround=(resultDict['rho_tr']>0.3) and ((resultDict['rho_ts']>0.15))# and resultDict['membershipTest']
    secondround=((abs(MRS.values)>maxMRS).sum()==0)
    isDesired=firstround and secondround
    return isDesired

#%% Hyperparameter tuning
def mTuning(filename):
    '''
    filename='tuning.csv'
    '''
    dfTune=pd.read_csv(filename)
    for row in dfTune.itertuples():
        if not np.isnan(row.rho5):
            continue
        print(row)
        i=0
        rhos=[]
        rhotss=[]
        rhodesired=[]
        membership=0
        desired=0
        while i<row.niter:
            i+=1 
            modelout, betas, MRSs, results, membs = train_model(
                nclass=3,homogeneous=False,embByClass=True,negBetaNum=0,max_norm=row.maxnorm,nnodes=row.nnodes,
                nepoch=row.nepoch,lrate=row.lrate,GammaMem=row.GammaMem,GammaEmb=row.GammaEmb,dropout=row.dropout)
            rhos.append(results['rho_tr'])
            rhotss.append(results['rho_ts'])
            if results['membershipTest']:
                membership+=1
                if desiredModel(results,MRSs,maxMRS=20):
                    desired+=1
                    rhodesired.append(results['rho_ts'])
            print(str(i))
        rhos=np.array(rhos)
        rhotss=np.array(rhotss)
        dfTune.loc[row.Index,'rho5']=sum(rhos>0.5)
        dfTune.loc[row.Index,'rhomax']=rhos.max()
        dfTune.loc[row.Index,'rhomean']=rhos[rhos>0].mean()
        dfTune.loc[row.Index,'membership']=membership
        dfTune.loc[row.Index,'desired']=desired
        dfTune.loc[row.Index,'successprop']=desired/row.niter
        if desired>0:
            dfTune.loc[row.Index,'testrhodes']=np.array(rhodesired).mean()
        dfTune.to_csv('tuning.csv',index=False)
    return None

#mTuning('tuning.csv')



#%% Getting Results
def calculate_mrae(row_true, row_estimated):
    """Calculate Mean Relative Absolute Error (MRAE) between two rows."""
    return np.mean(np.abs(row_true - row_estimated) / np.abs(row_true))

def match_rows(trueVal, estimatedMat):
    """Match rows of estimatedMat to trueVal based on MRAE."""
    #trueVal=trueMRS.copy()
    #estimatedMat=MRSs.copy()
    trueVal=trueVal.copy().reset_index(drop=True)
    estimatedMat=estimatedMat.copy().reset_index(drop=True)
    mrae_matrix = np.zeros((len(trueVal), len(estimatedMat)))    # Calculate the MRAE matrix
    for i, true_row in trueVal.iterrows():
        for j, estimated_row in estimatedMat.iterrows():
            mrae_matrix[i, j] = calculate_mrae(true_row.values, estimated_row.values)
    row_indices, col_indices = linear_sum_assignment(mrae_matrix)    # Determine the optimal row matching outputs 2
    estimatedMatModified = estimatedMat.iloc[col_indices].reset_index(drop=True)    # Reorder estimatedMat
    return estimatedMatModified,col_indices


def outputResults(MRScols=['ASC2','x1','x2','x3'],numOut=30,maxMRS=20,outName='modelout.csv'):
    trueMRS=pd.read_csv('trueParams.csv',index_col='ind')
    trueMRS=trueMRS[MRScols]
    desired=0
    i=1
    while desired<numOut: #tuning # 214
        modelout, betas, MRSs, results, membs = train_model(
            nclass=3,homogeneous=False,embByClass=True,negBetaNum=0,max_norm=7,
            nnodes=64,nepoch=500,lrate=0.03,GammaMem=0.003,GammaEmb=0.001,dropout=0)
        if desiredModel(results,MRSs,maxMRS=maxMRS):
            modelout.eval()
            with torch.no_grad():
                _, member_prop = modelout(seg, nums, mask,emb)
            member_prop=pd.DataFrame(member_prop.detach().cpu().numpy().astype(float))
            member_prop['assigned']=member_prop.idxmax(axis=1)+1
            member_prop.columns=np.append(np.char.add('class',((np.arange(modelout.nClasses)+1).astype(str))),'assigned')
            assignedmean=member_prop.assigned.mean()
            if modelout.nClasses==1 or (assignedmean>1 and assignedmean<2):
                desired+=1
                print(f'!!!!!!!!!!!!!!!{desired} Desired models found!!!!!!!!!!!!!!!')
                '''
                if MRSs.loc['Class1','tway']>MRSs.loc['Class2','tway']: # make class 1 as transitway likely class; lower better
                    betas.iloc[[0,1],:]=betas.iloc[[1,0],:]
                    MRSs.iloc[[0,1],:]=MRSs.iloc[[1,0],:]
                    member_prop['assigned']=(3-member_prop.assigned) #invert 1 and 2
                    member_prop.iloc[:,[0,1]]=member_prop.iloc[:,[1,0]] #swap cols
                '''
                if modelout.nClasses>1:
                    MRS2=MRSs[['ASC2','x1','x2','x3']].copy()
                    reordered, ordind=match_rows(trueMRS, MRS2)
                    MRS2=pd.DataFrame(reordered.values,index=MRS2.index,columns=MRS2.columns)
                    MRSs=MRSs.drop(columns=['ASC2','x1','x2','x3']).iloc[ordind,:]
                    MRSs=pd.concat([MRS2,MRSs],axis=1)
                betanames=np.array([f'{x}_{y}' for y in [str(i) for i in range(1, modelout.nClasses + 1)] for x in MRSs.columns])
                storeit=pd.Series(np.append(np.array([results['rho_tr'],results['rho_ts']]),np.array(MRSs).flatten()),
                                  index=np.append(np.array(['rho_tr','rho_ts']),betanames.flatten()))
                if desired==1:
                    dataOut=pd.DataFrame({('Model'+str(i)):storeit})
                else:
                    dataOut[('Model'+str(i))]=storeit
                    dataOut=dataOut.copy()
                dataOut.to_csv(outName)
        i+=1
    return dataOut

dfOut=outputResults(MRScols=['ASC2','x1','x2','x3'],numOut=100,maxMRS=20,outName='modelSEAL.csv')

#%% deprecated?
def desiredModel2(betas,rsq,rhots=None,strict=False,interDiff=0.5/2.5,betaNonSig=0.05/2.5,rsqCut=0.4): #sqrt(10-1)=3 maxElemExclAdj->2.5
    from numpy import linalg as LA
    adjL2norm=LA.norm(betas.flatten()[betas.flatten()<betas.max()]) #delete largest elem, then L2
    intercepttest=((betas[:,0].prod()<0) or (abs(betas[0,0]-betas[1,0])>interDiff*adjL2norm))
    betatest=all(betas[:,1:].flatten()<betaNonSig*adjL2norm)
    if strict: #when nonpositive constraints are applied to ivt and ntrans
        #intercepttest=(betas[:,0].prod()<0)
        betatest=sum(betas[:,-2:].flatten()<0)>(len(betas[:,-2:].flatten())-2) #allow one positive beta
    rhotest=rsq>rsqCut
    if rhots is not None:
        rhotest= rhotest*(rhots*3>rsq)
    testResult=intercepttest*rhotest*betatest
    print(f'intercept: {intercepttest}, coeffs: {betatest}, rho: {rhotest}')
    return testResult
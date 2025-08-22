library(dplyr)
library(forcats)
library(ggplot2)
library(tidyr)
library(viridis)
library(scales)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

df<-read.csv("result1.csv")



dftemp<-df %>% filter(Rank<180 & mode=="DCMSEAL")
dftemp$top10<-'No'
dftemp[dftemp$Rank<20,'top10']<-'Yes'
dftemp$nhidden<-as.factor(dftemp$params_num_hidden_layers)

p<-ggplot(dftemp,aes(x=params_updates_per_epoch,y=value,shape=nhidden,color=top10))+
  xlab("Minibatch updates per epoch")+ylab("Validation accuracy")+theme_bw()+ylim(0.275,0.285)+
  scale_x_log10()+
  geom_point(size=2)+facet_wrap(~factor(params_embedding_mode),ncol = 2)+
  theme(legend.title       = element_text(face = "bold",size = rel(0.8)),
        legend.text        = element_text(size = rel(0.8)),     
        legend.key.size    = unit(0.8, "lines"),               
        legend.key.height  = unit(0.8, "lines"),              
        legend.key.width   = unit(0.8, "lines"),             
        legend.spacing     = unit(0.4, "lines"),           
        legend.margin      = margin(2, 2, 2, 2),
        legend.position = c(0.55, 0.98),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = scales::alpha("white", 0.7), ,color='black'),
        strip.text.x = element_text(face = "bold",size = 10))
p


dftemp<-df %>% filter(Rank<100)
dftemp$segmetation_dropout<-dftemp$params_segmentation_dropout_rate

p<-ggplot(dftemp,aes(x=params_weight_decay_embedding,y=params_weight_decay_segmentation,shape=mode,color=value))+
  scale_color_viridis_c()+scale_x_log10()+scale_y_log10()+geom_point(size=2)+
  theme(legend.position = 'top',legend.justification = c(0, 1))
p

p<-ggplot(dftemp,aes(x=params_weight_decay_embedding,y=segmetation_dropout,shape=mode,color=value))+
  scale_color_viridis_c()+scale_x_log10()+geom_point(size=2)+
  theme(legend.position = 'top',legend.justification = c(0, 1))
p

p<-ggplot(dftemp,aes(x=params_weight_decay_segmentation,y=segmetation_dropout,shape=mode,color=value))+
  scale_color_viridis_c()+scale_x_log10()+geom_point(size=2)+
  theme(legend.position = 'top',legend.justification = c(0, 1))
p


p<-ggplot(dftemp,aes(y=model,x=meantop5,fill=loc))+ylab("Machine Learning Model")+xlab("Top 5 mean validation accuracy boxplot from each study")+
  xlim(lim1,lim2)+scale_fill_viridis(discrete=TRUE)+scale_color_viridis(discrete=TRUE)+theme_bw()+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_boxplot(alpha=0.7);p #+geom_dotplot(binaxis = "y", stackdir = "center", dotsize = 0.3,aes(color=cat,shape=cat))
ggsave("Fig8.jpg",width=7.5,height=4,units="in",dpi=300)

# Figure 9 delta vs d_out
dftemp<-df %>% filter(!is.na(encDim)) %>% group_by(denom,encDim,model) %>% summarize(meantop5=mean(meantop5))
dftemp2<-df %>% filter(!is.na(encDim)) %>% group_by(denom,encDim) %>% summarize(meantop5=mean(meantop5)) %>% mutate(model="Aggregated\nMean")
dftemp<-rbind(dftemp[,c('denom','encDim','meantop5','model')],dftemp2)
dftemp$model<-factor(dftemp$model,c(levels(df$model),"Aggregated\nMean"))
x_vals<-unique(dftemp$encDim)
y_vals<-unique(dftemp$denom)
level_labels <- with(expand.grid(x = x_vals, y = paste0(y_vals/1000, "K"), KEEP.OUT.ATTRS = FALSE),sprintf("(%d, %s)", x, y))
dftemp<-dftemp %>% mutate(combo = sprintf("(%d, %s)",encDim,paste0(denom/1000, "K")),
  combo = factor(combo, levels = level_labels))
rm(x_vals,y_vals,level_labels,dftemp2)
dftemp$denom<-factor(dftemp$denom)
dftemp$encDim<-factor(dftemp$encDim)
dftemp$meantop5<-round(dftemp$meantop5,3)
dftemp<-dftemp %>% filter(model!='Aggregated\nMean')

p<-ggplot(dftemp,aes(x=meantop5,y=combo))+facet_wrap(~factor(model),nrow = 2)+
  geom_segment(aes(yend=combo),xend=0)+geom_text(aes(label=sprintf("%.3f", meantop5)),hjust=-.2)+
  geom_point(aes(shape=encDim,color=denom),size=3)+theme_bw()+xlim(0.599,0.661)+
  scale_color_viridis(discrete=TRUE)+ylab("(d_out, delta)")+xlab("Average of top 5 validation accuracy")+
  labs(shape = "Output Dimension (d_out)",
       color = "Base (delta)") +
  guides(shape = guide_legend(order = 1),
         color = guide_legend(order = 2)) +
  theme(legend.position = c(0.96,0), legend.justification = c(1,0),
        legend.background = element_rect(fill=NA,color='black'),
        legend.title = element_text(face = "bold",size = 9),
        legend.text        = element_text(size = rel(0.8)),     
        legend.key.size    = unit(0.8, "lines"),               
        legend.key.height  = unit(0.8, "lines"),              
        legend.key.width   = unit(0.8, "lines"),             
        legend.spacing     = unit(0.4, "lines"),           
        legend.margin      = margin(2, 2, 2, 2),
        strip.text= element_text(face = "bold",size=9),
        strip.text.x = element_text(size = 10))
p
ggsave("Fig9.jpg",width=7.5,height=4,units="in",dpi=300)


#Table 5
df1<-df %>% group_by(model) %>% arrange(meantop5) %>% filter(meantop5==max(meantop5)) %>% select(model,meantop5,opt,locOpt,encDim,denom,timeOpt,catOpt) %>% mutate(res='best')
df2<-df %>% group_by(model) %>% arrange(meantop5) %>% filter(meantop5==min(meantop5)) %>% select(model,meantop5,opt,locOpt,encDim,denom,timeOpt,catOpt) %>% mutate(res='worst')
rbind(df1,df2) %>% arrange(model,res)

rm(list=ls()[ls()!="fpath"])


# Stage 2: Best Settings -----
rescols<-c('Home_precision','Home_recall','Home_f1.score','Work_precision','Work_recall','Work_f1.score','accuracy','zeroPreds','value')
l<-list.files(path=fpath,pattern="Final.*.csv")
for (i in 1:length(l)){
  dftemp<-read.csv(paste0(fpath,l[i]),1)
  dftemp<-dftemp %>% select(all_of(c("model","option",rescols))) %>% arrange(value) %>% slice(-(c(1:5, (n() - 4):n())))
  if (i==1){
    df<-dftemp
  } else{
    df<-rbind(df,dftemp)
  }
}
rm(dftemp,i,l)
unique(df$model)
df$model<-factor(factor(df$model,levels=unique(df$model),
                        labels=c("CatBoost","Neural\nNetworks","Random\nForest","Support Vector\nMachine","XGBoost"))
                 ,levels=c("Random\nForest","XGBoost","CatBoost","Neural\nNetworks","Support Vector\nMachine"))

summary(df$value)
#Fig 10
df$pipe<-"Best pipeline"
df[df$option=='worst','pipe']<-"Worst pipeline"
p<-ggplot(df,aes(y=model,x=value,fill=fct_rev(pipe)))+theme_bw()+geom_boxplot(alpha=0.75)+xlim(0.45,0.68)+
  labs(fill="Feature engineering\ntechniques applied")+
  scale_fill_viridis(discrete=TRUE)+guides(fill = guide_legend(reverse = TRUE))+
  ylab("Machine Learning Models")+xlab("Validation accuracy distribution (intermediate 100 per a boxplot)")+
  theme(legend.position = c(0.02, 0.02), legend.justification = c(0, 0),
        legend.background = element_rect(fill=NA,color='black'),
        legend.spacing = unit(0.5, "cm"),
        legend.title = element_text(face = "bold", size = 10))
p
ggsave("Fig10.jpg",width=5.4,height=3.6,units="in",dpi=300)


#Fig 12
l<-list.files(path=fpath,pattern=("(Final.*best.csv)|(CPU.*.csv)"))
for (i in 1:length(l)){
  dftemp<-read.csv(paste0(fpath,l[i]),1)
  dftemp<-dftemp %>% select(model,duration) %>% arrange(duration) %>% slice(-(c(1:5, (n() - 4):n())))
  dftemp$Hardware<-"GPU"
  if (grepl('CPU',l[i])==1){dftemp$Hardware<-"CPU"}
  if (i==1){
    dftime<-dftemp
  } else{
    dftime<-rbind(dftime,dftemp)
  }
}
rm(dftemp,i,l)
unique(dftime$model)
dftime$model<-factor(factor(dftime$model,levels=unique(dftime$model),
                        labels=c("CatBoost","Neural\nNetworks","Random\nForest","XGBoost","Support Vector\nMachine"))
                 ,levels=c("Random\nForest","XGBoost","CatBoost","Neural\nNetworks","Support Vector\nMachine"))

medians <- dftime %>% group_by(model,Hardware) %>% summarize(avg=mean(duration))
medians
p<-ggplot(dftime,aes(x=model,y=duration,fill=Hardware))+theme_bw()+
  geom_hline(yintercept = c(5,60, 600),color = "grey20",linetype = "dashed",linewidth = 0.4,alpha = 0.9)+geom_boxplot(alpha=0.75)+
  scale_fill_viridis(discrete=TRUE)+xlab("Machine Learning Models")+ylab("Model fitting time in seconds (logscale)")+
  annotation_logticks(sides = "l") +scale_y_log10(breaks=breaks_log(),minor_breaks=minor_breaks_log())+
  theme(legend.position = c(0.05, 0.95), legend.justification = c(0, 1),
        legend.background = element_rect(fill = scales::alpha("white", 0.7), ,color='black'),
        legend.spacing = unit(0.5, "cm"),
        legend.title = element_text(face = "bold", size = 10))
p
ggsave("Fig12.jpg",width=5.4,height=3.6,units="in",dpi=300)


# Table 6

rescols<-c('Home_precision','Home_recall','Home_f1.score','Work_precision','Work_recall','Work_f1.score','value')
l<-list.files(path=fpath,pattern="Final.*best.csv")
for (i in 1:length(l)){
  dftemp<-read.csv(paste0(fpath,l[i]),1)
  dftemp<-dftemp %>% select(all_of(c("model",rescols,'duration')))
  if (i==1){
    df<-dftemp
  } else{
    df<-rbind(df,dftemp)
  }
}
rm(dftemp,i,l)
unique(df$model)
df$model<-factor(factor(df$model,levels=unique(df$model),
                        labels=c("CatBoost","Neural\nNetworks","Random\nForest","Support Vector\nMachine","XGBoost"))
                 ,levels=c("Random\nForest","XGBoost","CatBoost","Neural\nNetworks","Support Vector\nMachine"))
df2<-df %>% group_by(model) %>% filter(value==max(value)) %>% ungroup() %>% arrange(model)
df2
write.table(df2,"clipboard",sep="\t",row.names=FALSE)

#Fig17 SVM Kernel 2D Scatter x:C y:accuracy shape:kernel facet:weight
# unique(df$params_kernel)="rbf"    "poly"   "linear"
rm(list=ls()[ls()!="fpath"])
dftemp<-read.csv(paste0(fpath,"Final_SVbest.csv"),1)
dftemp$Kernel<-"Radial basis"
dftemp[dftemp$params_kernel=="linear","Kernel"]<-"Linear"
dftemp[!is.na(dftemp$params_degree) & dftemp$params_degree == 2,"Kernel"]<-"Polynomial\n(dgree 2)"
dftemp[!is.na(dftemp$params_degree) & dftemp$params_degree == 3,"Kernel"]<-"Polynomial\n(dgree 3)"
dftemp[!is.na(dftemp$params_degree) & dftemp$params_degree == 4,"Kernel"]<-"Polynomial\n(dgree 4)"
dftemp$Kernel<-factor(dftemp$Kernel)
dftemp$weight<-"Label Weights: Not Weighted"
dftemp[dftemp$params_class_weight=="balanced",'weight']<-"Label Weights: Weighted"
dftemp$weight<-factor(dftemp$weight,levels=c("Label Weights: Weighted","Label Weights: Not Weighted"))

p<-ggplot(dftemp,aes(x=params_C,y=adj_accuracy,shape=Kernel,color=Kernel))+
  xlab("C or 1/lambda (sampled from a log-scale space)")+ylab("Validation Accuracy")+theme_bw()+ylim(0.349,0.651)+
  scale_x_log10(breaks=breaks_log())+
  scale_shape_manual(values = c(8,25,24,22,16)) +
  geom_point(size=2)+facet_wrap(~factor(weight),ncol = 2)+
  theme(legend.title       = element_text(face = "bold",size = rel(0.8)),
        legend.text        = element_text(size = rel(0.8)),     
        legend.key.size    = unit(0.8, "lines"),               
        legend.key.height  = unit(0.8, "lines"),              
        legend.key.width   = unit(0.8, "lines"),             
        legend.spacing     = unit(0.4, "lines"),           
        legend.margin      = margin(2, 2, 2, 2),
        legend.position = c(0.98, 0.02),
        legend.justification = c(1, 0),
        legend.background = element_rect(fill = scales::alpha("white", 0.7), ,color='black'),
        strip.text.x = element_text(face = "bold",size = 10))
p
ggsave("Fig17.jpg",width=6,height=3,units="in",dpi=300)



# prelim (deprecated) -----------

df$obj<-df$meantop5
p<-ggplot(df,aes(x=opt,y=obj,color=modelOpt))+xlab("Cat Vars Consolidation Method")+ylab("Objective Val")+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_point();p

p<-ggplot(df,aes(x=locOpt,y=obj,color=modelOpt))+xlab("Space Input")+ylab("Objective Val")+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_point();p

p<-ggplot(df,aes(y=modelOpt,x=obj,fill=timeOpt))+xlab("Space Input")+ylab("Objective Val")+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_boxplot();p

p<-ggplot(df,aes(x=timeOpt,y=obj,color=modelOpt))+xlab("Time Input")+ylab("Objective Val")+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_point();p

dfenc<-df %>% filter(!is.na(denom))

p<-ggplot(dfenc,aes(x=encDim,y=obj,color=modelOpt))+xlab("2DPE Output Dimension")+ylab("Objective Val")+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_point();p

p<-ggplot(dfenc,aes(x=denom,y=obj,color=modelOpt))+xlab("2DPE Denominator")+ylab("Objective Val")+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_point();p


df %>% group_by(modelOpt) %>% summarize(mean=mean(obj),sd=sd(obj))
df %>% group_by(opt) %>% summarize(mean=mean(obj),sd=sd(obj))
df %>% group_by(opt,modelOpt) %>% summarize(mean=mean(obj),sd=sd(obj))


dfcat<-df
dfcat<-df %>% filter(opt==0)
dfcat<-df %>% filter(opt==0 & locOpt=='zone')

p<-ggplot(dfcat,aes(x=catOpt,y=obj,color=modelOpt))+xlab("Feature Encoding")+ylab("Objective Val")+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_point();p

p<-ggplot(df,aes(x=catOpt,y=obj,fill=modelOpt))+xlab("Space Input")+ylab("Objective Val")+
  theme(legend.position = "top", legend.justification = c(0.5, 0.5),
        legend.background = element_blank(),legend.key = element_blank(),
        legend.margin = margin(),legend.spacing = unit(0.5, "cm"),legend.title=element_blank())+
  geom_boxplot();p

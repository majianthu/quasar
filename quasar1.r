library(copent)
library(astrodatR)
library(e1071)
library(randomForest)
library(lattice)

data("SDSS_QSO")

# plot data
x11(); 
hist(SDSS_QSO$z, 1000, xlab = "z", main = "Distribution of quasar redshifts")

# a subset used, adding noise to avoid ties
n1 = 5000 # #samples for estimating feature importance
data1 = SDSS_QSO[1:n1,c(2:12,15)]
for(k in 1:12){
  data1[,k] = data1[,k] + max(data1[,k]) * runif(n1,max = 0.0000001)
}

# correlation between bands and interpretation
corrmat1 = matrix(0,12,12)
for(i in 1:11){
  for(j in (i+1):12){
    corrmat1[i,j] = copent(data1[,c(i,j)])
  }
}
colnames(corrmat1) = rownames(corrmat1) = names(SDSS_QSO)[c(2:12,15)]
x11(); 
levelplot(corrmat1,scales=list(x=list(rot=90)), xlab = "", ylab = "", main = "CE matrix")

# compared with other feature important measures, such as RFs
ce1imp = 0
for(j in 2:12){
  ce1imp[j-1] = copent(data1[,c(1,j)])
}
names(ce1imp) = names(data1)[2:12]
x11(width = 12); 
bce1 = barplot(ce1imp, ylab = "copula entropy", col = 'gray')

rf1 = randomForest(data1[,2:12],data1[,1])
imp1 = importance(rf1)
x11(width = 12);
barplot(t(imp1), col = 'gray', ylab = "IncNodePurity")

# prediction errors and examination
n2 = 5000
traindata1 = SDSS_QSO[1:n2,c(3:12,15)]
trainlabel1 = SDSS_QSO[1:n2,2]
testdata1 = SDSS_QSO[(n2+1):77429,c(3:12,15)]
testlabel1 = SDSS_QSO[(n2+1):77429,2]
idx1t = which(testlabel1>4.0) # samples in test set with red shift larger than 4.0, totally 853 samples

svm1 = svm(traindata1,trainlabel1)
psvm1 = predict(svm1,traindata1)
psvm1t = predict(svm1, testdata1)

rf1 = randomForest(traindata1,trainlabel1)
prf1 = predict(rf1,traindata1)
prf1t = predict(rf1, testdata1)

fset1 = c(1,2,3,5,7,9,11) #feature set
svm2 = svm(traindata1[,fset1],trainlabel1)
psvm2 = predict(svm2,traindata1[,fset1])
psvm2t = predict(svm2, testdata1[,fset1])

rf2 = randomForest(traindata1[,fset1],trainlabel1)
prf2= predict(rf2,traindata1[,fset1])
prf2t = predict(rf2, testdata1[,fset1])

mae1svm = mean(abs(trainlabel1-psvm1))
mae2svm = mean(abs(trainlabel1-psvm2))
mae1svmt = mean(abs(testlabel1-psvm1t))
mae1svmt2 = mean(abs(testlabel1[idx1t]-psvm1t[idx1t]))
mae2svmt = mean(abs(testlabel1-psvm2t))
mae2svmt2 = mean(abs(testlabel1[idx1t]-psvm2t[idx1t]))
mae1rf = mean(abs(trainlabel1-prf1))
mae2rf = mean(abs(trainlabel1-prf2))
mae1rft = mean(abs(testlabel1-prf1t))
mae1rft2 = mean(abs(testlabel1[idx1t]-prf1t[idx1t]))
mae2rft = mean(abs(testlabel1-prf2t))
mae2rft2 = mean(abs(testlabel1[idx1t]-prf2t[idx1t]))

mae = c(mae1svm,mae2svm,mae1svmt,mae2svmt,mae1svmt2,mae2svmt2,mae1rf,mae2rf,mae1rft,mae2rft,mae1rft2,mae2rft2); 
names(mae) = c("SVM1","SVM2","SVM1t1","SVM2t1","SVM1t2","SVM2t2","RF1","RF2","RF1t1","RF2t1","RF1t2","RF2t2")
x11(width = 12); 
barplot(mae, col = c("gray","black","gray","black","gray","black","gray","black","gray","black","gray","black"), ylab = "MAE")

x11();
plot(testlabel1[idx1t],psvm1t[idx1t], xlab = "true value", ylab = "predicted with SVM1")
x11();
plot(testlabel1[idx1t],psvm2t[idx1t], xlab = "true value", ylab = "predicted with SVM2")

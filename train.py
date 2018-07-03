import pandas as pd
import numpy as np
from shutil import copyfile
#from numba import vectorize,cuda
#from scipy.misc import imread, imresize
#import matplotlib.pyplot as plt
import os 

from  argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--lr",action="store", dest="lr",default = 0.001,type= float)
parser.add_argument("--momentum",action="store", dest="momentum",default = 0.01,type = float)
parser.add_argument("--num_hidden",action="store", dest="num_hidden",default = 1,type = int)
parser.add_argument("--sizes",action="store", dest="sizes",default = 1)
parser.add_argument("--activation",action="store", dest="activation",default = "sigmoid")
parser.add_argument("--loss",action="store", dest="loss",default = "ce")
parser.add_argument("--opt",action="store", dest="opt",default = "adam")
parser.add_argument("--batch_size",action="store", dest="batch_size",default = 20,type=int)
parser.add_argument("--anneal",action="store", dest="anneal",default = False)
parser.add_argument("--save_dir",action="store",dest="save_dir",default="pa1/")
parser.add_argument("--expt_dir",action="store",dest="expt_dir",default="pa1/exp1/")
parser.add_argument("--train",action="store",dest="train",default="train.csv")
parser.add_argument("--val",action="store",dest="val",default="val.csv")
parser.add_argument("--test",action="store",dest="test",default="test.csv")
parser.add_argument("--epochs",action="store",dest="epochs",default=7,type=int)
args = parser.parse_args()
   
szOutStr = args.sizes

sizearr = []
for x in szOutStr.split(","):
	sizearr.append(int(x))

#Parameter Catching up from the outside
np.random.seed(1234)
lr = args.lr
momentum =args.momentum #not applicable in gd
num_hidden = args.num_hidden
sizes = sizearr
activation = args.activation
loss = args.loss
opt = args.opt
batch_size = args.batch_size
anneal = args.anneal

#print(anneal)

save_dir = args.save_dir
#modelCollector = "/home/sp/Deep_1/data/cfr/"
expt_dir = args.expt_dir
train = args.train
validation = args.val
test = args.test
beta1,beta2 = 0.9,0.999
eps = 1e-8
epochs = args.epochs #max number of time whole data set should be feeded


#Reading Data from outside

#Training
Data = pd.read_csv(train, low_memory=False)
data = Data.values
y = data[0:55000,785:786] #Label Classification
TrainData = data[0:55000,1:785] #Traning Input data

#Validation
vdata = pd.read_csv(validation, low_memory=False)
valData = vdata.values
yval = valData[0:5001,785:786] #Label Classification
validationD = valData[0:5000,1:785]

#Testing 
tdata = pd.read_csv(test, low_memory=False)
tttData = tdata.values
testData = tttData[:,1:785]

#Data Loading Code Complete

#Data Normalization Code Start

(no_of_row,no_of_col) = (np.shape(TrainData)) #Training Operation
(val_row,val_col) = (np.shape(validationD)) #Validation Operation
(test_row,test_col) = (np.shape(testData)) #Validation Operation

mn = np.mean(TrainData,axis = 0) #Mean of training data
tiled_mn = np.tile(mn, (no_of_row, 1)) #Training : tiling up mean for substraction
val_tiled_mn = np.tile(mn,(val_row,1)) #Validation : tiling up mean for validation
test_tiled_mn = np.tile(mn,(test_row,1)) #Testing : tiling up mean for validation

st_dev = np.std(TrainData,axis = 0) #Standard deviation of training data
tiled_st_dev = np.tile(st_dev, (no_of_row, 1)) #Training : tiling of variance for division
val_tiled_st_dev = np.tile(st_dev,(val_row,1)) #Validation : tiling up variance for division
test_tiled_st_dev = np.tile(st_dev,(test_row,1)) #Testing : tiling up variance for division

#Normalizing Training Data
mn_shifted_data = TrainData - tiled_mn 
normalize_data = mn_shifted_data/tiled_st_dev 

#Normalizing Validation Data
mn_shifted_val_d = validationD - val_tiled_mn
norm_val_d = mn_shifted_val_d/val_tiled_st_dev

#Normalizing Testing Data
mn_shifted_test_d = testData - test_tiled_mn
norm_test_d = mn_shifted_test_d/test_tiled_st_dev

#Data Normalization Code End





#Initialization of some important stuffs
mName = str(num_hidden) + "_"+ str(sizes[0]) + "_" + opt + "_" + str(batch_size) + "_" + loss

pseudoV = np.array([0])
b = [] #Bias List
w = [pseudoV] #Weight List
inp_layer_Size = no_of_col #Set Input Vector size
out_layer_Size = 10 #Set output classes size
all_Layer = [inp_layer_Size] + sizes + [out_layer_Size]

#Random Initialization of Bias
for i in range(len(all_Layer)):
    #print(i)
    b.append((np.random.randn(all_Layer[i],1)/(np.sqrt(all_Layer[i])/2)))

b = np.array(b) ## B Array Ban Chuka hai

#Random Initialization of Weight
for i in range(len(all_Layer)-1):
    w.append((np.random.randn(all_Layer[i+1],all_Layer[i])/(np.sqrt(all_Layer[i])/2)))

w = np.array(w) ## w Array ban chuka hai

#Do not even think to delete this
#Validation super Matrix
yval_mat = np.zeros((10,5000))
for i in range(0,5000):
    yval_mat[(yval[i],i)] = 1



parameters = [opt,momentum,lr,batch_size,epochs,beta1,beta2,eps,anneal,loss,activation] 
otherDdt = [activation,loss,num_hidden,sizes,mName]
#End of initialization of important stuff


#######################################################################
#####################################################################
#######    Support Code  ##########################################
#################### not used in main script ###########################
##################################################################

def theSuperLifeSaver(modelCollector,parameters,otherDdt):
    [opt,momentum,lr,batch_size,epochs,beta1,beta2,eps,anneal,loss,activation] = parameters
    [activation,loss,num_hidden,sizes,mName] = otherDdt
    modelCounter = np.load(modelCollector+"counter.npy")
    dirOfModel = modelCollector + mName + str(modelCounter) + "/"
    drst = modelCollector + mName  + str(modelCounter)
    os.makedirs(drst)
    modelItrDir = dirOfModel + "Models"
    os.makedirs(modelItrDir)
    modelCounter[0] += 1
    np.save(modelCollector+"counter.npy",modelCounter)
    
    #Configuration_File
    confg = open(dirOfModel+"ConfigFile.txt","w")
    confg.write("No of Hidden Layer : %d\n" %(num_hidden))
    confg.write("Hidden Layer sizes : ")
    for layerdata in sizes:
        confg.write("%d ," %(layerdata))
    confg.write("\nOptimization function : %s" %(opt))
    confg.write("\nLoss function : %s" %(loss))
    confg.write("\nActivation function : %s" %(activation))
    confg.write("\nLearning Rate : %f" %(lr))
    confg.write("\nMomentum : %f" %(momentum))
    confg.write("\nBatch Size : %f" %(batch_size))
    confg.write("\nEpoch : %d" %(epochs))
    confg.write("\nBeta_1 : %f" %(beta1))
    confg.write("\nBeta_2 : %f" %(beta2))
    confg.write("\nEPS : %f" %(eps))
    confg.close()
    return dirOfModel
    
def theSuperMover(expt_dir,dirOfModel):
    copyfile(expt_dir+"log_train.txt",dirOfModel+"log_train.txt")
    copyfile(expt_dir+"log_val.txt",dirOfModel+"log_val.txt")

def theSuperModelSaver(SaveDir,weight,bias,step,epo):
    wb = np.array([weight,bias])
    np.save(SaveDir+"We"+str(epo)+"S"+str(step),wb)

def theSuperModelLoader(SaveDir,modelName):
    modl = np.load(SaveDir+modelName+".npy")
    return modl

##########################################################################
##############  End of Support Code  #########################################
#########################################################################

#################################################################
##########  General Important Functions #################################3
######################################################################

def preActivation(w,x,b):
    return np.dot(w,x) + b

def preActivationV(w,x,b):
    v1 = np.dot(w,x)
    (n_row,n_col) = np.shape(v1)
    act = v1 + np.tile(b,(n_col))
    return act
    
    
def yGen(val):
    vv = np.zeros((out_layer_Size,1))
    vv[(val,0)] = 1
    return vv
    
def h_sig(z):
    #To prevent overflow of sigmoid
    z = np.clip(z,-500,500)
    return 1/(1+np.exp(-z))

def h_tan(z):
    z = np.clip(z,-500,500)
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def h_deriv(z,activation):
    z = np.clip(z,-500,500)
    if activation == "sigmoid":
        return (h_sig(z)*(1-h_sig(z)))
    return (1-((h_tan(z))**2)) #Else Tanh
    
def h_softmax(a):
    n = np.exp(a) 
    d = np.sum(n)  
    return np.divide(n,d)

def h_softmaxV(A): #Used when n points are taken together
    Aexp = np.exp(A)
    (nr,nc) = np.shape(Aexp)
    AexpSum = np.sum(Aexp,axis=0)
    AexpSumTile = np.tile(AexpSum,(nr,1))
    hfinal = np.divide(Aexp,AexpSumTile)
    return hfinal
    

def loss_ce(p,q): #p,q should be column vectors
    return -np.dot(p.T,np.log(q))

def loss_se(y,f):
    return np.sum((y-f)**2)

def feedforward(theta,inp,activation):
    Wh = theta[0]
    bh = theta[1]
    L = len(Wh)-1
    a = [0]
    h = [inp]
    
    for i in range(1,L):
        a.append(preActivation(Wh[i],h[i-1],bh[i]))
        if (activation == "sigmoid"):    
            h.append(h_sig(a[i]))
        elif (activation == "tanh"):
            h.append(h_tan(a[i])) 
    a.append(preActivation(Wh[L],h[L-1],bh[L]))
    y = h_softmax(a[L])
    return [a,h,y]


def grad_se(Yact,Yhat):
    #Assuming that both Yact and Yhat are n*1 in shape.
    (n,bkr) = np.shape(Yact)
    Y_hf = np.tile(Yhat,(1,n))
    Y_hft = Y_hf.T
    In = np.eye(n)
    return 2*(np.dot(((Y_hf)*(In - Y_hft)),(Yhat-Yact)))


def backprop(a,h,y,label,loss,activation):
    L = len(a)-1
    grad_a=[]
    grad_w=[]
    grad_b=[]
    grad_h =[]
    if loss == "ce":
        grad_a.insert(0,-(yGen(label)-y))
    else:
        grad_a.insert(0,grad_se(yGen(label),y))
        #print("Here")
    for i in range(L,0,-1):
        #print(i)
        grad_w.insert(0,np.outer(grad_a[0],h[i-1]))
        grad_b.insert(0,grad_a[0])
        grad_h.insert(0,np.dot(w[i].T,grad_a[0]))
        grad_a.insert(0,np.multiply(grad_h[0],h_deriv(a[i-1],activation)))
    grad_w.insert(0,np.array([0]))
    grad_b.insert(0,np.array([0]))
    grad_w = np.array(grad_w)
    grad_b = np.array(grad_b)
    return [grad_w,grad_b]



def validation_of_data(val_data,Wh,bh,y_act,lr,step,epoc,yval_mat,val_log,loss,activation):
    L = len(w)-1 #To get the no of loop for forward prop.
    a = [0]
    vdt = val_data.T
    h = [vdt]

    #forward propogation start.....
    for i in range(1,L):
        a.append(preActivationV(Wh[i],h[i-1],bh[i]))
        if (activation == "sigmoid"):    
            h.append(h_sig(a[i]))
        elif (activation == "tanh"):
            h.append(h_tan(a[i])) 
    a.append(preActivationV(Wh[L],h[L-1],bh[L]))
    h_O = h_softmaxV(a[L])
    #forward propogation end.....
    
    #Calculating Accuracy
    y_pred = np.array([np.argmax(h_O,axis=0)]).T
    accVec = (y_pred == y_act)
    hit = sum(accVec)
    (tot,bad) = np.shape(accVec)
    errVal = ((tot-hit)/tot)*100
    
    #Calculating loss for the log file.
    if loss == "ce": #If loss function is Cross Entropy 
        lh_O = -(np.log(h_O))
        LargestRemain = lh_O*yval_mat
        LossTotalSum_val = (np.sum(LargestRemain))/tot
    else: #If loss function is Squared Error
        LossTotalSum_val = np.sum(((yval_mat - h_O)**2))/tot
    #print(fnalEntropy)
    val_log.write("Epoch %d, Step %d, Loss: %0.2f, Error: %0.2f, lr: %f\n" %(epoc,step,LossTotalSum_val,errVal,lr))
    val_log.flush()
    return np.array([epoc,step,LossTotalSum_val,errVal,lr])

def TestingCode(val_data,Wh,bh,activation):
    L = len(Wh)-1
    a = [0]
    vdt = val_data.T
    h = [vdt]

    for i in range(1,L):
        a.append(preActivationV(Wh[i],h[i-1],bh[i]))
        if (activation == "sigmoid"):    
            h.append(h_sig(a[i]))
        elif (activation == "tanh"):
            h.append(h_tan(a[i])) 
    a.append(preActivationV(Wh[L],h[L-1],bh[L]))
    h_O = h_softmaxV(a[L])
    y_pred = (np.array(np.argmax(h_O,axis=0))).tolist()
    id_d = (np.arange(0,10000)).tolist()
    d = {"id":id_d,"label":y_pred}
    df = pd.DataFrame(data=d)
    df.to_csv(expt_dir+"test_submission.csv")
    

#################################################################
##########  General Important Functions End #################################3
######################################################################

#################################################################
##########  The Hulking Crunching Giant Start #################################3
######################################################################
dirOfModel = ""

def gradient_Descent(inp,w,b,label,parameters,val_data,y_val,yval_mat,dirOfModel):
    
    #For saving Traing and validation parameter in csv for ploting graph
    #traincsvData = np.empty((0,5),float)
    #valcsvData = np.empty((0,5),float)
    
    [opt,momentum,lr,batch_size,epochs,beta1,beta2,eps,anneal,loss,activation] = parameters
    
    #Loading File for writing log file
    train_log = open(expt_dir+ "log_train.txt" , "w")
    val_log = open(expt_dir+ "log_val.txt" , "w")
    
    #Fixing Batch size issue.
    #55k entries. 100 steps priniting is only possible in case of batch size <550
    if batch_size > 550:
        d = (55000//batch_size)-1
    else:
        d = 100
    
    grad_w, grad_b = 0,0
    totalLossInCalc = 0.0  
    total = 0   
    wrong = 0   
    step = 0
    count = 0
    m_w,m_b,v_w,v_b = 0,0,0,0
    
    nagInit = 0

    if opt == "gd":
        momentum = 0
    if opt == "nag":
        nagInit = 1
    u_w,u_b = 0,0 #Initialize Update for momentum
    
    prev_loss = 1000 # for the purpose of annealing
    anneal_stopper = 0
    max_mistakes = 5
    i = 0 
    anealRegFlag = False
    
    #Setting Paramter for suffling of code
    sfl_para = "false"
    
    while(i <epochs):
        
        ####----for shuffling the data
        if sfl_para == "true":
            combined = np.insert(inp,[784],label,axis = 1)
            np.random.shuffle(combined)
            inp = combined[0:55000,0:784]
            label = combined[0:55000,784]
            label = label.astype(int)
            label = np.array([label])
            label = label.T
            ##print(label)
            ##print("New Label")
        
            #####
        no_of_point_seen = 0 #no of point seen in epoch till no
        if anneal_stopper >= max_mistakes:
            break
        
        n = len(inp)
        for point_no in range(n):
            
            #Data Points Formatting Starts
            data_point_v = inp[point_no]
            data_point_mat = np.array([data_point_v])
            data_point_mat = (data_point_mat.T)
            #Data Point Formatting Ends


            
             #   If NAG then nagInit parameter will be 1
             #   So gradient will be calculated on w_look_ahead
            

            theta=[w - momentum*nagInit*u_w,b-momentum*nagInit*u_b]

            #Gradient Finding Core Start
            [a,h,y] = feedforward(theta,data_point_mat,activation)
            [grad_w1,grad_b1] = backprop(a,h,y,label[point_no,0],loss,activation)
            #Gradient Finding Core Ends

            grad_w += grad_w1
            grad_b += grad_b1
            
            if loss == "ce":
                totalLossInCalc += loss_ce(yGen(label[point_no,0]),y) #Finding Cross Entropy
            else:
                totalLossInCalc += loss_se(yGen(label[point_no,0]),y) #Finding Cross Entropy
            no_of_point_seen +=1
            total +=1


            #Count missclassification start
            if np.argmax(y) != label[point_no,0]:
                wrong += 1
            #Count missclassification ends
            
            #Mini-Batch Core Start
            if no_of_point_seen % batch_size == 0:
                #print("here batch completed %d"%(step))
                no_of_point_seen = 0
                #Whole batch proccessed update parameter
                #If gradient Descent moment is default zero
                
                if opt == "adam":
                    count +=1 # number of times updation has been done
                    m_w = beta1*m_w +(1-beta1)*grad_w
                    m_b = beta1*m_b +(1-beta1)*grad_b

                    v_w = beta2*v_w +(1-beta2)*(grad_w**2)
                    v_b = beta2*v_b +(1-beta2)*(grad_b**2)

                    m_w_cap = m_w /(1-np.power(beta1,count))
                    m_b_cap = m_b /(1-np.power(beta1,count))

                    v_w_cap = v_w /(1-np.power(beta2,count))
                    v_b_cap = v_b /(1-np.power(beta2,count))
                    #print(v_w_cap)
                    t1 = np.power((v_w_cap + eps),0.5)
                    t2 = np.power((v_b_cap + eps),0.5)
                    u_w = (lr*m_w_cap)/t1
                    u_b = ((lr*m_b_cap))/t2

                else:
                    #Summarizing Updates for Momentum/NAG
                    u_w = momentum*u_w + lr*grad_w
                    u_b = momentum*u_b + lr*grad_b

                #end of 
                # for the sake of annealing
                if anneal == "true":
                    tmp_w = w
                    tmp_b = b
                #Applying Updates to parameter
                w = w - u_w
                b = b - u_b

                #Resetting temprary gradient holder
                grad_w,grad_b = 0,0
                step += 1
                
                if step%d == 0:
                    #print("here 100 step")
                    actLoss = totalLossInCalc/total
                    errorRate = (wrong/total)*100
                    train_log.write("Epoch %d, Step %d, Loss: %0.2f, Error: %0.2f, lr: %f\n" %(i,step,actLoss,errorRate,lr))
                    #theSuperModelSaver(dirOfModel+"Models/",w,b,step,i)
                    #traincsvData = np.append(traincsvData,np.array([[i,step,actLoss,errorRate,lr]]),axis=0)
                    train_log.flush()
                    valnpArD = validation_of_data(val_data,w,b,y_val,lr,step,i,yval_mat,val_log,loss,activation)
                    #valcsvData = np.append(valcsvData,[valnpArD],axis=0)
                    
                    if anneal == "true":
                        #print("here anneal")
                        newval_loss = valnpArD[2]
                        if newval_loss > prev_loss:
                            lr =lr/2
                            # store previous weights and biases
                            w = tmp_w
                            b = tmp_b
                            # for stopping the program when u exceed 5 times halving the learning rate
                        
                            anneal_stopper +=1
                            anealRegFlag = True
                            break
                        prev_loss =newval_loss    
                    wrong,total,totalLossInCalc = 0,0,0
            #Mini-Batch Core Ends
        step=0
        if anealRegFlag == True:
            anealRegFlag = False
            continue
        i = i+1
        #EpochCounter Ended
    theSuperModelSaver(save_dir,w,b,"Fin","al")
    TestingCode(norm_test_d,w,b,activation)
    #Gradient Descent Ended
    
    ######## Extra Support Code ########################3
    #comment out before submit code
    #toTrain = pd.DataFrame(traincsvData)
    #toTrain.to_csv(dirOfModel+"InfoTrain.csv")
    #toValidate = pd.DataFrame(valcsvData)
    #toValidate.to_csv(dirOfModel+"InfoValidate.csv")
    #theSuperMover(expt_dir,dirOfModel)
    ####

#################################################################
##########  The Hulking Crunching Giant End #################################3
######################################################################



### And the calling ######################################
#dirOfModel = theSuperLifeSaver(modelCollector,parameters,otherDdt)

#Remove DirOfModel in submission version
gradient_Descent(normalize_data,w,b,y,parameters,norm_val_d,yval,yval_mat,dirOfModel)
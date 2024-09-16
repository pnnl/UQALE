import os
import pandas as pd
import numpy as np
import subprocess
import shutil
import keras
import rdkit
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import PandasTools, AllChem, rdMolDescriptors
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesMolSupplier
from rdkit.Chem.Draw import IPythonConsole, MolDrawing, DrawingOptions
from scipy.stats import pearsonr, spearmanr


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
# from keras.models import load_model
import matplotlib.pyplot as plt
#import evidential_deep_learning as edl
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
# import seaborn as sns
import math
from scipy.stats.stats import pearsonr
from scipy import stats
from tensorflow.keras.layers import Layer



# Set lower and upper quantile
LOWER_ALPHA = 0.1
UPPER_ALPHA = 0.9

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

def check_jobs(job_ids):
    command = f"squeue --job {','.join(job_ids)}"
    output = subprocess.check_output(command, shell=True).decode().strip()
    # If the output contains only the header or is empty, all jobs have finished
    return len(output.splitlines()) > 1



def MSE(y, y_, reduce=True):
    ax = list(range(1, len(y.shape)))

    mse = tf.reduce_mean((y-y_)**2, axis=ax)
    return tf.reduce_mean(mse) if reduce else mse

def RMSE(y, y_):
    rmse = tf.sqrt(tf.reduce_mean((y-y_)**2))
    return rmse

def Gaussian_NLL(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))

    logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
    loss = tf.reduce_mean(-logprob, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss

def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    ax = list(range(1, len(y.shape)))

    log_liklihood = 0.5 * (
        -tf.exp(-logvar)*(mu-y)**2 - tf.math.log(2*tf.constant(np.pi, dtype=logvar.dtype)) - logvar
    )
    loss = tf.reduce_mean(-log_liklihood, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*tf.math.log(np.pi/v)  \
        - alpha*tf.math.log(twoBlambda)  \
        + (alpha+0.5) * tf.math.log(v*(y-gamma)**2 + twoBlambda)  \
        + tf.math.lgamma(alpha)  \
        - tf.math.lgamma(alpha+0.5)

    return tf.reduce_mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*tf.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*tf.math.log(tf.abs(v2)/tf.abs(v1))  \
        - 0.5 + a2*tf.math.log(b1/b2)  \
        - (tf.math.lgamma(a1) - tf.math.lgamma(a2))  \
        + (a1 - a2)*tf.math.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = tf.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return tf.reduce_mean(reg) if reduce else reg

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg



class DenseNormal(Layer):
    def __init__(self, units):
        super(DenseNormal, self).__init__()
        self.units = int(units)
        self.dense = Dense(2 * self.units)

    def call(self, x):
        output = self.dense(x)
        mu, logsigma = tf.split(output, 2, axis=-1)
        sigma = tf.nn.softplus(logsigma) + 1e-6
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.units)

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config['units'] = self.units
        return base_config


class DenseNormalGamma(Layer):
    def __init__(self, units):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.dense = Dense(4 * self.units, activation=None)

    def evidence(self, x):
        # return tf.exp(x)
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4 * self.units)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config

def EvidentialRegressionLoss(true, pred):
    return EvidentialRegression(true, pred, coeff=0.003)

def get_EDL_unc(model, x_test):
    y_test_pred = model(x_test)
    data_uncertainty = y_test_pred[:,3]/(y_test_pred[:,2] - 1)
    model_uncertainty = y_test_pred[:,3]/(y_test_pred[:,1]*(y_test_pred[:,2] - 1))
    total_uncertainty = data_uncertainty + model_uncertainty
    return y_test_pred[:,0], data_uncertainty, model_uncertainty, total_uncertainty


act = {0:'relu', 1:'selu', 2:'sigmoid'}

args={'a1': 2, 'a2': 0, 'a3': 1, 'a4': 1, 'a5': 0, 'bs': 1, 'd1': 0.10696194799818459, 'd2': 0.6033824611348487,\
      'd3': 0.7388531806558837, 'd4': 0.9943053700072028, 'd5': 0.016358259737496605, 'h1': 128.0, 'h2': 576.0,\
      'h3': 448.0, 'h4': 256.0, 'h5': 128.0, 'lr': 0, 'nfc': 0, 'opt': 1}












SFE_output = "AL_samples_SFE.csv"
AL_mols = "AL_mols.csv"
MSE_R2 = "MSE_R2.csv"
saved_indices_file = "saved_indices.csv"

try:
    df_AL_mols = pd.read_csv(AL_mols)
except FileNotFoundError:
    df_AL_mols = pd.DataFrame()
    
X_train = np.genfromtxt('X_train_cluster_1.csv', delimiter=',')
X_val = np.genfromtxt('X_val_cluster_1.csv', delimiter=',')
# X_test = np.genfromtxt('X_test.csv', delimiter=',')
y_train = np.genfromtxt('y_train_cluster_1.csv', delimiter=',')
y_val = np.genfromtxt('y_val_cluster_1.csv', delimiter=',')
# y_test = np.genfromtxt('y_test.csv', delimiter=',')

pubchem_testset_mdm = np.genfromtxt('pubchem_testset_mdm.csv', delimiter=',')
pubchem_testset_y = np.genfromtxt('pubchem_testset_y.csv', delimiter=',')
pubchem_mdm = np.genfromtxt('pubchem_mdm.csv', delimiter=',')
pubchem_smiles = pd.read_csv('pubchem_smiles.csv')

uncert_prob_dist = np.genfromtxt('uncert_prob_dist_mdm.csv', delimiter=',')

# Number of iterations
iterations = 20

base_model = 'MDM' # 'GBM'
AL_method = 'EDL' # 'random', 'density', 'GBM'

for _ in range(iterations):

	if os.path.isfile(SFE_output):
	    df_SFE_output = pd.read_csv(SFE_output)
	    df_SFE_output_features = pubchem_mdm[df_SFE_output['Name']]
	    X_train = np.concatenate((X_train, df_SFE_output_features))
	    y_train = np.concatenate((y_train, df_SFE_output['total(kcal/mol)']))
	    df_SFE_output_nrow = df_SFE_output.shape[0]
	    print('added SFE mols')

	mask = ~np.isnan(y_train)
	X_train = X_train[mask]
	y_train = y_train[mask]
    
    if base_model == 'GBM':
		lower_model = GradientBoostingRegressor(loss="quantile",                   
		                                        alpha=LOWER_ALPHA)
		mid_model = GradientBoostingRegressor(loss="squared_error")
		upper_model = GradientBoostingRegressor(loss="quantile",
		                                        alpha=UPPER_ALPHA)

		lower_model.fit(X_train, y_train)
		mid_model.fit(X_train, y_train)
		upper_model.fit(X_train, y_train)

		pubchem_testset_pred = mid_model.predict(pubchem_testset_mdm)#/4.184

	elif base_model == 'MDM':
		model = tf.keras.Sequential([
		    Dense(int(args['h1']), input_shape = (X_train.shape[1],)),
		    Activation(act[args['a1']] ),
		    Dropout(args['d1'] ),
		    Dense(int(args['h2'])  ),
		    Activation(act[args['a2']] ),
		    Dropout(args['d2'] ),
		    DenseNormalGamma(1),
		])


		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
		model.compile(
		    optimizer=tf.keras.optimizers.Adam(5e-4),
		    loss=EvidentialRegressionLoss)
		model.fit(X_train, y_train, batch_size=64, epochs=1000, validation_data = (X_val, y_val), verbose = 0, callbacks=[callback])

		pubchem_testset_pred, data_uncertainty, model_uncertainty, total_uncertainty = get_EDL_unc(model, pubchem_testset_mdm)

	MSE = mean_squared_error(pubchem_testset_y, pubchem_testset_pred)

	R2 = r2_score(pubchem_testset_y, pubchem_testset_pred)

	if os.path.isfile(SFE_output):
		df_MSE_R2 = pd.DataFrame([{'MSE': MSE, 'R2': R2, 'addNum': df_SFE_output_nrow}])
	else:
		df_MSE_R2 = pd.DataFrame([{'MSE': MSE, 'R2': R2}]) 

	if os.path.isfile(MSE_R2):
	    df_MSE_R2.to_csv(MSE_R2, mode='a', header=False, index=False)  # Write the header only on the first iteration
	else:
	    df_MSE_R2.to_csv(MSE_R2, mode='a', header=True, index=False)


	if os.path.isfile(saved_indices_file):
	    saved_indices = np.genfromtxt(saved_indices_file, delimiter=',', dtype=int)
	    pubchem_mdm_now = np.delete(pubchem_mdm, saved_indices, axis=0)
	    pubchem_smiles_now = pubchem_smiles.drop(pubchem_smiles.index[saved_indices])

	    uncert_prob_dist_now = np.delete(uncert_prob_dist, saved_indices, axis=0)
	else:
	    pubchem_mdm_now = pubchem_mdm.copy()
	    pubchem_smiles_now = pubchem_smiles.copy()
	    uncert_prob_dist_now = uncert_prob_dist.copy()

	if AL_method == 'GBM':
		predictions_pc = pd.DataFrame()
		predictions_pc['lower'] = lower_model.predict(pubchem_mdm_now)
		predictions_pc['mid'] = mid_model.predict(pubchem_mdm_now)
		predictions_pc['upper'] = upper_model.predict(pubchem_mdm_now)

		unc = abs(predictions_pc['upper'] - predictions_pc['lower'])/2
		sample_probabilities = unc / np.sum(unc)
		sample_probabilities = np.asarray(sample_probabilities).astype('float64')
		sample_probabilities = sample_probabilities / np.sum(sample_probabilities)
		sample_indices = np.random.choice(pubchem_smiles_now['Name'], size=50, p=sample_probabilities, replace=False)


	elif AL_method == 'MDM':
		pubchem_mdm_now_pred, data_uncertainty, model_uncertainty, total_uncertainty = get_EDL_unc(model, pubchem_mdm_now)

		unc = total_uncertainty - np.min(total_uncertainty)
		sample_probabilities = unc / np.sum(unc)
		sample_probabilities = np.asarray(sample_probabilities).astype('float64')
		sample_probabilities = sample_probabilities / np.sum(sample_probabilities)
		sample_indices = np.random.choice(pubchem_smiles_now['Name'], size=50, p=sample_probabilities, replace=False)

	elif AL_method == 'density':
		sample_probabilities = uncert_prob_dist_now / np.sum(uncert_prob_dist_now)
		sample_indices = np.random.choice(pubchem_smiles_now['Name'], size=50, p=sample_probabilities, replace=False)

	elif AL_method == 'random':
		sample_indices = np.random.choice(pubchem_smiles_now['Name'], size=50, replace=False)

	if os.path.isfile(saved_indices_file):
	    with open(saved_indices_file, 'ab') as f:
	        np.savetxt(f, sample_indices, delimiter=",")
	else:
	    np.savetxt(saved_indices_file, sample_indices, delimiter=",")

	selected_pubchem_smiles = pubchem_smiles_now[pubchem_smiles_now['Name'].isin(sample_indices)]

	selected_pubchem_smiles.to_csv('AL_samples.csv', index=False)

	df_AL_mols = pd.concat([df_AL_mols, pubchem_smiles_now[pubchem_smiles_now['Name'].isin(sample_indices)]], ignore_index=True)
	df_AL_mols.to_csv(AL_mols, index=False)




	df=pd.read_csv("AL_samples.csv", names=["Name","Smiles"],header=0,encoding="ISO-8859-1")

	f=open("gensmiles.csv","w")
	f.write("index,SMILES,formalcharge\n")
	f.close()

	df2 = pd.read_csv("gensmiles.csv", names=["index","SMILES",'formalcharge'],index_col=0,header=0,encoding="ISO-8859-1")


	exe_path2 = 'mopac-22.1.0-linux/bin/mopac'
	exe_path3 = 'pre2.exe'
	exe_path4 = 'ese-pm7.exe'
	path = ""

	 
	casn=df['Smiles'].tolist()
	name = df["Name"].tolist()
	new=[]
	new1=[]
	j=0
	for i in casn:
	    mol = Chem.MolFromSmiles(i)
	    m2 = Chem.AddHs(mol)
	    if(m2):
	        AllChem.EmbedMolecule(m2,useRandomCoords=True,maxAttempts=5000)
	#         AllChem.UFFOptimizeMolecule(m2) #optimize force field
	        AllChem.MMFFOptimizeMolecule(m2)
        	new_filename = str(name[j]) + '.xyz'
	        rdkit.Chem.rdmolfiles.MolToXYZFile(m2, new_filename)
	        reorder_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False,kekuleSmiles=True)
	        new.append(reorder_smiles)
	        formalcharge=rdkit.Chem.rdmolops.GetFormalCharge(mol)
	        new1.append(formalcharge)
	        j = j + 1

	df2['SMILES']=new
	df2['formalcharge']=pd.Series(new1)
	df2.to_csv("gensmiles.csv",mode='a',header=False)


	length1 = len(df)
	file3 = "bsub_crest"


	# for i in range(length1):
	j=0
	job_ids = []
	for i in name:
	    path1 = path + "/" + str(i)
	    os.mkdir(path1)
	    file1 = path + "/" + str(i)+".xyz"
	    file2 = path1 + "/" +"a1.xyz"
	    file4 = path1 + "/" +"bsub_crest"
	    shutil.copy2(file1, file2)
	    shutil.copy2(file3, file4)   
	    
	    os.chdir(path1)
	    f2=open(file4, mode='w', encoding='utf-8')    
	    print("#!/bin/csh",file=f2)
	    print("#SBATCH -A esmig",file=f2)
	#    print("#SBATCH -t 12:00:00",file=f2)
	    print("#SBATCH -t 3:00:00",file=f2)
	    print("#SBATCH -N 1",file=f2)
	    print("#SBATCH -n 64",file=f2)
	    print("#SBATCH -J job",file=f2)
	    print("#SBATCH -o 1.out",file=f2)
	    print("#SBATCH -e 1.err",file=f2)
	    print("#SBATCH -p slurm,short",file=f2)
	    print("module purge",file=f2)
	    print("module load gcc/8.1.0",file=f2)
	    print("module load openmpi/4.1.0",file=f2)    
	    print("test/crest a1.xyz --chrg",int(df2.iloc[j].formalcharge),"--alpb water",file=f2)
	    f2.close()
	    

	    # subprocess.run(["sbatch","bsub_crest"])
	    output = subprocess.check_output("sbatch bsub_crest", shell=True).decode().strip()
	    job_id = output.split()[-1]
	    job_ids.append(job_id)
	    time.sleep(2)

	    j=j+1
	    
	# time.sleep(10800)  
	# time.sleep(3600)
	print("Waiting for jobs to finish...")
	while check_jobs(job_ids):
		time.sleep(3600)  # Check every hour

	file_name = "crest_best.xyz"
	path2 = path + "/total"
	if not os.path.exists(path2):
		os.mkdir(path2)
	
	# for i in range(length1):
	for i in name:
	    path3 = path + "/" + str(i)
	    file5 = path3 + "/" + "crest_best.xyz"
	    
	    if os.path.exists(file5):
	        print(f"File '{file_name}' exists in folder '{path3}'.")
	        file6 = path2 + "/" + str(i) + ".xyz"
	        shutil.copy2(file5, file6)
	        
	    else:
	        print(f"File '{file_name}' does not exist in folder '{path3}'.")
	        time.sleep(10)  
	        
	os.chdir(path)
	#calculation
	file_name1 = 'opt.xyz'
	file_name2 = 'opt.mop'
	file_name3 = 'opt.out'

	file_list = os.listdir(path2)

	if _ == 0:
		f=open("AL_samples_SFE.csv","w")
		f.write("Name,elec(kcal/mol),corr(kcal/mol),total(kcal/mol)\n")
		f.close()
		df = pd.read_csv("AL_samples_SFE.csv", names=["Name,elec(kcal/mol),corr(kcal/mol),total(kcal/mol)"],index_col=0,header=0,encoding="ISO-8859-1")
	else:
		df = pd.DataFrame()

	# df = pd.read_csv("AL_samples_SFE.csv", names=["Name,elec(kcal/mol),corr(kcal/mol),total(kcal/mol)"],index_col=0,header=0,encoding="ISO-8859-1")
	# df = pd.read_csv("AL_samples_SFE.csv", names=["Name,elec(kcal/mol),corr(kcal/mol),total(kcal/mol)"],header=0,encoding="ISO-8859-1")

	c1 = []
	c2 = []
	c3 = []
	c4 = []

	nfiles = len(file_list)

	# print(nfiles)
	m = 0
	#for k in range(nfiles):
	for k in file_list:

	#    print(file_name)
	    file1 = path2 + "/" +str(k)
	    file2 = os.path.join(path, file_name1)
	#    tc1 = file_name.split('.',1)
	#    tc2 = tc1[0]
	#    c1.append(tc2)

	    shutil.copy2(file1, file2)   
	    with open(file1, 'r') as file3:
	        lines = file3.readlines()
	#        print(len(lines))
	    len1 = len(lines)
	    cc1 = df2['formalcharge']
	    file_name2=open('opt.mop', 'w')


	    cc2 = df2.iloc[m].formalcharge
	    print(m+1,cc2)
	    print(f"PM7 OPT CHARGE={cc2} PRECISE",file=file_name2)
	    print("cc\n",file=file_name2)

	    for i in range(2,len1):
	        elements = lines[i].split()
	        element = elements[0]
	        x = elements[1]
	        y = elements[2]
	        z = elements[3]
	        output_line = f"{element} {x} 1 {y} 1 {z} 1"
	        print(f"{element} {x} 1 {y} 1 {z} 1",file=file_name2)

	    
	    file_name2.close()

	    subprocess.run(["mopac-22.1.0-linux/bin/mopac","opt.mop"])
	    subprocess.call(["pre2.exe", str(cc2)])

	    result=subprocess.run(["ESE-PM7.exe","sol.txt"], capture_output=True, text=True)
	#    print(file_name)
	    a1 = result.stdout
	#    print(a1)
	    word1 = "kcal/mol\n"
	    word2 = "kcal/mol\n\n"
	    word3 = "kcal/mol\n\n\n"
	    a2 = a1.split(' ')
	    index1 = a2.index(word1)
	    index2 = a2.index(word2)
	    index3 = a2.index(word3)

	    k1 = str(k).split(".")
	    k2 = k1[0]
	    c1.append(k2)
	    c2.append(a2[index1-1])
	    c3.append(a2[index2-1])
	    c4.append(a2[index3-1])
	    m = m+1
	    
	shutil.rmtree('total', ignore_errors=True)

	df["Name"]=pd.Series(c1)
	df["elec(kcal/mol)"]=pd.Series(c2)
	df["corr(kcal/mol)"]=pd.Series(c3)
	df["total(kcal/mol)"]=pd.Series(c4)


	if _ == 0:
		df.to_csv("AL_samples_SFE.csv", mode='w', header=True, index=False)
	else:
		df.to_csv("AL_samples_SFE.csv", mode='a', header=False, index=False)

	for dir_name in os.listdir():
	    try:
	        int(dir_name)
	        if os.path.isdir(dir_name):
	        	shutil.rmtree(dir_name, ignore_errors=True)
	    except ValueError:
	        pass

	for file_name in os.listdir():
		if file_name.endswith(".xyz"):
			os.remove(file_name)
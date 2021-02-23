import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy import ndimage as image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
# function to carry out cal/val for a random forest regression
def cal_val_train_test(X,y,rf,path2calval,country_code,version):

    #split train and test subset, specifying random seed
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=29)
    rf.fit(X_train,y_train)
    y_train_predict = rf.predict(X_train)
    y_test_predict = rf.predict(X_test)

    r2 = [r2_score(y_train,y_train_predict),
            r2_score(y_test,y_test_predict)]
    rmse = [np.sqrt(mean_squared_error(y_train,y_train_predict)),
            np.sqrt(mean_squared_error(y_test,y_test_predict))]

    #create some pandas df
    df_train = pd.DataFrame({'obs':y_train,'sim':y_train_predict})
    df_train.sim[df_train.sim<0] = 0.

    df_test =  pd.DataFrame({'obs':y_test,'sim':y_test_predict})
    df_test.sim[df_test.sim<0] = 0.

    #plot
    sns.set()
    fig = plt.figure('cal/val random',figsize=(10,6))
    fig.clf()
    #first ax
    titles = ['a) Calibration','b) Validation']
    labels = ['R$^2$ = %.02f\nRMSE = %.02f' % (r2[0],rmse[0]),
            'R$^2$ = %.02f\nRMSE = %.02f' % (r2[1],rmse[1])]

    #for dd, df in enumerate([df_train.sample(1000),df_test.sample(1000)]):
    for dd, df in enumerate([df_train,df_test]):
        ax = fig.add_subplot(1,2,dd+1,aspect='equal')
        sns.regplot(x='obs',y='sim',data=df,scatter_kws={'s':1},line_kws={'color':'k'},ax=ax)
        ax.annotate(labels[dd], xy=(0.95,0.05), xycoords='axes fraction',
                        backgroundcolor='none',horizontalalignment='right',
                        verticalalignment='bottom', fontsize=10)
        #adjust style
        ax.set_title(titles[dd]+' (n = %05i)' % df.shape[0])
        #plt.xlim(0,1);plt.ylim(0,1)
        plt.xlabel('AGB from Avitabile et al. (2016) [Mg ha $^{-1}$]')
        plt.ylabel('Reconstructed AGB [Mg ha $^{-1}$]')

    plt.savefig('%s/%s_%s_rf_iterative_calval.png' % (path2calval,country_code,version))
    return r2,rmse

# function to create block k-fold cal-val set
def get_k_fold_cal_val_blocked(template,block_res,buffer_width,training_mask=None,k=3):

    # Step 1: Create the blocks
    raster_res = template.attrs['res'][0]
    block_width = int(np.ceil(block_res/raster_res))
    blocks_array = np.zeros(template.values.shape)
    buffer = int(np.ceil(buffer_width/raster_res))

    # if no training mask specified, include full domain extent
    if training_mask is None:
        training_mask = np.ones(blocks_array.shape)

    block_label = 0
    for rr,row in enumerate(np.arange(0,blocks_array.shape[0],block_width)):
        for cc, col in enumerate(np.arange(0,blocks_array.shape[1],block_width)):
            blocks_array[row:row+block_width,col:col+block_width]=block_label
            block_label+=1

    # test blocks for training data presence
    blocks = blocks_array[training_mask]
    blocks_keep,blocks_count = np.unique(blocks,return_counts=True)
    # remove blocks with no training data
    blocks_array[~np.isin(blocks_array,blocks_keep)]=np.nan

    # Step 2: Split blocks into k-fold test and train sets (with buffer
    val_blocks={};cal_blocks = {}
    # permute blocks randomly
    val_blocks_array = blocks_array.copy()
    blocks_kfold = np.random.permutation(blocks_keep)
    blocks_in_fold = int(np.floor(blocks_kfold.size/k))
    kfold_idx = np.zeros(training_mask.sum())
    for ii in range(0,k):
        blocks_iter = blocks_kfold[ii*blocks_in_fold:(ii+1)*blocks_in_fold]
        # label calibration blocks with fold
        val_blocks_array[np.isin(blocks_array,blocks_iter)]=ii

        blocks_to_be_allocated = blocks_kfold.size%k
        blocks_allocated = blocks_kfold.size-blocks_to_be_allocated

    for ii in range(0,k):
        # add any outstanding blocks
        if ii+blocks_allocated<blocks_kfold.size:
            blocks_iter = blocks_kfold[ii+blocks_allocated]
            val_blocks_array[np.isin(blocks_array,blocks_iter)]=ii

        #kfold_idx[np.isin(blocks,blocks_iter)]=ii
        # expand neighbourhood of validation blocks with buffer
        val_data_mask = np.all((val_blocks_array==ii,training_mask),axis=0)
        val_data_mask = image.binary_dilation(val_data_mask,iterations=buffer)

        # save validation and calibration blocks to dictionary
        val_blocks['iter%i' % (ii+1)]=np.all((val_blocks_array==ii,training_mask),axis=0)[training_mask]
        cal_blocks['iter%i' % (ii+1)]=np.all((~val_data_mask,training_mask),axis=0)[training_mask]

    return cal_blocks, val_blocks

"""
# function to create block k-fold cal-val set
def get_k_fold_cal_val_blocked(template,block_res,buffer_width,training_mask=None,k=3):

    # Step 1: Create the blocks
    raster_res = template.attrs['res'][0]
    block_width = int(np.ceil(block_res/raster_res))
    blocks_array = np.zeros(template.values.shape)

    # if no training mask specified, include full domain extent
    if training_mask is None:
        training_mask = np.ones(blocks_array.shape)

    block_label = 0
    for rr,row in enumerate(np.arange(0,blocks_array.shape[0],block_width)):
        for cc, col in enumerate(np.arange(0,blocks_array.shape[1],block_width)):
            blocks_array[row:row+block_width,col:col+block_width]=block_label
            block_label+=1

    # test blocks for training data presence
    blocks = blocks_array[training_mask]
    blocks_keep,blocks_count = np.unique(blocks,return_counts=True)
    # remove blocks with no training data
    blocks_array[~np.isin(blocks_array,blocks_keep)]=np.nan

    # Step 2: Split blocks into k-fold test and train sets (with buffer

    # permute blocks randomly
    cal_blocks_array = blocks_array.copy()
    blocks_kfold = np.random.permutation(blocks_keep)
    blocks_in_fold = int(np.ceil(blocks_kfold.size/k))
    kfold_idx = np.zeros(training_mask.sum())
    for ii in range(0,k):
        blocks_iter = blocks_kfold[ii*blocks_in_fold:(ii+1)*blocks_in_fold]
        print(np.isin(blocks_array,blocks_iter).sum())
        # label calibration blocks with fold
        cal_blocks_array[np.isin(blocks_array,blocks_iter)]=ii
        kfold_idx[np.isin(blocks,blocks_iter)]=ii

    cal_blocks_array[~training_mask] = np.nan
    cal_blocks = cal_blocks_array[training_mask]

    # Step 3: filter the blocks based on proximity to validation data to avoid
    # neighbouring pixels biasing the validation
    buffer = int(np.ceil(buffer_width/raster_res))
    val_blocks_array = cal_blocks_array.copy()
    for ii in range(0,k):
        cal_data_mask = np.all((cal_blocks_array!=ii,training_mask),axis=0)
        # expand neighbourhood with buffer
        cal_data_mask = image.binary_dilation(cal_data_mask,iterations=buffer)
        val_blocks_array[np.all((cal_data_mask,val_blocks_array==ii),axis=0)]=np.nan

    val_blocks = val_blocks_array[training_mask]

    return cal_blocks, val_blocks
"""

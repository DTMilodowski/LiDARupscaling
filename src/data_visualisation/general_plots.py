"""
general_plots
--------------------------------------------------------------------------------
GENERATE PLOTS FOR WORKSHOP
David T. Milodowski, 25/03/2019
"""

"""
import libraries needed
"""
import numpy as np
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

"""
Part 2: Random forests
"""
# Figure 1 simple cross plot of three test datasets
def plot_test_data(X1,y1,X2,y2,X3,y3,show=True):
    fig,axes = plt.subplots(nrows=1,ncols=3,figsize = (8,3))
    axes[0].plot(X1,y1,'.')
    axes[1].plot(X2,y2,'.')
    axes[2].plot(X3,y3,'.')
    axes[0].set_xlim((0,10));axes[0].set_ylim((-5,28))
    axes[2].set_xlim((0,10));axes[2].set_ylim((-5,28))
    fig.tight_layout()
    if show:
        fig.show()
    return fig,axes

# Figure 2 adding regression results
def plot_test_data_with_regression_results(X1,y1,X2,y2,X3,y3,X_test,y1_test,y1_test_lm,y2_test,y3_test,y3_test_lm,show=True):
    fig2,axes = plt.subplots(nrows=1,ncols=3,figsize = (8,3))
    axes[0].plot(X1,y1,'.',label='data',color='0.5')
    axes[0].plot(X_test,y1_test,'-',color='red',label='naive rf model')
    axes[0].plot(X_test,y1_test_lm,'-',color='blue',label='linear regression')
    axes[1].plot(X2,y2,'.',color='0.5')
    axes[1].plot(X_test,y2_test,'-',color='red')
    axes[2].plot(X3,y3,'.',color='0.5')
    axes[2].plot(X_test,y3_test,'-',color='red')
    axes[2].plot(X_test,y3_test_lm,'-',color='blue')
    axes[0].set_xlim((0,10));axes[0].set_ylim((-5,28))
    axes[2].set_xlim((0,10));axes[2].set_ylim((-5,28))
    axes[0].legend(loc='lower right',fontsize = 8)
    fig2.tight_layout()
    if show:
        fig2.show()
    return fig2,axes

# Figure 3, basic cal-val example
def plot_cal_val(y_train,y_train_rf,y_test,y_test_rf,show=True):
    calval_df = pd.DataFrame(data = {'val_obs': y_test,
                                     'val_model': y_test_rf,
                                     'cal_obs': y_train,
                                     'cal_model': y_train_rf})

    fig3,axes= plt.subplots(nrows=1,ncols=2,figsize=[8,4])
    sns.regplot(x='cal_obs',y='cal_model',data=calval_df,marker='+',
                truncate=True,ci=None,ax=axes[0])
    axes[0].annotate('calibration R$^2$ = %.02f\nRMSE = %.02f' %
                (r2_score(y_train,y_train_rf),np.sqrt(mean_squared_error(y_train,y_train_rf))),
                xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
                horizontalalignment='left', verticalalignment='top')
    sns.regplot(x='val_obs',y='val_model',data=calval_df,marker='+',
                truncate=True,ci=None,ax=axes[1])
    axes[1].annotate('validation R$^2$ = %.02f\nRMSE = %.02f'
                % (r2_score(y_test,y_test_rf),np.sqrt(mean_squared_error(y_test,y_test_rf))),
                xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
                horizontalalignment='left', verticalalignment='top')
    axes[0].axis('equal')
    axes[1].axis('equal')
    fig3.tight_layout()
    if show:
        fig3.show()
    return fig3,axes

# figure 4,cal_val with regression line
def plot_cal_val_agb(y_train,y_train_rf,y_test,y_test_rf,show=True):
    cal_df = pd.DataFrame(data = {'cal_obs': y_train,
                                  'cal_model': y_train_rf})
    val_df = pd.DataFrame(data = {'val_obs': y_test,
                                  'val_model': y_test_rf})


    fig4,axes= plt.subplots(nrows=1,ncols=2,figsize=[8,4])
    sns.regplot(x='cal_obs',y='cal_model',data=cal_df,marker='.',
                truncate=True,ci=None,ax=axes[0],
                scatter_kws={'alpha':0.01,'edgecolor':'none'},
                line_kws={'color':'k'})
    axes[0].annotate('calibration R$^2$ = %.02f\nRMSE = %.02f' %
                (r2_score(y_train,y_train_rf),np.sqrt(mean_squared_error(y_train,y_train_rf))),
                xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
                horizontalalignment='left', verticalalignment='top')
    sns.regplot(x='val_obs',y='val_model',data=val_df,marker='.',
                truncate=True,ci=None,ax=axes[1],
                scatter_kws={'alpha':0.01,'edgecolor':'none'},
                line_kws={'color':'k'})
    axes[1].annotate('validation R$^2$ = %.02f\nRMSE = %.02f'
                % (r2_score(y_test,y_test_rf),np.sqrt(mean_squared_error(y_test,y_test_rf))),
                xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
                horizontalalignment='left', verticalalignment='top')
    axes[0].axis('equal')
    axes[1].axis('equal')
    fig4.tight_layout()
    if show:
        fig4.show()
    return fig4,axes

# figure 5: Importances
def plot_importances(imp_df,show=True):
    fig5,axes= plt.subplots(nrows=1,ncols=2,figsize=[8,8],sharex=True)
    sns.barplot(x='permutation_importance',y='variable',data=imp_df,ax=axes[0])
    axes[0].annotate('permutation importance',
                xy=(0.95,0.98), xycoords='axes fraction',backgroundcolor='none',
                horizontalalignment='right', verticalalignment='top')
    sns.barplot(x='gini_importance',y='variable',data=imp_df,ax=axes[1])
    axes[1].annotate('gini importance',
                xy=(0.95,0.98), xycoords='axes fraction',backgroundcolor='none',
                horizontalalignment='right', verticalalignment='top')
    plt.setp(axes[1].get_yticklabels(),visible=False)
    axes[1].yaxis.set_ticks_position('left')
    fig5.tight_layout()
    if show:
        fig5.show()
    return fig5,axes

# Plot partial dependencies
def plot_partial_dependencies_simple(rf, X, x_label=None, y_label=None,
                                        variable_position=0, show=True):

    n_variables=X.shape[1]
    var_ = np.linspace(np.min(X[:,variable_position]),np.max(X[:,variable_position])+1,200)
    X_RM = np.zeros((var_.size,n_variables))
    for i in range(0,n_variables):
        if i == variable_position:
            X_RM[:,i] = var_.copy()
        else:
            X_RM[:,i] = np.mean(X[:,i])

    # predict with rf model
    y_RM = rf.predict(X_RM)
    # now plot
    fig6,ax = plt.subplots(figsize=[8,5])
    ax.plot(var_, y_RM,'-')
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    fig6.tight_layout()
    if show:
        fig6.show()
    return fig6, ax


def plot_partial_dependencies_multiple(rf, X, x_label=None, y_label=None,
                                        variable_position=0, show=True):
    n_variables=X.shape[1]

    var_ = np.linspace(np.min(X[:,variable_position]),np.max(X[:,variable_position]),200)
    X_RM = np.zeros((var_.size,n_variables))

    for i in range(0,n_variables):
        if i == variable_position:
            X_RM[:,i] = var_.copy()
        else:
            X_RM[:,i] = np.mean(X[:,i])

    # predict with rf model
    y_RM = rf.predict(X_RM)

    # now plot
    fig7,ax = plt.subplots(figsize=[8,5])


    N_iterations = 20
    for i in range(0,N_iterations):
        sample_row = np.random.randint(0,X.shape[0]+1)
        X_i = np.zeros((var_.size,n_variables))
        for j in range(0,n_variables):
            if j == variable_position:
                X_i[:,j] = var_.copy()
            else:
                X_i[:,j] = (X[sample_row,j])

        # predict with rf model
        y_i = rf.predict(X_i)
        ax.plot(var_, y_i,'-',c='0.5',linewidth=0.5,alpha=0.8)

    ax.plot(var_, y_RM,'-') # also plot line from before for comparison
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    fig7.tight_layout()
    if show:
        fig7.show()
    return fig7,axis

# just add new partial dependency onto existing plot
def add_partial_dependency_instance(ax,rf,X,variable_position=1,show=True,N_iterations=1):
    ax=ax or plt.gca()
    n_variables=X.shape[1]
    var_ = np.linspace(np.min(X[:,variable_position]),np.max(X[:,variable_position]),200)

    for i in range(0,N_iterations):
        sample_row = np.random.randint(0,X.shape[0]+1)
        X_i = np.zeros((var_.size,n_variables))
        for j in range(0,n_variables):
            if j == variable_position:
                X_i[:,j] = var_.copy()
            else:
                X_i[:,j] = (X[sample_row,j])

        # predict with rf model
        y_i = rf.predict(X_i)
    return ax.plot(var_, y_i,'-',c='0.5',linewidth=0.5,alpha=0.8)

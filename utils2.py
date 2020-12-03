import os
import pickle
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def classifier_check(X_train, X_test, y_train, y_test):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.datasets import load_iris
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    
    sns.set_theme(style="white")   #darkgrid, whitegrid, dark, white, ticks
    
    # Check data imbalance
    training_count = pd.concat([X_train, y_train],axis=1)
    training_count['test_tran'] = 'train'
    test_count = pd.concat([X_test, y_test],axis=1)
    test_count['test_tran'] = 'test'
    test_train_count = pd.concat([training_count[['classifier','test_tran']],test_count[['classifier','test_tran']]])
    sns.set_theme(style="white")   #darkgrid, whitegrid, dark, white, ticks
    sns.countplot(x="test_tran", hue='classifier',data=test_train_count)
    plt.show

    classifiers = {
        "Naive Bayes": GaussianNB(),
        "Logisitic Regression": LogisticRegression(random_state = 1),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(random_state = 1),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state = 1),
        "Random Forest Classifier": RandomForestClassifier(random_state = 1,max_depth= 30, min_samples_leaf= 1, min_samples_split=2, n_estimators=100),
        "AdaBoost Classifier":AdaBoostClassifier(random_state = 1,n_estimators=50,learning_rate= 0.1),
        "Gradient Boosting Classifier":GradientBoostingClassifier(random_state = 1),
    }


    f, axes = plt.subplots(2, 4, figsize=(20, 10), sharey='row')
    accuracy = {}
    kappa = {}
    f1 = {}
    r=0
    j=0
    for i, (key, classifier) in enumerate(classifiers.items()):
        if r >=4:
            j=1
            r=0

        y_pred = classifier.fit(X_train, y_train).predict(X_test)
        cf_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=axes[j][r])
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i!=0:
            disp.ax_.set_ylabel('')
        accuracy.update({key:round(accuracy_score(y_test, y_pred),3)})
        kappa.update({key:round(cohen_kappa_score(y_test, y_pred),3)})
        f1.update({key:round(f1_score(y_test, y_pred),3)})
        r+=1

    
    #f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.3, hspace=0.02)
    accuracy = pd.DataFrame([accuracy])
    kappa = pd.DataFrame([kappa])
    f1 = pd.DataFrame([f1])
    results = pd.concat([accuracy,kappa,f1],axis=0)
    results.index=['Accuracy','Kappa','f1']
    results = results.transpose()
    
    f.colorbar(disp.im_, ax=axes)
    plt.show()
    display(results.sort_values(['Accuracy','Kappa','f1'],ascending=False))



def load_shuffle_split_scale():
    # Import data
    loaded_data = pd.read_csv('phrophet_var_call.csv')
    loaded_data['callPut'] = 1

    loaded_data2 = pd.read_csv('phrophet_var_put.csv')
    loaded_data2['callPut'] = 0

    loaded_data = pd.concat([loaded_data,loaded_data2],ignore_index=True)
    loaded_data = loaded_data.reset_index(drop=True)
    #display(data[data.stock_scale == 'AAPL'])
    data = loaded_data.drop(columns=['Percent_profit','prophet_yhat_lower','prophet_yhat_upper','prophet_yhat_lower','callPut'])
    
    # Shuffle stocks
    list_stocks = list(data.stock_scale.unique())
    np.random.shuffle(list_stocks)
    cut_off = round(0.7*len(list_stocks))
    stocks_train = list_stocks[:cut_off]
    stocks_test = list_stocks[cut_off:]
    X_train = pd.DataFrame(columns = data.columns)
    
    # Test train split 
    for i in stocks_train:
        new = data[data.stock_scale ==i]
        X_train = pd.concat([X_train,new])

    X_test = pd.DataFrame(columns = data.columns)
    for i in stocks_test:
        new = data[data.stock_scale ==i]
        X_test = pd.concat([X_test,new])
        
        
    y_train = X_train['classifier']
    y_test = X_test['classifier']
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')


    X_train = X_train.drop(columns=['stock_scale','classifier','p_L'])
    X_test = X_test.drop(columns=['stock_scale','classifier','p_L'])
    
    X_train_index = X_train.index
    y_train_index = y_train.index
    
    # Scale the data
    cols = X_train.columns
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train,columns=cols)

    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test,columns=cols)

    X_train['index'] = X_train_index
    X_train = X_train.set_index('index',drop=True)


    return X_train, X_test, y_train, y_test, loaded_data

def upsampling(X_train, X_test, y_train, y_test):
    #y_train = y_train.reset_index(drop=True)
    # Join training sets 
    data_upsampled = pd.concat([X_train,y_train],axis=1)

    # Count catagories
    category_0 = data_upsampled[data_upsampled['classifier'] == 0]
    category_1 = data_upsampled[data_upsampled['classifier'] == 1]

    # resample the minority group to the count of the majority
    category_1 = category_1.sample(len(category_0), replace=True)

    # Join datasets again row wise
    data_upsampled = pd.concat([category_0, category_1], axis=0)

    # Shuffle the data
    data_upsampled = data_upsampled.sample(frac=1)

    # Seperate target variable
    X_train_up = data_upsampled.drop(columns=['classifier'])
    y_train_up = data_upsampled['classifier']

    # Checking for duplicate values
    a = X_train_up.index.unique()
    b = X_test.index.unique()
    list_double = [i for i in a if i in b]
    
    return X_train_up, X_test, y_train_up, y_test

def generate_models(X_train, X_test, y_train, y_test,show_graphs=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    ###########################
    #Linear Regression
    ###########################

    reg = LogisticRegression(penalty = 'none',
                             tol=0.5,
                             random_state=1)
    
    reg.fit(X_train, y_train)
    reg_pred = reg.predict(X_test)

   
    if show_graphs == True:
         # print the initial results
        print("Linear Regression")
        print("The accuracy of the model on test set is: %4.2f " % accuracy_score(y_test, reg_pred))
        print("The Kapa of your model is: ",round(cohen_kappa_score(y_test,reg_pred),3))

        # plot confusion matrix
        confusion_matrix(y_test, reg_pred)
        plot_confusion_matrix(reg, X_test, y_test) 
        plt.show()
        # print classification report
        print(classification_report(y_test, reg_pred))
    # Save model
    with open('lin_reg.pkl','wb') as file:
            pickle.dump(reg,file)
    
    ###########################
    #Support Vector Classifier
    ###########################   

    svc = SVC(random_state=1,probability=True)
    svc.fit(X_train, y_train)
    svc_pred = reg.predict(X_test)
    
    if show_graphs == True:

        # print the initial results
        print("Support Vector Classifier")
        print("The accuracy of the model on test set is: %4.2f " % accuracy_score(y_test, reg_pred))
        print("The Kapa of your model is: ",round(cohen_kappa_score(y_test,reg_pred),3))
        # plot confusion matrix
        confusion_matrix(y_test, svc_pred)
        plot_confusion_matrix(svc, X_test, y_test) 
        plt.show()
        # print classification report
        print(classification_report(y_test, svc_pred))
    # Save model
    with open('svc.pkl','wb') as file:
            pickle.dump(svc,file)

    ###########################
    #Random Forest
    ###########################
    RanFor = RandomForestClassifier(max_depth = 25, 
                                    n_estimators = 1200, 
                                    min_samples_split = 2, 
                                    min_samples_leaf = 1)
    
    RanFor.fit(X_train, y_train)
    RanFor_pred = RanFor.predict(X_test)
    
    if show_graphs == True:
        # print the initial results
        print("Random Forest")
        print("The accuracy of the model on test set is: %4.2f " % accuracy_score(y_test, RanFor_pred))
        print("The Kapa of your model is: ",round(cohen_kappa_score(y_test,RanFor_pred),3))

        # plot confusion matrix
        confusion_matrix(y_test, RanFor_pred)
        plot_confusion_matrix(RanFor, X_test, y_test) 
        plt.show()

        # print classification report
        print(classification_report(y_test, RanFor_pred))

    # Save model
    with open('forest.pkl','wb') as file:
            pickle.dump(RanFor,file)
    
    return reg_pred,svc_pred,RanFor_pred

def create_trade_list(reg_pred,svc_pred,RanFor_pred,loaded_data,X_test,y_test,show_graphs=True):
    with open ('lin_reg.pkl','rb') as file:
            reg = pickle.load(file)
    with open ('svc.pkl','rb') as file:
            svc = pickle.load(file)
    with open ('forest.pkl','rb') as file:
            RanFor = pickle.load(file)

    pred_results = pd.DataFrame(index=y_test.index)
    pred_results["y_test"] = y_test
    pred_results["y_forest"] = RanFor_pred
    pred_results["y_reg"] = reg_pred
    pred_results['y_svc'] = svc_pred
    pred_results['score'] = pred_results["y_forest"] + pred_results["y_reg"] + pred_results["y_svc"]

    # Append probablitities of achieving 1
    _, proba_forest = list(zip(*RanFor.predict_proba(X_test)))
    pred_results['proba_forest'] = proba_forest

    _, proba_reg = list(zip(*reg.predict_proba(X_test)))
    pred_results['proba_reg'] = proba_reg

    _, proba_svc = list(zip(*svc.predict_proba(X_test)))
    pred_results['proba_svc'] = proba_svc

    # Joing origional data
    pred_results = pd.merge(pred_results,loaded_data,left_index=True,right_index=True,how='left')

    # seperating predicted positives
    positives = pred_results[pred_results["score"]>=1]

    # columns to keep
    cols = ['stock_scale','y_test', 'y_forest','proba_forest','y_reg','proba_reg', 'proba_svc', 'mark_x','strikePrice','timeValue', 'p_L',
            'Percent_profit','vega','underlyingPrice','prophet_yhat','callPut']
    pred_results = pred_results[cols]

    # Seperating True posivites and false positives
    false_positives = positives[positives["y_test"]==0]
    true_positives = positives[positives["y_test"]==1]  

    # report of false positives
    false_positives['loss'] = np.where(false_positives.p_L <= 0, 1,0)
    
    if show_graphs == True:
        print("The number of false positives that were negative results are shown below and sum up to: ",len(false_positives[false_positives['loss']==1]))
        sns.displot(false_positives, x="p_L", hue='loss',binwidth=0.2, height=5, facet_kws=dict(margin_titles=True),legend=False,multiple="stack")
        sns.displot(false_positives, x="proba_forest", hue='loss',binwidth=0.04, height=5, facet_kws=dict(margin_titles=True),legend=False,multiple="stack")
        sns.displot(false_positives, x="proba_reg", hue='loss',binwidth=0.04, height=5, facet_kws=dict(margin_titles=True),legend=False,multiple="stack")
        sns.displot(false_positives, x="proba_svc", hue='loss',binwidth=0.04, height=5, facet_kws=dict(margin_titles=True),legend=False,multiple="stack")
        plt.show()

    return positives

def threshold_trades(df,th=0.5,inv=10000):
    
    #round fees (in and out)
    trade_fees = 0.04
    commision_fee = 2.6
    
    # Calculate required margins
    df['margin_required'] = df['mark_x']*100
    # profit
    df['profit'] = df['p_L']*100-(trade_fees + commision_fee)
    
    # order list and drop duplicate stocks
    df['proba_mean'] = ((df['proba_forest']*2)+(df['proba_reg']*1)+(df['proba_svc']*1))/4
    df = df.sort_values(['proba_mean'],ascending=False)
    ordered_top_trades = df.drop_duplicates(subset='stock_scale', keep="first")
    ordered_top_trades

    
    #Set probablity threshold to wothin 10%
    ordered_top_trades["threshold"] = np.where(ordered_top_trades["proba_mean"] > th,1,0)
    ordered_top_trades = ordered_top_trades[ordered_top_trades.threshold == 1]
    cols = ['y_test', 'y_forest','y_reg', 'y_svc','proba_mean', 'mark_x', 'strikePrice',
            'underlyingPrice','p_L', 'stock_scale','Percent_profit', 'prophet_yhat', 
            'callPut','margin_required','profit']
    ordered_top_trades = ordered_top_trades[cols] 

    ordered_top_trades = ordered_top_trades.sort_values("margin_required")
    if len(ordered_top_trades) == 0:
        print("please choose a lower theshold!")
        return 
    else:
        margin_required = 0
        total_margin = 0
        trades_taken = pd.DataFrame()
        min_margin = ordered_top_trades['margin_required'].iloc[0]
        while min_margin < (inv*0.5-total_margin):
            for i in range(0,len(ordered_top_trades['margin_required'])):
                if ordered_top_trades['margin_required'].iloc[i] < (inv*0.5-total_margin):
                    trades_taken = trades_taken.append(pd.DataFrame(ordered_top_trades.iloc[i]).transpose())
                    trades_taken = trades_taken.reset_index(drop=True)
                    total_margin += ordered_top_trades['margin_required'].iloc[i]
                if (inv*0.5-total_margin) < min_margin:
                    break

        #trades_taken = trades_taken.groupby('stock_scale').sum()
        trades_taken = trades_taken[['stock_scale','strikePrice','underlyingPrice','profit','margin_required','callPut','proba_mean']]
        trades_taken["callPut"] = np.where(trades_taken.callPut == 0,"Put","Call")
        trades_taken = trades_taken.reset_index(drop=True)
        trades_taken.columns = ['Stock Ticker','Strike','Stock Price','Profit per Contract','Margin Required','Option','Calculated Probability']
        trades_taken['Contracts'] = 1
        trades_taken = trades_taken.groupby(['Stock Ticker','Strike','Stock Price','Profit per Contract','Margin Required','Option','Calculated Probability'])["Contracts"].count().reset_index()
        trades_taken['P/L %'] = round((trades_taken["Profit per Contract"]/trades_taken["Margin Required"])*100)
        trades_taken['Total Profit'] = round(trades_taken['Contracts']*trades_taken['Profit per Contract'],2)
        trades_taken.index=trades_taken.index+1

        summary_dict = {}
        Total_Return = trades_taken['Total Profit'].sum()

        summary_dict.update({'Total Investment' : inv})
        summary_dict.update({'Probability Risk (1:10)' : round(th*10)}) 
        summary_dict.update({'Margin used' : total_margin})
        summary_dict.update({'Cash' : inv-total_margin})
        summary_dict.update({'Total Return' : Total_Return})
        summary_dict.update({'Total Return %': round((Total_Return/inv)*100,1)})
        summary = pd.DataFrame([summary_dict]).transpose()
        summary.columns = ['Summary']


        return trades_taken,th,inv,summary
    
def resulting_investment(trades_taken,th,inv,summary):
    
    if min(trades_taken['Profit per Contract']) < 0:
        col_pal = "coolwarm"
    else:
        col_pal = "crest"
    
    sns.set(style="whitegrid", color_codes=True)
    pal = sns.color_palette(col_pal, len(trades_taken))
    rank = trades_taken['Total Profit'].argsort().argsort()  
    trades_taken["Profit or loss"] = np.where(trades_taken['Total Profit']>1,0,1 )
    plt.figure(figsize=(15,round(len(trades_taken)/2.5)))
    sns.barplot(y="Stock Ticker", x="Total Profit", data=trades_taken, dodge=False,palette=np.array(pal[::-1])[rank])
    plt.show()
    
    rank = trades_taken['P/L %'].argsort().argsort()  
    trades_taken["Profit or loss"] = np.where(trades_taken['Total Profit']>1,0,1 )
    plt.figure(figsize=(15,round(len(trades_taken)/2.5)))
    sns.barplot(y="Stock Ticker", x="P/L %", data=trades_taken, dodge=False,palette=np.array(pal[::-1])[rank])
    plt.show() 
    
    trades_taken = trades_taken.drop(columns="Profit or loss")
    
    display(summary)
    display(trades_taken)
    
    return trades_taken


def user_threshold():
    from IPython.display import clear_output
    while True:
        try:
            th = int(input('Select from 1 - 10 where you would like to set your probability threshold?  '))
            th = int(th)
            
        except ValueError:
            print("Please enter an Integer")
            clear_output(wait=True)
            continue
        else:
            while True:
                try:
                    inv = int(input('How much are you willing to invest?  '))  
                    inv = int(inv)
                except ValueError:
                    print("Please enter an Integer")
                    clear_output(wait=True)
                    continue
                else:
                    th = th/10
                    return th, inv
                    break 

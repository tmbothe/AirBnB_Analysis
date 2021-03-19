import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def read_data():
    """
     INPUT:
       Path - Path were data reside
     OUTPUT:
       dataframe for all necessary files
         - Listings
         - Calendar
         - Reviews
    """
    boston_reviews = pd.read_csv('data/boston_reviews.csv',low_memory =False)
    boston_listings = pd.read_csv('data/boston_listings.csv',low_memory =False)
    boston_calendar = pd.read_csv('data/boston_calendar.csv',low_memory =False)
    
    return (boston_reviews,boston_listings,boston_calendar)


def select_variables(df,target,num_var):
    '''
    INPUT
      df      - original dataframe will all variables
      target  - The target variable
      num_var - Number of numerical variables to select
      drop_var - list of variables to drop
    OUTPUT
      variables to be used in prediction
    '''
    corr_num = df.corr()[target].abs().sort_values(ascending=False)[:num_var]
    new_df = df[corr_num.index]
    return new_df

def create_dummy_df(df, cat_cols,target_col ,dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    
    df = pd.concat([df[cat_cols],df[target_col]], axis=1)
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df

#Clean data putting all together

def clean_data(df, target_col,cat_cols,cols_drops, num_variables=15,cat_variables=20,remove_outliers=False):
    '''
    INPUT
      df - pandas dataframe 
      cols_drops - columns to drop
      target_col - target col
      cat_cols   - categorical columns to include in the model
      num_variables - number of numerical variables to consider
    OUTPUT
       X - A matrix holding all of the variables you want to consider when predicting the response
       y - the corresponding response vector
    
    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
     
    df = df.drop(cols_drops, axis=1)
    
    #dropping all the columns that have more than 50% of missing values
    missing_val_cols = df.columns[df.isnull().mean() > 0.5].tolist()
    df = df.drop(missing_val_cols, axis=1)
    
       
    #clean price
    price_clean   = lambda x: x.replace('$','') if str(x).startswith('$') else x
    is_num_cols = df.columns[df.columns.str.contains('fee|price|deposit|people')]
    num_cols = df[is_num_cols].columns.tolist()
    for col in num_cols:
        if df[col].dtypes=='O':
            df[col] = pd.to_numeric(df[col].apply(price_clean).str.replace(',',''))
    
    #clean rate
    rate_clean = lambda col: str(col).replace("%","") if str(col).endswith('%') else col
    is_rate_cols = df.columns[df.columns.str.contains('rate')]
    rate_cols=df[is_rate_cols].columns.tolist()
    for col in rate_cols:
        df[col] = pd.to_numeric(df[col].apply(rate_clean))
        
    if remove_outliers:
        # computing the quartiles and the interquartile range
        Q1 = np.percentile(df.price,25)
        Q3 = np.percentile(df.price,75)
        IQR = Q3-Q1
        lower_bound = Q1-(IQR*2)
        upper_bound = Q3+(IQR*2)
        df = df[(df[target_col]>=lower_bound) & (df[target_col]<=upper_bound)]
        
    y  = df[target_col]
    
    num_variabledf  = select_variables(df=df,target = target_col,num_var = num_variables)
    #all_num_cols = [x for x in all_num_cols if x!= 'price']
    
    cat_dummy_df= create_dummy_df(df, cat_cols,target_col, dummy_na=False)
    cat_variabledf = select_variables(df=cat_dummy_df,target = target_col,num_var = cat_variables)
    
    df.loc[:,'amenities'] = df['amenities'].str.replace('[{}" ]', '')
    df_amenities = df.amenities.str.get_dummies(sep = ",")
    #amenities_variable_df = select_variables(df=df_amenities,target = target_col,num_var = cat_variables)
    
    #df = df.dropna(subset=[target_col], axis=0)
    X  = pd.concat([num_variabledf.drop(target_col,axis=1),cat_variabledf.drop(target_col,axis=1),df_amenities], axis=1) #df.drop([target_col], axis=1)
        
     #fill mean
    fill_mean = lambda col:col.fillna(col.mean())
    X = X.apply(fill_mean,axis=0)
    
    
    return X, y

def fit_linear_mod(X,y,test_size=0.3, rand_state=42):
    '''
    INPUT:
     X - a dataframe holding all the variables of interest
     y - a string holding the name of the column 
     rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    
    Your function should:
    1. Split your data into an X matrix and a response vector y
    2. Create training and test sets of data
    3. Instantiate a LinearRegression model with normalized data
    4. Fit your model to the training data
    5. Predict the response for the training data and the test data
    6. Obtain an rsquared value for both the training and test data
    '''
  
    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test , y_test_preds, y_train_preds


# this function is used to find the best linear regression model
def find_optimal_lm_mod(df,number_num_variables,number_cat_variables,cat_cols, target_col,test_size=42 , random_state=42):
    '''
    INPUT
    number_num_variables - list of ints, number of numerical vars
    number_cat_variables - list of ints, number of categorical vars
    num_cols - pandas dataframe, numerical variavles
    cat_cols - pandas dataframe, categorical variavles
    target_col - str, column name of the target_col
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    

    OUTPUT
    results - dictionary of r2 scores for different combination of number of numerical and categorical variables
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    best_r2_score_test - float, best r2 score on the test data
    best_num_variables - int, number of numerical variables for best r2 score 
    best_cat_variables - int, number of categorical variables for best r2 score 
    best_lm_model - optimal model object from sklearn
    best_x_train, best_x_test, best_y_train, best_y_test - output from sklearn train test split used for optimal model
    '''
    
    best_r2_score_test, best_num_variables, best_cat_variables, best_lm_model = 0, 0, 0, []
    best_x_train, best_x_test, best_y_train, best_y_test = [], [], [], []
    r2_scores_test, r2_scores_train, results = [], [], dict()
    
    for number in number_num_variables:
        for cat_number in number_cat_variables:
            
            X,y  = clean_data(df= df , target_col=target_col,cat_cols = cat_cols,cols_drops=[], num_variables=number,cat_variables=cat_number,remove_outliers=True)
            
            # split the data into train and test
            x_train, x_test, y_train, y_test = \
            train_test_split(X, y, test_size = test_size, random_state = random_state)
            
            # fit the model and obtain pred target_col
            lm_model = LinearRegression()
            lm_model.fit(x_train, y_train)
            y_test_preds = lm_model.predict(x_test)
            y_train_preds = lm_model.predict(x_train)
            
            # record the best model
            r2_score_test = r2_score(y_test, y_test_preds)
            if r2_score_test > best_r2_score_test:
                best_r2_score_test = r2_score_test
                best_num_variables = number
                best_cat_variables = cat_number
                best_lm_model = lm_model
                best_x_train, best_x_test, best_y_train, best_y_test \
                = x_train, x_test, y_train, y_test
                
            # append the r2 value from the test set
            r2_scores_test.append(r2_score_test)
            r2_scores_train.append(r2_score(y_train, y_train_preds))
            variables = str(number) + ' num_variables,' + str(cat_number) + ' cat_variables'
            results[variables] = r2_score(y_test, y_test_preds)

    return results, r2_scores_test, r2_scores_train, best_r2_score_test, best_num_variables, \
        best_cat_variables, best_lm_model, best_x_train, best_x_test, best_y_train, best_y_test
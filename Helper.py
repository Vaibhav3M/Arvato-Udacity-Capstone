
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings('ignore')




column_transform_dict = {'LNR':'id', 'AGER_TYP':'age_type', 'ALTER_HH':'age_HH', 'ALTERSKATEGORIE_GROB':'age_prename',
                         'ANZ_HAUSHALTE_AKTIV':'count_household_building','ANZ_HH_TITEL':'count_academic_holder_building',
                         'ANZ_PERSONEN':'count_adult_house','ANREDE_KZ':'sex','ANZ_TITEL':'count_professional_HH',
                         'ARBEIT':'share_unemployed_community','BALLRAUM':'distance_metropole','CAMEO_DEUG_2015':'income_class',
                         'CAMEO_DEU_2015':'income_class_detailed','CAMEO_INTL_2015':'international_class',
                         'D19_BANKEN_ANZ_12':'bank_activity_12', 'D19_BANKEN_ANZ_24':'bank_activity_24','D19_BANKEN_DATUM':'bank_last_trans',
                         'D19_BANKEN_DIREKT':'bank_activity_DIRECT_BANKS', 'D19_BANKEN_GROSS':'bank_activity_BIG_BANK', 'D19_BANKEN_LOKAL':'bank_activity_LOCAL_BANK',
                         'D19_BANKEN_OFFLINE_DATUM':'segment_bank_last_trans_offline','D19_BANKEN_ONLINE_DATUM':'segment_bank_last_trans_online',
                         'D19_BANKEN_ONLINE_QUOTE_12':'percent_online_trans_segment_bank','D19_BANKEN_REST':'trans_FURTHER_BANKS',
                         'D19_BEKLEIDUNG_GEH':'trasn_CLOTHING_LUX','D19_BEKLEIDUNG_REST':'trasn_CLOTHING_FURTHER','D19_BILDUNG':'trans_EDUCATION',
                         'D19_BIO_OEKO':'trans_ECOLOGI','D19_BUCH_CD':'trans_BOOKSCD','D19_DIGIT_SERV':'trans_DIGITAL','D19_DROGERIEARTIKEL':'trans_DRUGSTORE',
                         'D19_ENERGIE':'trans_ENERGY','D19_FREIZEIT':'trans_LEISURE','D19_GARTEN':'trans_GARDEN','D19_GESAMT_ANZ_12':'trans_TOTAL_12',
                         'D19_GESAMT_ANZ_24':'trans_TOTAL_24','D19_GESAMT_DATUM':'trans_last_OVERALL','D19_GESAMT_ONLINE_DATUM':'trans_last_ONLINE',
                         'D19_GESAMT_OFFLINE_DATUM':'trans_last_OFFLINE','D19_GESAMT_ONLINE_QUOTE_12':'trans_pre_ONLINE_12','D19_HANDWERK':'TP_DOYOURSELF',
                         'D19_HAUS_DEKO':'TP_DECORATION','D19_KINDERARTIKEL':'TP_CHILD','D19_KONSUMTYP':'consumption_type','D19_KONSUMTYP_MAX':'consumption_type_MAX',
                         'D19_KOSMETIK':'TP_COSMATIC','D19_LEBENSMITTEL':'TP_FOOD','D19_LOTTO':'TP_lotto','D19_NAHRUNGSERGAENZUNG':'TP_DIETARY',
                         'D19_RATGEBER':'TP_GUIDEBOOK','D19_REISEN':'tran_prod_TRAVEL','D19_SAMMELARTIKEL':'TP_COLLLECTABLE','D19_SCHUHE':'TP_SHOES',
                         'D19_SONSTIGE':'TP_OTHER','D19_TECHNIK':'TP_TECH','D19_TELKO_ANZ_12':'TP_TELECOM_12','D19_TELKO_ANZ_24':'TP_TELECOM_24',
                         'D19_TELKO_DATUM':'TP_TELECOM_TOTAL','D19_TELKO_MOBILE':'TP_TELECOM_MOBILE','D19_TELKO_OFFLINE_DATUM':'TP_last_TELECOM_OFFLINE',
                         'D19_TELKO_ONLINE_DATUM':'TP_last_TELECOM_ONLINE','D19_TELKO_ONLINE_QUOTE_12':'TP_last_TELECOM_ONLINE_12',
                         'D19_TELKO_REST':'TP_last_FURTHERMOBILE','D19_TIERARTIKEL':'TP_ANIMAL','D19_VERSAND_ANZ_12':'TP_MAIL_12',
                         'D19_VERSAND_ANZ_24':'TP_MAIL_24', 'D19_VERSAND_DATUM': 'TP_MAIL_TOTAL', 'D19_VERSAND_OFFLINE_DATUM':'TP_MAIL_OFFLINE',
                         'D19_VERSAND_ONLINE_DATUM':'TP_MAIL_ONLINE','D19_VERSAND_ONLINE_QUOTE_12':'TP_prec_MAIL','D19_VERSAND_REST': 'TP_FURTHERMAIL',
                         'D19_VERSICHERUNGEN':'TP_INSURANCE','D19_VERSI_ANZ_12':'TP_INSURANCE_12','D19_VERSI_ANZ_24':'TP_INSURANCE_24',
                         'D19_VERSI_DATUM':'TP_INSURANCE_TOTAL','D19_VERSI_OFFLINE_DATUM':'TP_INSURANCE_OFFLINE','D19_VERSI_ONLINE_DATUM':'TP_INSURANCE_ONLINE',
                         'D19_VERSI_ONLINE_QUOTE_12':'TP_INSURANCE_ONLINE_12','D19_VOLLSORTIMENT':'TP_COMPLETERMAIL','D19_WEIN_FEINKOST':'TP_WINE',
                         'EWDICHTE':'POPULATION_DENSITY_KM','FINANZTYP':'FINANCIAL_TYPE','FINANZ_ANLEGER':'FT_INVESTOR','FINANZ_HAUSBAUER':'FT_OWNHOUSE',
                         'FINANZ_MINIMALIST':'FT_MINIMILIST','FINANZ_SPARER':'FT_SAVER','FINANZ_UNAUFFAELLIGER':'FT_UNREMARK','FINANZ_VORSORGER':'FT_PREPARED',
                         'GEBAEUDETYP':'BUILDING_TYPE','GEBAEUDETYP_RASTER':'INDUSTRIAL_AREA','GEBURTSJAHR':'DOB','GFK_URLAUBERTYP':'VACATION_HABIT',
                         'GREEN_AVANTGARDE':'GREEN_MOVEMENT','HH_EINKOMMEN_SCORE':'HH_NETINCOME_RANGE','INNENSTADT':'distance_center','KBA05_ALTER1':'CAR_SHARE_0031',
                         'KBA05_ALTER2':'CAR_SHARE_3145','KBA05_ALTER3':'CAR_SHARE_4560','KBA05_ALTER4':'CAR_SHARE_61PLUS','KBA05_ANHANG':'TRAILER_SHARE_MC',
                         'KBA05_ANTG1':'COUNT_FAMILYHOUSE_CELL_12','KBA05_ANTG2':'COUNT_FAMILYHOUSE_CELL_35','KBA05_ANTG3':'COUNT_FAMILYHOUSE_CELL_610',
                         'KBA05_ANTG4':'COUNT_FAMILYHOUSE_CELL_10PLUS','KBA05_AUTOQUOT':'CAR_SHARE_HOUSEHOLD','KBA05_BAUMAX':'COMMON_BUILDINGTYPE_IN_CELL',
                         'KBA05_CCM1': 'CAR_SHARE_CMM1','KBA05_CCM2': 'CAR_SHARE_CMM2','KBA05_CCM3': 'CAR_SHARE_CMM3','KBA05_CCM4': 'CAR_SHARE_CMM4',
                         'KBA05_DIESEL':'CAR_DIESEL_SHARE_MC','KBA05_FRAU':'CAR_SHARE_FEMALE','KBA05_GBZ':'COUNT_BUILDING_MC','KBA13_HALTER_20':'CAR_SHARE_PLZ8_21BELOW',
                         'KBA13_HALTER_25':'CAR_SHARE_PLZ8_2125','KBA13_HALTER_30':'CAR_SHARE_PLZ8_2630','KBA13_HALTER_35':'CAR_SHARE_PLZ8_3135',
                         'KBA13_HALTER_40':'CAR_SHARE_PLZ8_3640','KBA13_HALTER_45':'CAR_SHARE_PLZ8_4145','KBA13_HALTER_50':'CAR_SHARE_PLZ8_4650',
                         'KBA13_HALTER_60':'CAR_SHARE_PLZ8_5660','KBA13_HALTER_65':'CAR_SHARE_PLZ8_6165','KBA13_HALTER_66':'CAR_SHARE_PLZ8_66PLUS',
                         'KBA13_HERST_ASIEN':'SHARE_ASIAN_MANUF_PLZ8','KBA13_HERST_AUDI_VW':'SHARE_AUDI_VW_PLZ8','KBA13_HERST_BMW_BENZ':'SHARE_BMW_BENZ_PLZ8',
                         'KBA13_HERST_EUROPA':'SHARE_EUROPE_MANUF_PLZ8','KBA13_HERST_FORD_OPEL':'SHARE_FORD_OPEL_PLZ8','KBA13_HERST_SONST':'SHARE_OTHER_CARMANU_PLZ8',
                         'KBA13_KMH_0_140':'SHARE_CAR_SPEED_0_140','KBA13_KMH_110':'SHARE_CAR_SPEED_110','KBA13_KMH_140':'SHARE_CAR_SPEED_140',
                         'KBA13_KMH_140_210':'SHARE_CAR_SPEED_140_210','KBA13_KMH_180':'SHARE_CAR_SPEED_180','KBA13_KMH_210':'SHARE_CAR_SPEED_210',
                         'KBA13_KMH_211':'SHARE_CAR_SPEED_211','KBA13_KMH_250':'SHARE_CAR_SPEED_250','KBA13_KMH_251':'SHARE_CAR_SPEED_251',

                         }


def plot_comparison_charts(column, df1, df2):
    '''
    Plots 2 charts, one for AZDIAS and other for CUSTOMER dataframes for a column.

    Input:
        column: A column to be plotted
        df1: The AZDIAS dataframe
        df2: The CUSTOMERS dataframe
        '''
    fig, (ax1, ax2) = plt.subplots(figsize=(12,4), ncols=2)
    sns.countplot(x = column, data=df1, ax=ax1, palette="husl")
    ax1.set_xlabel('Value')
    ax1.set_title('Distribution of ' + column + ' in AZDIAS')
    sns.countplot(x = column, data=df2, ax=ax2, palette="husl")
    ax2.set_xlabel('Value')
    ax2.set_title('Distribution of ' + column + ' in CUSTOMER')
    fig.tight_layout()
    plt.show()


def data_pre_process(df, drop=True):
    '''
    This method performs following preprocessing steps on data

    Input:
        df: The dataframe to perform preprocessing steps

    Output:
        df: The preprocessed dataframe
        '''

    labelEncoder = LabelEncoder()
    scaling = StandardScaler()
    imputer = SimpleImputer()


    df['CAMEO_DEU_2015'] = df[['CAMEO_DEU_2015']].fillna(value = '0')
    df['CAMEO_DEU_2015'] = labelEncoder.fit_transform(df['CAMEO_DEU_2015'])

    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].apply(lambda x: 0 if x == 'X' else x)
    df['CAMEO_DEUG_2015'] = df[['CAMEO_DEUG_2015']].fillna(value = 0)
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].apply(lambda x: int(x))

    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].apply(lambda x: 0 if x == 'XX' else x)
    df['CAMEO_INTL_2015'] = df[['CAMEO_INTL_2015']].fillna(value = 0)
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].apply(lambda x: int(x))

    df['D19_LETZTER_KAUF_BRANCHE'] = df[['D19_LETZTER_KAUF_BRANCHE']].fillna(value = '0')
    df['D19_LETZTER_KAUF_BRANCHE'] = labelEncoder.fit_transform(df['D19_LETZTER_KAUF_BRANCHE'])

    df['OST_WEST_KZ'] = df[['OST_WEST_KZ']].fillna(value = 'X')
    df['OST_WEST_KZ'] = labelEncoder.fit_transform(df['OST_WEST_KZ'])

    df['EXTSEL992'] = df[['EXTSEL992']].fillna(value = df['EXTSEL992'].median())


    df['KK_KUNDENTYP'] = df[['KK_KUNDENTYP']].fillna(value = 0.0)

    df['ALTER_KIND1'] = df['ALTER_KIND1'].apply(lambda x: 0 if type(x) != int else x)
    df['ALTER_KIND2'] = df['ALTER_KIND2'].apply(lambda x: 0 if type(x) != int else x)
    df['ALTER_KIND3'] = df['ALTER_KIND3'].apply(lambda x: 0 if type(x) != int else x)
    df['ALTER_KIND4'] = df['ALTER_KIND4'].apply(lambda x: 0 if type(x) != int else x)

    try:
        df['YEAR_ADDED'] = df['EINGEFUEGT_AM'].apply(lambda x: -1 if str(x) == 'nan'
                                                    else datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').year)

        df.drop(columns=['EINGEFUEGT_AM'],inplace=True)
    except:
        pass

    try:
        df.drop(columns=['LNR'],inplace=True)
    except:
        pass

    columns = df.columns
    df = pd.DataFrame(imputer.fit_transform(df),columns = columns)

    if drop:
        df.dropna(inplace = True)


    return df

def data_pre_process_test(df):
    '''
    This method performs following preprocessing steps on test data.

    Input:
        df: The dataframe to perform preprocessing on.

    Output:
        df: The preprocessed dataframe

        '''

    imputer = SimpleImputer()

    df = df.drop(["ALTER_KIND4","ALTER_KIND3","ALTER_KIND2","ALTER_KIND1"],axis = 1)
    df['CAMEO_DEUG_2015']=[-1 if i == "X" else i for i in df['CAMEO_DEUG_2015']]
    df['CAMEO_INTL_2015']= [-1 if i == "XX" else i for i in df['CAMEO_INTL_2015']]
    df["EINGEFUEGT_AM"] = df["EINGEFUEGT_AM"].astype("datetime64")
    df["CAMEO_DEUG_2015"] = df["CAMEO_DEUG_2015"].astype("float64")
    df["CAMEO_INTL_2015"] = df["CAMEO_INTL_2015"].astype("float64")
    df["EINGEFUEGT_AM"] = df["EINGEFUEGT_AM"].apply(lambda x: x.year - 1991)

    df["OST_WEST_KZ"] = df["OST_WEST_KZ"].replace({'W': 1,'O': 2,})


    df['PRAEGENDE_JUGENDJAHR_decade'] =  df['PRAEGENDE_JUGENDJAHRE'].replace({
        1: '1',2: '1',3: '2',4: '2',5: '3',6: '3',7: '3',8: '4',9: '4',10: '5',11: '5',12: '5',13: '5',14: '6',15: '6'})

    df['PRAEGENDE_JUGENDJAHR_movements'] = df['PRAEGENDE_JUGENDJAHRE'].replace({
        1: 2,2: 1,3: 2,4: 1,5: 2,6: 1,7: 1,8: 2,9: 1,10: 2,11: 1,12: 2,13: 1,14: 2,15: 1})

    df['CAMEO_INTL_2015_wealth'] = df['CAMEO_INTL_2015'].replace({
        11: 5,12: 5,13: 5,14: 5,15: 5,21: 4,22: 4,23: 4,24: 4,25: 4,31: 3,32: 3,33: 3,34: 3,35: 3,41: 2,42: 2,43: 2,44: 2,
        45: 2,51: 1,52: 1,53: 1,54: 1,55: 1})

    df['CAMEO_INTL_2015_lifestage'] = df['CAMEO_INTL_2015'].replace({
        11: '1',12: '2',13: '3',14: '4',15: '5',21: '1',22: '2',23: '3',24: '4',25: '5',31: '1',32: '2',33: '3',34: '4',35: '5',
        41: '1',42: '2',43: '3',44: '4',45: '5',51: '1',52: '2',53: '3',54: '4',55: '5'})

    df['WOHNLAGE_rural'] = df['WOHNLAGE'].replace({
        0: 2,1: 2,2: 2,3: 2,4: 2,5: 2,7: 1,8: 1})

    df['WOHNLAGE_neighborhood'] = df['WOHNLAGE'].replace({
        0: 1,1: 6,2: 5,3: 4,4: 3,5: 2,7: 1,8: 1})

    df = df.drop(['PRAEGENDE_JUGENDJAHRE','CAMEO_INTL_2015','WOHNLAGE'],axis = 1)

    df["D19_LETZTER_KAUF_BRANCHE"] = df["D19_LETZTER_KAUF_BRANCHE"].astype("category")
    df["D19_LETZTER_KAUF_BRANCHE"] = df["D19_LETZTER_KAUF_BRANCHE"].cat.codes

    df["CAMEO_DEU_2015"] = df["CAMEO_DEU_2015"].astype("category")
    df["CAMEO_DEU_2015"] = df["CAMEO_DEU_2015"].cat.codes

    #df = df.drop(corr[1],axis = 1)
    columns = df.columns
    df = pd.DataFrame(imputer.fit_transform(df),columns = columns)

    df.dropna(inplace = True)

    return df

def classifier_GS(clf, param_grid, X_train, y_train):
    '''
    Fits a classifier to its training data and prints its ROC AUC score.

    INPUT:
    - clf (classifier): classifier to fit
    - param_grid (dict): classifier parameters used with GridSearchCV
    - X_train (DataFrame): training input
    - y_train (DataFrame): training output

    OUTPUT:
    - classifier: input classifier fitted to the training data
    '''

    # cv uses StratifiedKFold
    # scoring roc_auc available as parameter
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_score_)

    return grid.best_estimator_

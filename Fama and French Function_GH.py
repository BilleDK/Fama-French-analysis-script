# Fama and French - factor model analysis
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import statsmodels.formula.api as sm
import statsmodels.stats.api as sms
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt

# downloads daily factor returns from the ken-french library
# Outputs a full factor analysis by running the function factor_analysis
# ff_model can take 3, 5, 6 referring to fama3, fama5 or fama5 + momentum factor models
# region can take "US", "Europe"
# rolling_coef is set to 90 as default. Indicates the number of historical observations to use in the factor regression


def factor_analysis(ff_model, region, rolling_coef=90):
    price_data = # input your daily price series here in a pd df with date as index
    if ff_model == 3 and region == "US":
        ff_data = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start='1-1-1926')[0]
    elif ff_model == 3 and region == "Europe":
        ff_data = pdr.get_data_famafrench('Europe_3_Factors_Daily', start='1-1-1926')[0]
        ff_data.index = ff_data.index.to_timestamp()
    elif ff_model == 5 and region == "US":
        ff_data = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily', start='1-1-1926')[0]
    elif ff_model == 5 and region == "Europe":
        ff_data = pdr.get_data_famafrench('Europe_5_Factors_Daily', start='1-1-1926')[0]
        ff_data.index = ff_data.index.to_timestamp()
    elif ff_model == 6 and region == "US":
        ff_5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily', start='1-1-1926')[0]
        ff_mom = pdr.get_data_famafrench('F-F_Momentum_Factor_daily', start='1-1-1926')[0]
        ff_data = pd.merge(ff_5, ff_mom, how = 'left', left_index=True, right_index=True)
        ff_data = ff_data.rename(columns={'Mom   ': 'WML'}) # As FF extracted data contains "Mom" col name with 3 spaces at end
    elif ff_model == 6 and region == "Europe":
        ff_5 = pdr.get_data_famafrench('Europe_5_Factors_Daily', start='1-1-1926')[0]
        ff_5.index = ff_5.index.to_timestamp()
        ff_mom = pdr.get_data_famafrench('Europe_Mom_Factor_Daily', start='1-1-1926')[0]
        ff_mom.index = ff_mom.index.to_timestamp()
        ff_data = pd.merge(ff_5, ff_mom, how = 'right', left_index=True, right_index=True) #Right instead of left join as Mom timeseries is shorter
    else:
        print("Wrong model input")
    
    price_data['Return'] = ((price_data['Price'] / price_data['Price'].shift()) - 1) * 100
    data = pd.merge(price_data, ff_data, how = 'left', left_index=True, right_index=True) #joining two df's with price_data being the left table and 'how' specifies a left join
    data['Excess_return'] = data['Return'] - data['RF']
    data = data.dropna() # drop NA's as factor returns is not present at all days where NAV's are present
    data.columns = data.columns.str.replace('-', '_') #replaces Mkt-RF with Mkt_RF as hyphen breaks down statsmodel functions

    # OLS regression over entire time-series
    # intercept is alpha
    if ff_model == 3:
        result = sm.ols(formula = "Excess_return ~ Mkt_RF + SMB + HML", data = data).fit()
        print(result.summary())
    elif ff_model == 5:
        result = sm.ols(formula = "Excess_return ~ Mkt_RF + SMB + HML + RMW + CMA", data = data).fit()
        print(result.summary())
    # Momentum factor is named "Mom" in US dataset and "WML" in EU dataset
    elif ff_model == 6:
        result = sm.ols(formula = "Excess_return ~ Mkt_RF + SMB + HML + RMW + CMA + WML", data = data).fit()
        print(result.summary())
    # p-values at 0,05 indicate that it is statistically significant at confidence levels below 95%
    # params can be called on the regression object after which the coefficients are stored in a dataframe and can be accessed using pandas indexing

    # return decomposition over entire time-series
    Excess_return = (1 + (data['Excess_return'] / 100)).prod() - 1
    Mkt_contribution = result.params['Mkt_RF'] * ((1 + (data['Mkt_RF'] / 100)).prod() - 1)
    Size_contribution = result.params['SMB'] * ((1 + (data['SMB'] / 100)).prod() - 1)
    Value_contribution = result.params['HML'] * ((1 + (data['HML'] / 100)).prod() - 1)
    Value_added = (result.params['Intercept'] / 100) * len(data)
    if ff_model == 3:
        Residual = Excess_return - Mkt_contribution - Size_contribution - Value_contribution - Value_added
    if ff_model == 5:
        RMW_contribution = result.params['RMW'] * ((1 + (data['RMW'] / 100)).prod() - 1)
        CMA_contribution = result.params['CMA'] * ((1 + (data['CMA'] / 100)).prod() - 1)
        Residual = Excess_return - Mkt_contribution - Size_contribution - Value_contribution - Value_added - RMW_contribution - CMA_contribution
    if ff_model == 6:
        RMW_contribution = result.params['RMW'] * ((1 + (data['RMW'] / 100)).prod() - 1)
        CMA_contribution = result.params['CMA'] * ((1 + (data['CMA'] / 100)).prod() - 1)
        Momentum_contribution = result.params['WML'] * ((1 + (data['WML'] / 100)).prod() - 1)
        Residual = Excess_return - Mkt_contribution - Size_contribution - Value_contribution - Value_added - RMW_contribution - CMA_contribution - Momentum_contribution
 
    # Bar chart showing return contribution
    if ff_model == 3:
        labels = ["Market", "Size", "Value", "Alpha", "Residual"]
        Contributions = [Mkt_contribution, Size_contribution, Value_contribution, Value_added, Residual]
        contrib_rounded = list(np.around(np.array(Contributions),2)) # creates a list rounded to 2 decimals using numpy function
        fig_contribution_plot = plt.bar(labels, Contributions)
        # Adding the bar values and aligning with the bar index
        for index, value in enumerate(contrib_rounded):
            plt.text(index, value, str(value))
        plt.title('Fama-French 3 - Cumulative return contribution per factor')
        plt.show()
    if ff_model == 5:
        labels = ["Market", "Size", "Value", "RMW", "CMA", "Alpha", "Residual"]
        Contributions = [Mkt_contribution, Size_contribution, Value_contribution, RMW_contribution, CMA_contribution, Value_added, Residual]
        contrib_rounded = list(np.around(np.array(Contributions),2)) # creates a list rounded to 2 decimals using numpy function
        fig_contribution_plot = plt.bar(labels, Contributions)
        # Adding the bar values and aligning with the bar index
        for index, value in enumerate(contrib_rounded):
            plt.text(index, value, str(value))
        plt.title('Fama-French 5 - Cumulative return contribution per factor')
        plt.show()
    if ff_model == 6:
        labels = ["Market", "Size", "Value", "RMW", "CMA", "WML", "Alpha", "Residual"]
        Contributions = [Mkt_contribution, Size_contribution, Value_contribution, RMW_contribution, CMA_contribution, Momentum_contribution, Value_added, Residual]
        contrib_rounded = list(np.around(np.array(Contributions),2)) # creates a list rounded to 2 decimals using numpy function
        fig_contribution_plot = plt.bar(labels, Contributions)
        # Adding the bar values and aligning with the bar index
        for index, value in enumerate(contrib_rounded):
            plt.text(index, value, str(value))
        plt.title('Fama-French 6 - Cumulative return contribution per factor')
        plt.show()            
    # Why does prod of return column + RF column not equal prod of excess_return column? - Due to compounding effects!

    # Line plot of pure factor returns
    if ff_model == 3:
        plot_data = pd.DataFrame()
        plot_data['Mkt_RF'] = (1 + (data['Mkt_RF'] / 100)).cumprod()
        plot_data['HML'] = (1 + (data['HML'] / 100)).cumprod()
        plot_data['SMB'] = (1 + (data['SMB'] / 100)).cumprod()
        plot_data['Portfolio_return'] = (1 + (data['Return'] / 100)).cumprod()
        plot_data.plot.line(figsize = (24,16), title = 'Fama-French 3 - Cumulative factor returns')
    elif ff_model == 5:
        plot_data = pd.DataFrame()
        plot_data['Mkt_RF'] = (1 + (data['Mkt_RF'] / 100)).cumprod()
        plot_data['HML'] = (1 + (data['HML'] / 100)).cumprod()
        plot_data['SMB'] = (1 + (data['SMB'] / 100)).cumprod()
        plot_data['RMW'] = (1 + (data['RMW'] / 100)).cumprod()
        plot_data['CMA'] = (1 + (data['CMA'] / 100)).cumprod()
        plot_data['Portfolio_return'] = (1 + (data['Return'] / 100)).cumprod()
        plot_data.plot.line(figsize = (24,16), title = 'Fama-French 5 - Cumulative factor returns')       
    elif ff_model == 6:
        plot_data = pd.DataFrame()
        plot_data['Mkt_RF'] = (1 + (data['Mkt_RF'] / 100)).cumprod()
        plot_data['HML'] = (1 + (data['HML'] / 100)).cumprod()
        plot_data['SMB'] = (1 + (data['SMB'] / 100)).cumprod()
        plot_data['RMW'] = (1 + (data['RMW'] / 100)).cumprod()
        plot_data['CMA'] = (1 + (data['CMA'] / 100)).cumprod()
        plot_data['WML'] = (1 + (data['WML'] / 100)).cumprod()        
        plot_data['Portfolio_return'] = (1 + (data['Return'] / 100)).cumprod()
        plot_data.plot.line(figsize = (24,16), title = 'Fama-French 6 - Cumulative factor returns')      
          
    # Sub plot with both factor return and coefficient
    if ff_model == 3:
        mod = RollingOLS.from_formula('Excess_return ~ Mkt_RF + SMB + HML', data = data, window= rolling_coef).fit()
        params = mod.params.copy() # needs to copy in order to use the data in a df
        fig, axes = plt.subplots(2, 2, figsize=(24,12)) #creates 1 figure with 2 axes in a 2x2 matrix, each axes can be chosen through indexing
        plot_data['Mkt_RF'].plot(ax=axes[0, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['Mkt_RF'].plot.line(ax=axes[0, 0], title = "Market", legend=True, label="Rolling factor coefficient")
        plot_data['SMB'].plot(ax=axes[0, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['SMB'].plot.line(ax=axes[0, 1], title = "Size / SMB", legend=True, label="Rolling factor coefficient")
        plot_data['HML'].plot(ax=axes[1, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['HML'].plot.line(ax=axes[1, 0], title = "Value / HML", legend=True, label="Rolling factor coefficient")
        fig.suptitle("Rolling factor coefficent & cumulative factor return")
        fig.tight_layout() # adds spacing between suptitle and graphs
        axes[1][1].set_visible(False) # makes axes at index 1,1 invinsible
    if ff_model == 5:
        mod = RollingOLS.from_formula('Excess_return ~ Mkt_RF + SMB + HML + RMW  + CMA', data = data, window= rolling_coef).fit()
        params = mod.params.copy() # needs to copy in order to use the data in a df
        fig, axes = plt.subplots(3, 2, figsize=(24,12)) #creates 1 figure with 6 axes, in a 3x2 matrix, each axes can be chosen through indexing
        plot_data['Mkt_RF'].plot(ax=axes[0, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['Mkt_RF'].plot.line(ax=axes[0, 0], title = "Market", legend=True, label="Rolling factor coefficient")        
        plot_data['SMB'].plot(ax=axes[0, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['SMB'].plot.line(ax=axes[0, 1], title = "Size / SMB", legend=True, label="Rolling factor coefficient")
        plot_data['HML'].plot(ax=axes[1, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['HML'].plot.line(ax=axes[1, 0], title = "Value / HML", legend=True, label="Rolling factor coefficient")
        plot_data['RMW'].plot(ax=axes[1, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['RMW'].plot.line(ax=axes[1, 1], title = "Profitability / RMW", legend=True, label="Rolling factor coefficient")
        plot_data['CMA'].plot(ax=axes[2, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['CMA'].plot.line(ax=axes[2,0], title = "Investments / CMA", legend=True, label="Rolling factor coefficient")
        fig.suptitle("Rolling factor coefficent & cumulative factor return")
        fig.tight_layout() # adds spacing between suptitle and graphs
        axes[2][1].set_visible(False) # makes axes at index 2,1 invinsible
    if ff_model == 6:
        mod = RollingOLS.from_formula('Excess_return ~ Mkt_RF + SMB + HML + RMW  + CMA + WML', data = data, window= rolling_coef).fit()
        params = mod.params.copy() # needs to copy in order to use the data in a df
        fig, axes = plt.subplots(3, 2, figsize=(24,12)) #creates 1 figure with 6 axes, in a 3x2 matrix, each axes can be chosen through indexing
        plot_data['Mkt_RF'].plot(ax=axes[0, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['Mkt_RF'].plot.line(ax=axes[0, 0], title = "Market", legend=True, label="Rolling factor coefficient")     
        plot_data['SMB'].plot(ax=axes[0, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['SMB'].plot.line(ax=axes[0, 1], title = "Size / SMB", legend=True, label="Rolling factor coefficient")
        plot_data['HML'].plot(ax=axes[1, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['HML'].plot.line(ax=axes[1, 0], title = "Value / HML", legend=True, label="Rolling factor coefficient")
        plot_data['RMW'].plot(ax=axes[1, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['RMW'].plot.line(ax=axes[1, 1], title = "Profitability / RMW", legend=True, label="Rolling factor coefficient")
        plot_data['CMA'].plot(ax=axes[2, 0], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['CMA'].plot.line(ax=axes[2,0], title = "Investments / CMA", legend=True, label="Rolling factor coefficient")
        plot_data['WML'].plot(ax=axes[2, 1], secondary_y=True, colormap='PRGn', legend=True, label="Cum. factor return")
        params['WML'].plot.line(ax=axes[2, 1], title = "Momentum / WML", legend=True, label="Rolling factor coefficient")
        fig.suptitle("Rolling factor coefficent & cumulative factor return")   
        fig.tight_layout() # adds spacing between suptitle and graphs
    
    print("Regression F-test is significant at a", f"{(1 - result.f_pvalue):.0%}", "confidence level")
    print("SMB factor is significant at a", f"{(1 - result.pvalues['SMB']):.0%}", "confidence level")
    print("HML factor is significant at a", f"{(1 - result.pvalues['HML']):.0%}", "confidence level")
    if ff_model == 5:
        print("RMW factor is significant at a", f"{(1 - result.pvalues['RMW']):.0%}", "confidence level")
        print("CMA factor is significant at a", f"{(1 - result.pvalues['CMA']):.0%}", "confidence level")
    if ff_model == 6:
        print("RMW factor is significant at a", f"{(1 - result.pvalues['RMW']):.0%}", "confidence level")
        print("CMA factor is significant at a", f"{(1 - result.pvalues['CMA']):.0%}", "confidence level")
        print("WML factor is significant at a", f"{(1 - result.pvalues['WML']):.0%}", "confidence level")
    print("R-squared:", round(result.rsquared, 2), "meaning that", round(result.rsquared, 2), "of the total risk is related to the systematic factors in the model")
    print("implying that", 1 - round(result.rsquared, 2), "of total risk is related to idiosyncratic/non-systematic risk")
    # Not possible to extract Durbin Watson value directly from the summary, so durbin_watson function from statsmodels needed. Critical values are not available.
    dw = (round(durbin_watson(result.resid), 2))
    if dw < 2.25 and dw > 1.75:
        print("Durbin Watson value is:", dw, "which is between 1.75 and 2.25, assuming no autocorrelation")
    else:
        print("Durbin Watson value is:", dw, "which is either greater than 2.25 or less than 1.75, assuming autocorrelation")

    if result.rsquared > 0.7 and result.f_pvalue > 0.05 and result.pvalues.mean() > 0.075:
        print("Based on R-squared, F-statics and t-values, multicollinarity might be present")
    else:
        print("Based on R-squared, F-statics and t-values, multicollinarity seems to not be present")
    
    heteroskadicity = sms.het_breuschpagan(result.resid, result.model.exog)
    if heteroskadicity[3] > 0.05:
        print("Heteroskedasticity is not present based on the Breusch-Pagan test at a 95% confidence level")
    else:
        print("Heteroskedasticity is present based on the Breusch-Pagan test at a 95% confidence level")
    # null hypothesis is that variance of error terms, conditional on explanatory variables is constant. Meaning variance of errors does not change when return changes.
    return print("Analysis done")

factor_analysis(6, "Europe", 60)

# Fama-French-analysis-script
Input your price series and receive a full factor analysis based on fama french

The price data from your security or portfolio you input must be stored in a pandas DataFrame format with prices in the first column and dates in the index column. The dates must be in a datetime format as the index is being used to join the df later on.

The analysis will only yield viable results for single stock or an equity portfolio price series.

The script will automatically download daily factors from the ken-french library and the three factor models to choose from is Fama-French 3, Fama-French 5 or Fama-French5 + Momentum. The model specification needs to be done when running the function in the script. Region can be set to US or Europe depending on the primary exposure of the input prices.

The output is a full analysis including:
Regression output showing exposures to each factor across the entire time period, including the statistical significance of the exposure to each factor.
Signals on whether any regression conditions have been breached.
The strength of the factor model including exposure to other factors than the ones specified in the model.
A bar chart showing the return contribution from each factor, including alpha and return contributions from factors outside of the model.
A line graph showing the cumulative factor returns across the full time period
A multigraph showing the cumulative factor returns including the rolling factor exposure based on the user input for look-back period for the rolling regression. It defaults to 90 days/observations. This can be used to graph how the portfolio exposure moved with the factor return, as a decreasing factor exposure benefits from a decreasing cumulative factor return. In contrast, if a factor have positive returns, the portfolio benefits from increasing exposure to the factor.

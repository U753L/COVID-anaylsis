# notice that the new cases/new deaths graph is kind of awkward because of just how many data points there are.

import matplotlib.pyplot as plt # To actually graph the Data
import matplotlib as mpl
import datetime # To handle the dates
import matplotlib.dates as mdates # To handle the dates
import pandas as pd # To efficiently read the csv file and essentially make my code much, much shorter
from sklearn.linear_model import LinearRegression
import numpy
# Making the subplots
fig, axs = plt.subplots(2,2)
# Making the whole graph appear full screen
plt.tight_layout()
variable = plt.get_current_fig_manager()
variable.window.state('zoomed')
# Reading the CSV file and making all of the dates into a datetime object
covid_data = pd.read_csv('___covid19data_.csv')
covid_data['Date_reported'] = covid_data['Date_reported']
covid_data['Date_reported'] = covid_data['Date_reported'].apply(datetime.datetime.fromisoformat)
# Making the dictionaries for each country
covid_dataIN = covid_data[covid_data['Country_code']=='IN']
covid_dataBR = covid_data[covid_data['Country_code']=='BR']
covid_dataGB = covid_data[covid_data['Country_code']=='GB']
covid_dataUS = covid_data[covid_data['Country_code']=='US']
covid_dataFR = covid_data[covid_data['Country_code']=='FR']


# predictions
k = int(input("how many days from now would you like to predict? "))
# cases
modelUS_TC=LinearRegression()
modelUS_TC.fit(numpy.array([x for x in range(1, covid_dataUS['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataUS['Cumulative_cases'].values.reshape(-1,1))
print(int(modelUS_TC.predict([[k+covid_dataUS['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total cases for the United States after', k, 'days')
modelBR_TC=LinearRegression()
modelBR_TC.fit(numpy.array([x for x in range(1, covid_dataBR['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataBR['Cumulative_cases'].values.reshape(-1,1))
print(int(modelBR_TC.predict([[k+covid_dataBR['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total cases for Brazil after', k, 'days')
modelGB_TC=LinearRegression()
modelGB_TC.fit(numpy.array([x for x in range(1, covid_dataGB['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataGB['Cumulative_cases'].values.reshape(-1,1))
print(int(modelGB_TC.predict([[k+covid_dataGB['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total cases for Great Britain after', k, 'days')
modelIN_TC=LinearRegression()
modelIN_TC.fit(numpy.array([x for x in range(1, covid_dataIN['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataIN['Cumulative_cases'].values.reshape(-1,1))
print(int(modelIN_TC.predict([[k+covid_dataIN['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total cases for India after', k, 'days')
modelFR_TC=LinearRegression()
modelFR_TC.fit(numpy.array([x for x in range(1, covid_dataFR['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataFR['Cumulative_cases'].values.reshape(-1,1))
print(int(modelFR_TC.predict([[k+covid_dataFR['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total cases for France after', k, 'days')

# deaths
modelUS_TD=LinearRegression()
modelUS_TD.fit(numpy.array([x for x in range(1, covid_dataUS['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataUS['Cumulative_deaths'].values.reshape(-1,1))
print(int(modelUS_TD.predict([[k+covid_dataUS['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total deaths for the United States after', k, 'days')
modelBR_TD=LinearRegression()
modelBR_TD.fit(numpy.array([x for x in range(1, covid_dataBR['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataBR['Cumulative_deaths'].values.reshape(-1,1))
print(int(modelBR_TD.predict([[k+covid_dataBR['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total deaths for Brazil after', k, 'days')
modelGB_TD=LinearRegression()
modelGB_TD.fit(numpy.array([x for x in range(1, covid_dataGB['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataGB['Cumulative_deaths'].values.reshape(-1,1))
print(int(modelGB_TD.predict([[k+covid_dataGB['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total deaths for Great Britain after', k, 'days')
modelIN_TD=LinearRegression()
modelIN_TD.fit(numpy.array([x for x in range(1, covid_dataIN['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataIN['Cumulative_deaths'].values.reshape(-1,1))
print(int(modelIN_TD.predict([[k+covid_dataIN['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total deaths for India after', k, 'days')
modelFR_TD=LinearRegression()
modelFR_TD.fit(numpy.array([x for x in range(1, covid_dataFR['Date_reported'].shape[0]+1)]).reshape(-1,1), covid_dataFR['Cumulative_deaths'].values.reshape(-1,1))
print(int(modelFR_TD.predict([[k+covid_dataFR['Date_reported'].shape[0]]])[0][0]), 'is the predicted number of total deaths for France after', k, 'days')


# Plotting New Deaths
axs[1,1].plot(covid_dataIN['Date_reported'], covid_dataIN['New_deaths'], label='India', color='red')
axs[1,1].plot(covid_dataBR['Date_reported'], covid_dataBR['New_deaths'], label='Brazil', color='blue')
axs[1,1].plot(covid_dataGB['Date_reported'], covid_dataGB['New_deaths'], label='Great Britain', color='yellow')
axs[1,1].plot(covid_dataUS['Date_reported'], covid_dataUS['New_deaths'], label='United States', color='green')
axs[1,1].plot(covid_dataFR['Date_reported'], covid_dataFR['New_deaths'], label='France', color='black')
axs[1,1].set_title("New Deaths")
axs[1,1].get_xaxis().set_major_locator(mdates.MonthLocator(interval=2))
axs[1,1].get_xaxis().set_major_formatter(mdates.DateFormatter('%b'))
# Plotting New Cases
axs[0,1].plot(covid_dataIN['Date_reported'], covid_dataIN['New_cases'], label='India', color='red')
axs[0,1].plot(covid_dataBR['Date_reported'], covid_dataBR['New_cases'], label='Brazil', color='blue')
axs[0,1].plot(covid_dataGB['Date_reported'], covid_dataGB['New_cases'], label='Great Britain', color='yellow')
axs[0,1].plot(covid_dataUS['Date_reported'], covid_dataUS['New_cases'], label='United States', color='green')
axs[0,1].plot(covid_dataFR['Date_reported'], covid_dataFR['New_cases'], label='France', color='black')
axs[0,1].set_title("New Cases")
axs[0,1].get_xaxis().set_major_locator(mdates.MonthLocator(interval=2))
axs[0,1].get_xaxis().set_major_formatter(mdates.DateFormatter('%b'))
# Plotting Total Cases
axs[0,0].plot(covid_dataIN['Date_reported'], covid_dataIN['Cumulative_cases'], label='India', color='red')
axs[0,0].plot(covid_dataBR['Date_reported'], covid_dataBR['Cumulative_cases'], label='Brazil', color='blue')
axs[0,0].plot(covid_dataGB['Date_reported'], covid_dataGB['Cumulative_cases'], label='Great Britain', color='yellow')
axs[0,0].plot(covid_dataUS['Date_reported'], covid_dataUS['Cumulative_cases'], label='United States', color='green')
axs[0,0].plot(covid_dataFR['Date_reported'], covid_dataFR['Cumulative_cases'], label='France', color='black')
axs[0,0].set_title("Total Cases")
axs[0,0].get_xaxis().set_major_locator(mdates.MonthLocator(interval=2))
axs[0,0].get_xaxis().set_major_formatter(mdates.DateFormatter('%b'))
# Plotting Total Deaths
axs[1,0].plot(covid_dataIN['Date_reported'], covid_dataIN['Cumulative_deaths'], label='India', color='red')
axs[1,0].plot(covid_dataBR['Date_reported'], covid_dataBR['Cumulative_deaths'], label='Brazil', color='blue')
axs[1,0].plot(covid_dataGB['Date_reported'], covid_dataGB['Cumulative_deaths'], label='Great Britain', color='yellow')
axs[1,0].plot(covid_dataUS['Date_reported'], covid_dataUS['Cumulative_deaths'], label='United States', color='green')
axs[1,0].plot(covid_dataFR['Date_reported'], covid_dataFR['Cumulative_deaths'], label='France', color='black')
axs[1,0].set_title("Total Deaths")
axs[1,0].get_xaxis().set_major_locator(mdates.MonthLocator(interval=2))
axs[1,0].get_xaxis().set_major_formatter(mdates.DateFormatter('%b'))

# Making the legends
axs[0,0].legend()
axs[1,0].legend()
axs[0,1].legend()
axs[1,1].legend()
# Showing it all with a . . .
plt.show()

# end

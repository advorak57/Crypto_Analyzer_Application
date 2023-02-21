# Initial imports
import requests
import pandas as pd
from dotenv import load_dotenv
import os
from pycoingecko import CoinGeckoAPI
import datetime as dt
from pytrends.request import TrendReq
import holoviews as hv
hv.extension('bokeh')
import hvplot.pandas
import numpy as np
import matplotlib.pyplot as plt


crypto_name = input("Please enter the full name of the cryptocurrency you want to search: ").lower()
country_name = input("What country do you live in? Please use your country's initials for your answer.")
trader_input = input("Are you a day trader? (yes/no): ")
if trader_input.lower() == "yes":
    time_frame = "3 months"
else:
    long_term_input = input("Are you a long-term investor? (yes/no): ")
    if long_term_input.lower() == "yes":
        time_frame = "3 years"
    else:
        time_frame = "1 year"
        print("You must be somewhere in between!")

print("You have selected", crypto_name, "and based on your investing preferences we have selected to use a time frame of", time_frame)

import time

def simulate_thinking():
    print("Thinking...", end="")
    for i in range(3):
        time.sleep(0.8)
        print(".", end="")
    print("\n")
    final_statement = "We have taken into account your preferences and will now provide you with the historical data you requested and perform several calculations for your pleasure."
    words = final_statement.split(" ")
    for word in words:
        time.sleep(0.1)
        print(word, end=" ")

# Run Thinking...
simulate_thinking()

import pandas as pd

# Get the number of days based on the time frame
if time_frame == '3 months':
    days = 90
elif time_frame == '3 years':
    days = 1095
else:
    days = 365

# Create a CoinGeckoAPI object
cg = CoinGeckoAPI()

# Get the current price of the selected cryptocurrency in USD
data = cg.get_coin_market_chart_by_id(id=crypto_name, vs_currency='usd',days= days)

if data:
    prices = data['prices']
    volumes = data['total_volumes']
    market_caps = data['market_caps']

    dates = []
    close_prices = []
    volumes = []
    market_caps = []
    
    for d in range(len(prices)):
        timestamp = prices[d][0] / 1000
        date = dt.datetime.fromtimestamp(timestamp).date()
        close_price = prices[d][1]
        volume = data['total_volumes'][d][1]
        market_cap = data['market_caps'][d][1]

        dates.append(date)
        close_prices.append(close_price)
        volumes.append(volume)
        market_caps.append(market_cap)

# Convert the market_info list of lists to a pandas dataframe
    marketData_df = pd.DataFrame({'Date': dates, 'Close Price': close_prices, 'Volume': volumes, 'Market Cap': market_caps})
    marketData_df['Date'] = pd.to_datetime(marketData_df['Date'])
    marketData_df.set_index('Date', inplace=True)
    #df = df.resample('D').mean()

# Resample the data to daily frequency and keep only the last value of the day
    marketData_df = marketData_df.resample('D').last()

    print(marketData_df)
else:
    print(f'No Market data found for {crypto_name}')

# Round data
clean_data_df = marketData_df.apply(lambda x: round(x, 2))
# Display the dataframe
clean_data_df

# Narrow down DataFrame to get price action
price_action_df = clean_data_df.drop(columns = ["Volume", "Market Cap"])

# Plot Price Action

# price_action_df_plot = price_action_df.hvplot(kind= 'line', 
#                                                         width=1200, 
#                                                         height=600, 
#                                                         rot=90, 
#                                                         hover_color='orange').opts(
#     xlabel='Date',
#     ylabel='Price',
#     title='Daily Price Action for' + " " + crypto_name.capitalize(),
# )
# # Create and Save Plot.png
# # hvplot.save(price_action_df_plot, "Price_Action_Plot.png")
# display(price_action_df_plot)

# Plot Price Action
price_action_df.plot(kind='line', y='Close Price', title='Daily Price Action for' + " " + crypto_name.capitalize())
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

monte_carlo_price_df = marketData_df.drop(columns = ['Volume', 'Market Cap'])

import pandas as pd
import numpy as np
import hvplot.pandas
from IPython.display import display

# Calculate the daily returns
returns = monte_carlo_price_df['Close Price'].pct_change()

# Define the number of simulations and the number of days to simulate
num_simulations = 1000
num_days = 365

# Create a list to store the results of each simulation
results = []

# Loop through the number of simulations
for i in range(num_simulations):
    
    # Generate a random walk for the daily returns
    sim_returns = np.random.normal(returns.mean(), returns.std(), num_days)
    
    # Calculate the simulated price for each day
    sim_price = [monte_carlo_price_df['Close Price'][-1]]
    for j in range(num_days):
        sim_price.append(sim_price[-1] * (1 + sim_returns[j]))
    
    # Store the results of the simulation
    results.append(sim_price[-1])

# Convert the results list to a pandas DataFrame
results_df = pd.DataFrame(results, columns=['final_price'])


# Plot the distribution of the final simulated prices
results_df['final_price'].plot.hist(bins=100, edgecolor = 'grey', color='cadetblue', title = crypto_name.capitalize() + " " + "Simulated Future Price Results")
plt.xlabel=('final_price')
plt.ylabel=('Frequency')
plt.show()

print(f"Here are the results of a 1000 simulations run to predict the future price of" + " " + crypto_name.capitalize() + " " + "in 1 years time, " 
      f"extrapolating from the previous" + " " + time_frame + " " + "of market data.") 
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


# Import required libraries for news API Request
import requests
import os
from pycoingecko import CoinGeckoAPI
import pandas as pd
from dotenv import load_dotenv
from pytrends.request import TrendReq
import hvplot.pandas
import time
import aylien_news_api
from aylien_news_api.rest import ApiException
from aylienapiclient import textapi
from pprint import pprint
import os
from dotenv import load_dotenv
from textblob import TextBlob
import datetime
import json
import csv
import matplotlib.pyplot as plt
import datetime as dt
import holoviews as hv
import panel as pn
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# Check .env is being called
load_dotenv()



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







api_id = os.getenv('X-AYLIEN-NewsAPI-Application-ID')
api_key = os.getenv('X-AYLIEN-NewsAPI-Application-Key')

configuration = aylien_news_api.Configuration()
configuration.api_key['X-AYLIEN-NewsAPI-Application-ID'] = api_id
configuration.api_key['X-AYLIEN-NewsAPI-Application-Key'] = api_key

# Set Client variable
client = aylien_news_api.ApiClient(configuration)
api_instance = aylien_news_api.DefaultApi(client)


import time

def simulate_thinking():
    print("Reading articles...", end="")
    for i in range(3):
        time.sleep(0.8)
        print(".", end="")
    print("\n")
    final_statement = """Give me a few seconds to read every single article on the internet looking for information on your crypto.   I have now read articles from
US JP GB CN JP DE IN GB FR IT CA KR RU BR AU ES MX ID NL SA TR CH PL SE BE TH IE IL AR VE NO AT NG ZA BD AE EG DK SG PH MY HK VN IR PK CL CO FI RO CZ PT NZ annnnnnnnd 3 2 1   DONE"""
    words = final_statement.split(" ")
    for word in words:
        time.sleep(0.1)
        print(word, end=" ")

# Run Thinking...
simulate_thinking()

Countries =['US','JP','GB','FR','IT']
field = 'cryptocurrency'
for x in Countries:
    try:
    ## Make a call to the Stories endpoint for stories that meet the criteria of the search operators
        api_response = api_instance.list_stories(       
            title= '"bitcoin" OR "ethereum"',
            source_locations_country= Countries,
            # body= '',
            language= ['en'],
            published_at_start= 'NOW-30DAYS',
            published_at_end= 'NOW',
            per_page= 40,
            sort_by= 'relevance'
        )
## Print the returned story

        # pprint(api_response)
    except ApiException as e:
        print('Exception when calling DefaultApi->list_stories: %s\n' % e)  


stories = api_response.stories
data = []
for story in stories:
    published_at = story.published_at
    headline = story.title
    body_text = story.body
    combined_text = headline + " " + body_text
    polarity = TextBlob(combined_text).polarity
    subjectivity = TextBlob(combined_text).subjectivity
#     here i am removing the headline and body text so that i can plot the overall polarity and subjectivity. I will be adding this back in later for headline analysis
# 'Headline': headline, 'BodyText': body_text,
    data.append({'PublishDate': published_at, 'Polarity': polarity, 'Subjectivity': subjectivity})
    Sentiment_df = pd.DataFrame(data)



# Create the data frame from the collected data
Sentiment_df = pd.DataFrame(data)  
Sentiment_df = Sentiment_df.set_index(pd.DatetimeIndex(Sentiment_df['PublishDate']))
Sentiment_df.drop(['PublishDate'], axis=1, inplace= True)
Sentiment_df['Polarity'] = Sentiment_df['Polarity'].astype(float)
Sentiment_df['Subjectivity'] = Sentiment_df['Subjectivity'].astype(float)

print(Sentiment_df.dtypes)
Sentiment_df



quadrant_1 = sum(np.logical_and(Sentiment_df['Polarity'] < 0, Sentiment_df['Subjectivity'] < 0))
# display(quadrant_1)

quadrant_2 = sum(np.logical_and(Sentiment_df['Polarity'] > 0, Sentiment_df['Subjectivity'] < 0))
# display(quadrant_2)

quadrant_3 = sum(np.logical_and(Sentiment_df['Polarity'] < 0, Sentiment_df['Subjectivity'] > 0))
# display(quadrant_3)

quadrant_4 = sum(np.logical_and(Sentiment_df['Polarity'] > 0, Sentiment_df['Subjectivity'] > 0))
# display(quadrant_4)


fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axhline(y=0, color="black", linestyle="--")
plt.axvline(x=0, color="black", linestyle="--")

plt.plot(Sentiment_df['Polarity'],Sentiment_df['Subjectivity'],"o")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().axes.get_xaxis().set_visible(True)
plt.gca().axes.get_yaxis().set_visible(True)


plt.text(0,-0.95,'Subjectivity',horizontalalignment='center', verticalalignment='center')
plt.text(-0.95,0,'Polarity', horizontalalignment='center', verticalalignment='center', rotation=90)

plt.text(0,1,'Polarity(Positivity vs Negativity) vs Subjectivity(How much opinion is a factor)',horizontalalignment='center', verticalalignment='center')
plt.text(0.4,0.4,'Count: ' + str(quadrant_4),horizontalalignment='center', verticalalignment='center')
plt.text(-0.4,0.4,'Count: ' + str(quadrant_3), horizontalalignment='center', verticalalignment='center')
plt.text(-0.4,-0.4,'Count: ' + str(quadrant_1),horizontalalignment='center', verticalalignment='center')
plt.text(0.4,-0.4,'Count: ' + str(quadrant_2), horizontalalignment='center', verticalalignment='center')




plt.xlim([-0.8, 0.8])
plt.ylim([-0.8, 0.8])
plt.show()


import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
# url = "https://www.coindesk.com/markets/2023/02/23/bitcoin-hovers-near-24k-as-investors-mull-economic-uncertainties/"

url = input("Have you any specific articles you would like analyzed : ").lower()

import time
def simulate_thinking():
    print("Analyzing your article...", end="")
    for i in range(3):
        time.sleep(0.8)
        print(".", end="")
    print("\n")
    final_statement = """Stand by, article almost fully analyzed. 3 2 1   DONE"""
    words = final_statement.split(" ")
    for word in words:
        time.sleep(0.1)
        print(word, end=" ")

# Run Thinking...
simulate_thinking()
response = requests.get(url)
html = response.text
soup = BeautifulSoup(html, "html.parser")
text = ""
for p in soup.find_all("p"):
    text += p.get_text()
UserArticle = TextBlob(text)
print(UserArticle)

# print(UserArticle)
SentimentBlob = TextBlob(text)
ArticlePolarity = SentimentBlob.sentiment.polarity
ArticleSubjectivity = SentimentBlob.sentiment.subjectivity
# print(ArticlePolarity)
# print(ArticleSubjectivity)
ArticleData= {"Polarity Score":[ArticlePolarity],
              "Subjectivity Score":[ArticleSubjectivity]}
YourArticle_df = pd.DataFrame(ArticleData)
YourArticle_df

quadrant_1 = sum(np.logical_and(YourArticle_df['Polarity Score'] < 0, YourArticle_df['Subjectivity Score'] < 0))
# display(quadrant_1)

quadrant_2 = sum(np.logical_and(YourArticle_df['Polarity Score'] > 0, YourArticle_df['Subjectivity Score'] < 0))
# display(quadrant_2)

quadrant_3 = sum(np.logical_and(YourArticle_df['Polarity Score'] < 0, YourArticle_df['Subjectivity Score'] > 0))
# display(quadrant_3)

quadrant_4 = sum(np.logical_and(YourArticle_df['Polarity Score'] > 0, YourArticle_df['Subjectivity Score'] > 0))
# display(quadrant_4)

fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axhline(y=0, color="black", linestyle="--")
plt.axvline(x=0, color="black", linestyle="--")

plt.plot(YourArticle_df['Polarity Score'],YourArticle_df['Subjectivity Score'],"o")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().axes.get_xaxis().set_visible(True)
plt.gca().axes.get_yaxis().set_visible(True)


plt.text(0,-0.95,'Subjectivity Score',horizontalalignment='center', verticalalignment='center')
plt.text(-0.95,0,'Polarity Score', horizontalalignment='center', verticalalignment='center', rotation=90)

plt.text(0,1,'Your Articles Polarity(Positivity vs Negativity) vs Subjectivity(How much opinion is a factor)',horizontalalignment='center', verticalalignment='center')
plt.text(0.4,0.4,'Count: ' + str(quadrant_4),horizontalalignment='center', verticalalignment='center')
plt.text(-0.4,0.4,'Count: ' + str(quadrant_3), horizontalalignment='center', verticalalignment='center')
plt.text(-0.4,-0.4,'Count: ' + str(quadrant_1),horizontalalignment='center', verticalalignment='center')
plt.text(0.4,-0.4,'Count: ' + str(quadrant_2), horizontalalignment='center', verticalalignment='center')

plt.xlim([-0.8, 0.8])
plt.ylim([-0.8, 0.8])
plt.show()


import time
def simulate_thinking():
    print("Analysis complete...", end="")
    for i in range(3):
        time.sleep(0.8)
        print(".", end="")
    print("\n")
    final_statement = """I hope you enjoyed my analysis. Invoice for my services will be sending in 3 2 1 Sent!
Please check the spam folder in your email."""
    words = final_statement.split(" ")
    for word in words:
        time.sleep(0.2)
        print(word, end=" ")

# Run Thinking...
simulate_thinking()


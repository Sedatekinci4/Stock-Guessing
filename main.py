import yfinance as yf
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as yft
import numpy as np
import datetime as dt
import time as tm

# giving the start and end dates
startDate = '2023-05-01'
endDate = '2023-09-19'
ticker = 'THYAO.IS'


# ticker = yf.Ticker('THYAO.IS').info
# ticker01 = yf.Ticker('GOOGL').info
# marketPrice = ticker['regularMarketOpen']
# marketPrice01 = ticker01['regularMarketOpen' ]
# previousClosePrice = ticker['regularMarketPreviousClose']
# previousClosePrice01 = ticker01['regularMarketPreviousClose']

# downloading the data of the ticker value between
# the start and end dates
resultData = yf.download(ticker, startDate, endDate)

# print('Ticker Value: THYAO')
# print('Market Price Value:', marketPrice)
# print('Previous Close Price Value:', previousClosePrice)

print(resultData)


init_df = yft.get_data(
    'THYAO.IS',
    start_date=startDate,
    end_date=endDate,
    interval='1d')

# remove columns which our neural network will not use
init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
# create the column 'date' based on index column
init_df['date'] = init_df.index

print(init_df)


# Let's preliminary see our data on the graphic
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(init_df['close'][-200:])
plt.xlabel("days")
plt.ylabel("price")
plt.legend([f'Actual price for {ticker}'])
plt.show()

# Scale data for ML engine
# scaler = MinMaxScaler()
# init_df['close'] = scaler.fit_transform(np.expand_dims(init_df['close'].values, axis=1))

# print('Ticker Value: GOOGL')
# print('Market Price Value:', marketPrice01)
# print('Previous Close Price Value:', previousClosePrice01)


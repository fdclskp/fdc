#README
#銘柄ごとにインジケーターを計算し、特徴量として追加する
#追加された特徴量を使って、企業間の相関をエッジの重みとした完全グラフを生成
#追加する特徴量
#SMA,EMA,SAR,BBANDS,MACD,ROC,RSI,ADX,ADOSC,CCI,ATR,STOCHRSI,STOCHF,AROON,OBV,MFI.VOLUME 多すぎ？
#BBANDS, SAR,MACD,RSI,STOCHRSI,VOLUME 
import pandas as pd
import pathlib
import os

def SMA(close, period):
	MA_period = period
	return close.rolling(MA_period).mean()

def EMA(close, period):
	EMA_period = period
	return close.ewm(span=EMA_period).mean()

def SMMA(close, period):
	SMMA_period = period
	return close.ewm(alpha = 1 / SMMA_period).mean()

def MOMENTUM(close, period):
	Mom_period = period
	shift = close.shift(Mom_period)
	return close / shift * 100

def BBAND(close,period, Deviation):
	Bands_period = 10
	Deviation = 2
	Base = close.rolling(Bands_period).mean()
	sigma = close.rolling(Bands_period).std(ddof=0)
	Upper = Base+sigma*Deviation
	Lower = Base-sigma*Deviation
	return Upper, Lower

def MACD(close, FastEMA_period, SlowEMA_priod, SignalSMA_period):
	MACD = close.ewm(span=FastEMA_period).mean() - close.ewm(span=SignalSMA_period).mean()
	Signal = MACD.rolling(SignalSMA_period).mean()
	return MACD, Signal
	
def RSI(close, period):
	RSI_period = 14
	diff = close.diff(1)
	positive = diff.clip_lower(0).ewm(alpha=1/RSI_period).mean()
	negative = diff.clip_upper(0).ewm(alpha=1/RSI_period).mean()
	return 100-100/(1-positive/negative)	

def STOCHASTICS(close, Kperiod, Dperiod, Slowing):	
	Kperiod = 14 #%K期間
	Dperiod = 3  #%D期間
	Slowing = 3  #平滑化期間
	Hline = high.rolling(Kperiod).max()
	Lline = low.rolling(Kperiod).min()
	sumlow = (close-Lline).rolling(Slowing).sum()
	sumhigh = (Hline-Lline).rolling(Slowing).sum()
	Stoch = sumlow/sumhigh*100
	Signal = Stoch.rolling(Dperiod).mean()
	return Stoch, Signal

def PSAR(close, high, low, iaf,maxaf):
	length = len(close)
	psar = close[0:len(close)]
	psarbull = [None]* length
	psarbear = [None]* length
	bull = True
	af = iaf
	ep = low[0]
	hp = high[0]
	lp = low[0]

	for i in range(2,length):
		if bull:
			psar[i] = psar[i-1] + af * (hp - psar[i-1])
		else:
			psar[i] = psar[i-1] + af * (lp - psar[i-1])

		reverse = False

		if bull:
			if low[i] < psar[i]:
				bull = False
				reverse = True
				psar[i] = hp
				lp = low[i]
				af = iaf
		else:
			if high[i] > psar[i]:
				bull = True
				reverse = True
				psar[i] = lp
				hp = high[i]
				af = iaf

		if not reverse:	
			if bull:
				if high[i] > hp:
					hp = high[i]
					af = min(af + iaf, maxaf)
				if low[i-1] < psar[i]:
					psar[i] = low[i-1]
				if low[i-2] < psar[i]:
					psar[i] = low[i-2]
			else:
				if low[i] < lp:
					lp = low[i]
					af = min(af + iaf, maxaf)
				if high[i-1] < psar[i]:
					psar[i] = high[i-1]
				if high[i-2] < psar[i]:
					psar[i] = high[i-2]
		if bull:
			psarbull[i] = psar[i]
		else:
			psarbear[i] = psar[i]

	return psar,psarbull,psarbear
				
#years = ['2016', '2017', '2018']
years = ['2018']
period = "/daily/"

for y in years:
	STOCK_DATA = "../stock/stock_2016-2018/split_by_company"+ y + period
	for d in os.listdir(STOCK_DATA):
		filename = STOCK_DATA + d
		df = pd.read_csv(filename, index_col=0,parse_dates=True)
		high = df['HIGH']
		low = df['LOW']
		close = df['close']
	
		sma = SMA(close, 10)
		df['SMA'] = sma
		ema = EMA(close, 10)
		df['EMA'] = ema
		smma = SMMA(close,10)
		df['SMMA'] = smma
		mom = MOMENTUM(close, 10)
		df['MOMENTUM'] = mom
		bband_upper, bband_lower =BBAND(close,10,2)
		df['BBAND_UPPER'] = bband_upper
		df['BBAND_LOWER'] = bband_lower
		macd,macd_signal = MACD(close,12,26,9)
		df['MACD'] = macd
		df['MACD_SIGNAL'] = macd_signal
		rsi = RSI(close,14)
		df['RSI'] = rsi
		stoch, stochas_signal = STOCHASTICS(close, 14, 3, 3)
		df['STOCH'] = stoch
		df['STOCHAS_SIGNAL'] = stochas_signal
		psar, psarbull, psarbear = PSAR(close, high, low,0.02,0.2)
		df['PSARBULL'] = psarbull
		df['PSARBEAR'] = psarbear
		
		#read_file = open("stock/stock_2016-2018/split_by_company"+"2016"+"/daily/" + d)
		#num_line = sum([1 for i in open(read_file,'r')])
		#f = open(read_file)	
		#row_data = f.readline() + features_col
		WRITTEN_DIR = "stock_2016-2018/stock_indicator" + y + period
		if (not os.path.isfile(WRITTEN_DIR)):
			pathlib.Path(WRITTEN_DIR).mkdir(exist_ok=True)
		df.to_csv(WRITTEN_DIR+d+".csv")
		#
		#if (not os.path.isfile(written_file)) :
		#	with open(written_file, mode = 'w') as w:
		#		w.write(row_data)
		#line = f.readline() + features
		#with open(written_file, mode = 'w') as w:
		#	w.write(line)
		#f.close()

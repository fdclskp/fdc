#coding: UTF-8 -*-
#----------------------------------------------------
#README
#stock_year.csvから企業（銘柄コード）ごとにcsvに変換するコード
import pathlib
import os
years = ['2016', '2017', '2018']


for y in years:
	filename = "stock_" + y + ".csv"
	f = open(filename)
	lines = f.readlines()
	row_data = lines[0]
	lines = lines[1:]
	f.close()
	for line in lines:
		tmp = line.split(",")
		stock_data = "split_by_company" + y + "/" + tmp[1] + ".csv"
		if (not os.path.isfile(stock_data)) :
			with open(stock_data, mode = 'w') as d:
				d.write(row_data)
				d.write(line)
		else:
			with open(stock_data, mode = 'a') as d:
				d.write(line)

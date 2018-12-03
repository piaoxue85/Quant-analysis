from datetime import datetime

ts_start = datetime(2010,1,1,0,0,0)
ts_end =datetime(2018,1,1,0,0,0)
test_ratio = 0.1
cols = ['OpenPrice', 'HighPrice', 'LowPrice','TurnoverVolume', 'TurnoverValue','PCT']
sequence_length = 50
normalize = True
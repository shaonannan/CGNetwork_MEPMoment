import os
try:
	cmd = 'python RunVoltageMoment.py -stimulus 1'
	os.system(cmd)
except Exception as e:
	print('Something Gonna Wrong',e)
    
'''
cmd = 'python RunVoltageMoment.py -stimulus 0'
cmd = 'python RunVoltageMoment.py -stimulus 1 -see 0.8 -sei -0.6'
'''
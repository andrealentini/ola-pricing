# ola-pricing
Online Learning Applications Course Project - Pricing &amp; Social Influence
## Students
- Antonio Ercolani
- Francesco Romanò
- Andrea Lentini
- Ledio Sheshori

## Run
- Step 1:

		python3 greedy_test.py

- Steps 2, 3, 4 and 7:
	
		python3 simulator_tester.py
	 Change the parameters of 'Simulator' (line 136 of simulator_tester.py), as commented in the code,  to: 
	- Enable uncertainty in units of items sold
	- Enable Context generation 
	- Choose between the complete/approximated pull arm algorithm in the bandit
	
	Instantiate the bandit as *UCB_Learner(prices)* or *TS_Learner(prices)* in line 92 of simulator_tester.py to choose the bandit type 

- Step 5:
	
		python3 simulator_tester_step5.py
		
	Instantiate the bandit as *UCB_Learner(prices)* or *TS_Learner(prices)* in line 108 of simulator_tester_step5.py to choose the bandit type
	
- Step 6:
	
		python3 simulator_tester_step6.py
		
	Instantiate the bandit as *CDUCB_Learner(prices)* (change detection UCB) or *SWUCB_Learner(prices)* (sliding window UCB) in line 130 of simulator_tester_step6.py to choose the bandit type

	

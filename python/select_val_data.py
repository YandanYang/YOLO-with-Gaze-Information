import random
f = open("/home/gazetracker/otherSw/darknet/data/kitchen/kitchen_pupil.txt")
val= open('/home/gazetracker/otherSw/darknet/data/kitchen/pupil_val.txt', 'a') 
train= open('/home/gazetracker/otherSw/darknet/data/kitchen/pupil_train.txt', 'a')   
for line in f:
	c=random.randint(1,25)
	if c==1 :	
		val.write(line)
	else:
		train.write(line)




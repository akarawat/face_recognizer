from datetime import datetime
import time
now = datetime.now()

i = 0
while i < 10:
    i+=1
    time.sleep(1)
    stampTime = datetime.now()
    print(datetime.now())


# returns (minutes, seconds)
diff = stampTime - now
print(now)
print(stampTime)
print(diff)
print(diff.total_seconds())
minutes = divmod(diff.total_seconds(), 60) 
print('Total difference in minutes: ', minutes[0], 'minutes', minutes[1], 'seconds')


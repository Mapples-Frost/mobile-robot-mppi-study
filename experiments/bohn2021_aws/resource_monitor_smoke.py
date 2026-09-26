import time
end=time.monotonic()+4
while time.monotonic()<end:
    sum(i*i for i in range(1000))
print('resource-monitor integration smoke complete; no training/test data accessed')

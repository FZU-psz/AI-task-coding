```python
lst = [i*10 for i in range(1,7)]
lst
```




    [10, 20, 30, 40, 50, 60]




```python
lst=[i**2 for i in range(1,8)]
lst
```




    [1, 4, 9, 16, 25, 36, 49]




```python
lst= [i*10  if i%2==1 else -i*10 for i in range(1,7)]
lst
```




    [10, -20, 30, -40, 50, -60]




```python
lst = [i*10 for i in range(1,7) if i%2==1]
lst
```




    [10, 30, 50]




```python
lst=[[str(i)+'*'+str(j)+"="+str(i*j) for i in range(1,j+1)] for j in range(1,10)]
lst
```




    [['1*1=1'],
     ['1*2=2', '2*2=4'],
     ['1*3=3', '2*3=6', '3*3=9'],
     ['1*4=4', '2*4=8', '3*4=12', '4*4=16'],
     ['1*5=5', '2*5=10', '3*5=15', '4*5=20', '5*5=25'],
     ['1*6=6', '2*6=12', '3*6=18', '4*6=24', '5*6=30', '6*6=36'],
     ['1*7=7', '2*7=14', '3*7=21', '4*7=28', '5*7=35', '6*7=42', '7*7=49'],
     ['1*8=8',
      '2*8=16',
      '3*8=24',
      '4*8=32',
      '5*8=40',
      '6*8=48',
      '7*8=56',
      '8*8=64'],
     ['1*9=9',
      '2*9=18',
      '3*9=27',
      '4*9=36',
      '5*9=45',
      '6*9=54',
      '7*9=63',
      '8*9=72',
      '9*9=81']]




```python
class Timer : #计时器
    import time
    def __init__(self):
        self.lst=[]
        self.startime=0
    def start(self):
        import time
        self.startime= time.time()
    def stop (self):
        import time
        self.lst.append(time.time()-self.startime)
        return self.lst[-1]
    def avg (self):
        return self.lst.sum()/len(lst)
    def sum(self):
        return self.lst.sum()
    def cumsum (self): #
        import time
        return time.time()-self.starttime

timer=Timer()
timer.start()
for i in range (10000000): pass
print('{:.5f}sec'.format(timer.stop()))
```

    0.21226sec
    


```python
class Accumulator:  #@save   #定义一个累加器类
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
```


```python
def libs(n):
    a = 0
    b = 1
    while True:
        a, b = b, a + b
        if a > n:
            return
        yield a
for each in libs(100): # 0-100的斐波那契数列
    print(each, end=' ')
```

    1 1 2 3 5 8 13 21 34 55 89 


```python
def fib():
    a=0
    b=1
    while True:
        a,b=b,a+b;
        yield a
iterator=fib();
for i in range(11):
    print(next(iterator),end=" ")
```

    1 1 2 3 5 8 13 21 34 55 89 


```python

```

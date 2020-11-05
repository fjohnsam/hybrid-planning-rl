from multiprocessing import Process


def f1(a):
	print("inside f1 a=",c)

def f2(b):
	print("Inside f2 b=",c)

if __name__ == '__main__':
	print("starting")
	a=1
	b=2
	c=3
	p1=Process(target=f1,args=(a,))
	p2=Process(target=f2,args=(b,))
	p2.start()
	p1.start()
	p1.join()
	p2.join()

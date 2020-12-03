# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:33:00 2020

@author: Marcos
"""
from multiprocessing import Process, Lock, Pool, Manager
import random, time

CONST = 100

"""
Pool:    
When you have junk of data, you can use Pool class.
Only the process under execution are kept in the memory.
I/O operation: It waits till the I/O operation is completed & does not schedule another process. This might increase the execution time.
Uses FIFO scheduler.

Process:    
When you have a small data or functions and less repetitive tasks to do.
It puts all the process in the memory. Hence in the larger task, it might cause to loss of memory.
I/O operation: The process class suspends the process executing I/O operations and schedule another process parallel.
Uses FIFO schedul
"""

#%% 
# This creates ONE process for each job created. If running thousands or 
# millions of jobs, that can incur in an OS problem, because all processes
# are loaded into memory. Fasters, but unsitable for very big jobs.

def worker1(lock, i):
    lock.acquire()
    with open('test.txt', 'a') as f:
        f.write(f'hello world {i}\n')
    lock.release()
    time.sleep(random.random())
    
def test1():
    print('Started!')
    
    lock = Lock()
    
    start = time.time()
    with open('test.txt', 'w') as f:
        f.write('Initializing\n')
    
    procs = [Process(target=worker1, args=(lock, num)) for num in range(CONST)]
    
    for p in procs:
        p.start()
    
    for p in procs:
        p.join() #to make sure to wait until all processes are done.
        
    print(f'Done! {time.time()-start} seconds elapsed.')

#%%
# This evaluates all the iterator across all processes, saves them into a list
# (in memory) and iterates over the list. Slowest, but reliable.


def worker2(i):
    time.sleep(random.random())
    return i
    
    
def test2():
    
    start = time.time()
    with Pool(4) as pool:
        with open('test.txt', 'w') as f:
            f.write('Initializing\n')
            for result in pool.imap(worker2, range(CONST)):
                # print(f'writing line at {time.time()-start:.4f}')
                f.write(f'hello {result}\n')
    
    print(f'Done! {time.time()-start} seconds elapsed.')
    
#%%
# Has a pool of workers doing the work, same as the las one. But isntead of 
# dumping everything into a list and then writing, each process dumps into a 
# queue and an extra process reads the queue and writes. Intermediate fast, 
# safe in terms of memory.

from multiprocessing import cpu_count

def worker3(arg, q):
    '''stupidly simulates long running process'''
    time.sleep(random.random())
    q.put(arg)
    return arg

def listener(q):
    '''listens for messages on the q, writes to file. '''

    with open('test.txt', 'w') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m) + '\n')
            # print('wrote!')
            f.flush() #to make sure buffer is empty

def test3():
    start = time.time()
    
    #must use Manager queue here, or will not work
    manager = Manager()
    q = manager.Queue()    
    pool = Pool(cpu_count() + 2)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    
    print(f'Started doing jobs at {time.time()-start}')
    jobs = []
    for i in range(CONST):
        job = pool.apply_async(worker3, (i, q))
        jobs.append(job)
        
    print(f'Finished doing jobs at {time.time()-start}')
    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()
        
    print(f'Finished getting jobs at {time.time()-start}')

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()
    
    print(f'Exiting {time.time()-start}')


#%%
# A combination of cases 1 and 3, where each worker itself writes to file, but
# the workers come from a pool, they are not all created at the start, freeing
# some memory. Slightly faster than case 3.



def worker4(i, j):
    lock.acquire()
    with open('test.txt', 'a') as f:
        f.write(f'hello world {i=}, {j=}\n')
    # global count
    # count+=1
    # print(f'Run #{count}')
        
    lock.release()
    time.sleep(random.random())


# using global Lock is faster than using a Manager(), because that take sup a process
# https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes
def init(l):
    global lock
    # global count
    lock = l
    # count = 0
    

def test4():
    start = time.time()
      
            
    # lock = Manager().Lock()
    l = Lock()
    
    #fire off workers
    with Pool(cpu_count() + 2, initializer=init, initargs=(l,)) as pool:
    
        start = time.time()
        with open('test.txt', 'w') as f:
            f.write('Initializing\n')
            
        
        print(f'Started doing jobs at {time.time()-start}')
        jobs = []
        for i in range(CONST):
            job = pool.apply_async(worker4, (i,i*i) )
            jobs.append(job)
            
        print(f'Finished doing jobs at {time.time()-start}')
        
        for job in jobs: 
            job.get() #wait for all jobs to be done (blocks)
            
        print(f'Finished getting jobs at {time.time()-start}')
    
        
    print(f'Exiting {time.time()-start}')



#%%
if __name__ == '__main__':
    s = time.time()
    test4()
    print(time.time()-s)
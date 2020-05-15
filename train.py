from time import time, sleep
from train_input_args import train_input_args
from train_classifier import classifier

def main():
    start_time = time()
    
    in_arg = train_input_args()
    classifier(in_arg)
    end_time = time()
    
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)))

if __name__ == "__main__":
    main()
    
    

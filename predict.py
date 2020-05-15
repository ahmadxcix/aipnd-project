from time import time, sleep
from predict_input_args import predict_input_args
from predict_classifier import classifier
def main():
    start_time = time()
    
    in_arg = predict_input_args()
    classifier(in_arg)
    end_time = time()
    
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)))

if __name__ == "__main__":
    main()
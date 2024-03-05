import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf) 

class Data_Pre:
    # ------데이터 load-------
    def data_load():
        input_arr=np.load(f"/home/aims/obb_contents/annotation/circle_gt/numpy/input_arr_1.npy")
        input_label=np.load(f"/home/aims/obb_contents/annotation/circle_gt/numpy/input_label_1.npy")
        num_video=2 
        last_video=29
        while True:
            temp_arr=np.load(f"/home/aims/obb_contents/annotation/circle_gt/numpy/input_arr_{num_video}.npy")
            temp_label=np.load(f"/home/aims/obb_contents/annotation/circle_gt/numpy/input_label_{num_video}.npy")
            input_arr=np.concatenate((input_arr,temp_arr),axis=0)
            input_label=np.concatenate((input_label,temp_label),axis=0)
            if num_video==last_video:
                break
            else:
                num_video+=1
        # input_arr = input_arr[:, :, :-1]
        shape=input_arr.shape
        input_arr=input_arr.reshape((shape[0],30,6))
        return input_arr, input_label

    def min_max_normalize(data):
        print(len(data))
        shape=data.shape
        normalized_data=np.zeros((0,shape[1],shape[2]))
        min_val=999
        max_val=0
        for i in range(len(data)):
            for j in range(6):
                min_vals=np.min(data[i],axis=0)
                max_vals=np.max(data[i],axis=0)
        print(min_vals)
        print(max_vals)

        # for i in range(len(data)):
        #     for j in range(6):    
        #         if j!=2 and j!=5:
        #             temp_data=(data[i][:,j]-min_val)/(max_val-min_val)
        #             normalized_data 
    # # Transpose the normalized_data to get the original structure of the data
    # normalized_data = list(map(list, zip(*normalized_data)))
    
    # return normalized_data

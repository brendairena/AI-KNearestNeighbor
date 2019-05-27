import pandas as pd
import numpy as np
import math
import operator

#read training data from csv file and store into array
def read_data_train():
    read_data = pd.read_csv('data\[2019] DataTrain Tugas 2 AI.csv.csv')
    return (np.array(read_data))

#read testing data from csv file and store into array
def read_data_test():
    read_data = pd.read_csv('data\[2019] DataTest Tugas 2 AI.csv')
    return (np.array(read_data))

#calculate euclidean distance of 2 data
def euclidean_distances(item1, item2, length):
    distance = 0
    for x in range(length):
        distance += pow((item1[x] - item2[x]),2)
    return math.sqrt(distance)

#find neighbors related to test item with amount of neighbors depending on k value
def get_neighbors(training_set,test_item, k):
    distances = []
    length = len(test_item)-1
    for x in range(len(training_set)):
        euclide = euclidean_distances(test_item,training_set[x],length)
        distances.append((training_set[x],euclide))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

#find the maximum class value appears
def calculate_class(neighbors):
    sum_votes = {}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in sum_votes:
            sum_votes[response]+=1
        else:
            sum_votes[response]=1
    sorted_votes = sorted(sum_votes.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_votes[0][0]

#calculate accuracy on sample testing set
def calculate_accuracy(data_validation, prediction):
    correct = 0
    for x in range(len(data_validation)):
        if data_validation[x][-1] == prediction[x]:         
            correct += 1                                    
    return (correct / float(len(data_validation))) * 100.0  

#save array data to csv file
def save_to_file(array_data):
    np.savetxt\
        ('File Prediksi_Tugas2AI_1301164641.csv', array_data, fmt='%.3f', delimiter=',',
         header="atribut 1, atribut 2, atribut 3, atribut 4, kelas")

#find the best k value, and pass k value to main program
def best_k_value(data):
    accuracy = []
    prediction = []
    test_validation = data[:100]        
    training_validation = data[100:]    

    #check all condition for eack k neighbor
    for k in range(3,13):                                                         
        del prediction[:]                                                        
        for x in range(len(test_validation)):
            neighbors = get_neighbors(training_validation, test_validation[x],k) 
            test_class = calculate_class(neighbors)                              
            prediction.append(test_class)
        rate=calculate_accuracy(test_validation,prediction)                      
        accuracy.append((rate,k))                                                
        print('> K =',k,'--> Accuracy =',format(rate,'.2f'),'%')
    accuracy.sort(key=operator.itemgetter(0), reverse=True)                       
    k=accuracy[0][1]                                                             
    print('  !! Maximum accuracy is reached by using K =',k,'accuracy =',np.amax(accuracy),'% !!  ')
    return k    

#main program
def main():
    result = []                        
    training_set = read_data_train()   
    testing_set = read_data_test()     

    print('--------------------------------------------------------------------------') 
    print('                   TUGAS PROGRAM 2 KECERDASAN BUATAN                      ')
    print('                 BRENDA IRENA - IFIK 40 03 - 1301164641                   ')
    print('--------------------------------------------------------------------------')
    
    print('         *Finding the best K neighbor with the highest accuracy*          ')
    k = best_k_value(training_set)  
    print('               *Next operation is against real test data*                 ')
    print('                             PLEASE WAIT ...                              ')

   
    for x in range(len(testing_set)):                             
        neighbors =get_neighbors(training_set,testing_set[x],k)    
        test_class= calculate_class(neighbors)                     
        testing_set[x][4] = test_class                              
        result.append(testing_set[x])                              
    save_to_file(result)                                         
    print('Process is DONE, check on your file -> File Prediksi_Tugas2AI_1301164641.csv')

main()

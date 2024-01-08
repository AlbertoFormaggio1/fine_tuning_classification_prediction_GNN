import json
import os
import sys

# num -> finds best "num" runs
# print_output -> if true print it on terminal
# save_output -> if true save output in a txt file
def find_best_params(dataset_name, net_name, results_dict, params_dict, num : int, print_output : bool = False, save_output : bool = False, file_name : str = ""):

    val_accuracies = []
    for k, v in results_dict.items():
        for i in range(len(v)):    # finde tuple associated to classification 2b
            if v[i][0] == "results_class2b":
                for j in range(len(v[i][1])):  # find list associated to validation accuracy
                    if v[i][1][j][0] == "val_acc":
                        val_accuracies.append([k, v[i][1][j][1][-1], []])

    sorted_accuracies = sorted(val_accuracies, key=lambda x: -x[1])

    for i in range(len(sorted_accuracies)): 
        key = sorted_accuracies[i][0]
        sorted_accuracies[i][2] = params_dict[key]
    
    if num > len(sorted_accuracies):
        num = len(sorted_accuracies)
    
    if save_output:
        with open(file_name, "w") as file:
            file.write("PARAMETERS OF THE " + str(num) + " BEST RUNS IN " + dataset_name + " with " + net_name + ":\n\n")
            for j in range(num):

                endline1 = "" if j == num-1 else "\n"

                net = sorted_accuracies[j][0].split("||")[0]

                file.write(str(j+1) + ") val_acc = " + str(sorted_accuracies[j][1]) + "\n")
                for i in range(len(sorted_accuracies[0][2])):
                    name = sorted_accuracies[j][2][i][0]
                    val = sorted_accuracies[j][2][i][1]

                    endline2 = "" if i == len(sorted_accuracies[0][2])-1 else "\n"
                    file.write(name + " : " + str(val) + endline2)
                file.write(endline1 + endline1)

    if print_output:
        print("PARAMETERS OF THE " + str(num) + " BEST RUNS IN " + dataset_name + " with " + net_name + ":\n\n")
        for j in range(num):

            net = sorted_accuracies[j][0].split("||")[0]

            print(str(j+1) + ") val_acc = " + str(sorted_accuracies[j][1]))
            for i in range(len(sorted_accuracies[0][2])):
                name = sorted_accuracies[j][2][i][0]
                val = sorted_accuracies[j][2][i][1]
                print(name + " : " + str(val))
            print()

    # for i in range(len(sorted_accuracies)):
    #     print(str(i) + ": " + str(sorted_accuracies[i][1]))

if __name__ == "__main__":

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "cora"
    net_name = sys.argv[2] if len(sys.argv) > 1 else "GCN"

    results_file = os.path.join(dataset_name + "_" + net_name + "_results.json")
    if(os.path.exists(results_file)):
        with open(results_file) as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    params_file = os.path.join(dataset_name + "_" + net_name + "_params.json")
    if(os.path.exists(params_file)):
        with open(params_file) as f:
            params_dict = json.load(f)
    else:
        params_dict = {}

    find_best_params(dataset_name, net_name, results_dict, params_dict, 3, print_output=True)
    # find_best_params(dataset_name, net_name, results_dict, params_dict, 5, print_output=False, save_output=True, file_name="cora_GCN_best_runs.txt")
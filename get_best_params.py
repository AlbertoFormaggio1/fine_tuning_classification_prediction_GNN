import json
import os
import sys

# num -> finds best "num" runs
# print_output -> if true print it on terminal
# save_output -> if true save output in a txt file
def find_best_params(dataset_name, net_name, results_dict, params_dict, num : int, print_output : bool = False, save_output : bool = False, file_name : str = ""):

    val_accuracies = []
    for k, v in results_dict.items():

        class2b_found = False
        for i in range(len(v)):    # find tuple associated to classification 2b (if done, i.e. freeze < 1.0, otherwise classification 2a)
            if v[i][0] == "results_class2b":
                if len(v[i][1]) > 0:
                    class2b_found = True
                for j in range(len(v[i][1])):  # find list associated to validation accuracy
                    if v[i][1][j][0] == "val_acc":
                        val_accuracies.append([k, v[i][1][j][1][-1], []])
        
        if not class2b_found:
            for i in range(len(v)):    # finde tuple associated to classification 2b
                if v[i][0] == "results_class2a":
                    for j in range(len(v[i][1])):  # find list associated to validation accuracy
                        if v[i][1][j][0] == "val_acc":
                            val_accuracies.append([k, v[i][1][j][1][-1], []])

    sorted_accuracies = sorted(val_accuracies, key=lambda x: -x[1]) # sort by val_acc descending
    
    for i in range(len(sorted_accuracies)): 
        key = sorted_accuracies[i][0] # key identifying the run
        sorted_accuracies[i][2] = params_dict[key] # get params of the run
    
    if num > len(sorted_accuracies):
        num = len(sorted_accuracies)
    
    if save_output:
        with open(file_name, "w") as file:
            file.write("PARAMETERS OF THE " + str(num) + " BEST RUNS IN " + dataset_name + " with " + net_name + ":\n\n")
            for j in range(num):

                endline1 = "" if j == num-1 else "\n"

                # net = sorted_accuracies[j][0].split("||")[0]

                file.write(str(j+1) + ") val_acc = " + str(sorted_accuracies[j][1]) + "\n")
                for i in range(len(sorted_accuracies[j][2])):
                    name = sorted_accuracies[j][2][i][0]    # name of the param
                    val = sorted_accuracies[j][2][i][1]     # val of the param
                    
                    endline2 = "" if i == len(sorted_accuracies[0][2])-1 else "\n"
                    file.write(name + " : " + str(val) + endline2)
                file.write(endline1 + endline1)

    if print_output:
        print("PARAMETERS OF THE " + str(num) + " BEST RUNS IN " + dataset_name + " with " + net_name + ":\n\n")
        for j in range(num):

            net = sorted_accuracies[j][0].split("||")[0]

            print(str(j+1) + ") val_acc = " + str(sorted_accuracies[j][1]))
            for i in range(len(sorted_accuracies[0][2])):
                name = sorted_accuracies[j][2][i][0]    # name of the param
                val = sorted_accuracies[j][2][i][1]     # val of the param
                print(name + " : " + str(val))
            print()

    return sorted_accuracies

    # for i in range(len(sorted_accuracies)):
    #     print(str(i) + ": " + str(sorted_accuracies[i][1]))


def count_params_in_best_runs(sorted_accuracies, num_best_runs, filepath):

    if num_best_runs > len(sorted_accuracies):
        num_best_runs = len(sorted_accuracies)

    params_counter = {}   # k = (dropout, 0.4), V = num
    for j in range(num_best_runs):
        for i in range(len(sorted_accuracies[0][2])):   # elem of sorted_accuracies = [run_key, run_final_val_acc, [params_list]]
            name = sorted_accuracies[j][2][i][0]
            val = sorted_accuracies[j][2][i][1]
            if isinstance(val, list) and len(val) == 1:     # manage params with list value (e.g., ml hidden sizes)
                val = int(val[0])
            # if isinstance(val, list):
            #     val = '-'.join([str(v) for v in val])

            if (name, val) in params_counter.keys():
                params_counter[(name, val)] += 1
            else:
                params_counter[(name, val)] = 1
    
    with open(filepath, "w") as file:

        tolist = list(params_counter.items())
        tolist = sorted(tolist, key=lambda x: (x[0][0], x[0][1]))

        prev_param = tolist[0][0][0]
        j = 1
        file.write(str(j) + ") " + prev_param + "\n")

        for i in range(len(tolist)):

            if tolist[i][0][0] != prev_param:
                j += 1
                prev_param = tolist[i][0][0]
                file.write("\n" + str(j) + ") " + prev_param + "\n")

            file.write(str(tolist[i][0][1]) + "  -->  " + str(tolist[i][1]) + "\n")

    


if __name__ == "__main__":

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "cora"
    net_name = sys.argv[2] if len(sys.argv) > 1 else "GAT"

    out_dir = dataset_name + "_" + net_name
    os.makedirs(out_dir, exist_ok=True)
    
    results_file = os.path.join(out_dir, dataset_name + "_" + net_name + "_results.json")
    if(os.path.exists(results_file)):
        with open(results_file) as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    params_file = os.path.join(out_dir, dataset_name + "_" + net_name + "_params.json")
    if(os.path.exists(params_file)):
        with open(params_file) as f:
            params_dict = json.load(f)
    else:
        params_dict = {}

    out_filename = dataset_name + "_" + net_name + "_best_runs.txt"
    filepath = os.path.join(out_dir, out_filename)
    num_best_runs = 20
    
    sorted_accuracies = find_best_params(dataset_name, net_name, results_dict, params_dict, num_best_runs, print_output=False, save_output=True, file_name=filepath)
    # find_best_params(dataset_name, net_name, results_dict, params_dict, 5, print_output=False, save_output=True, file_name="cora_GCN_best_runs.txt")

    out_filename = dataset_name + "_" + net_name + "_params_counter.txt"
    filepath = os.path.join(out_dir, out_filename)
    count_params_in_best_runs(sorted_accuracies, num_best_runs, filepath)
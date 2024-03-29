from thesis_common import *
from world_modeller_dataset import *
from datasets.trainingset__5000_200 import training_scenarios
# from trainingset_5000_1000 import *

from datasets.trainingset__5000_200_ltd import training_scenarios_ltd, log_file_name
import matplotlib.pyplot as plt
import scipy.stats
import random

plt.rcParams['figure.dpi'] = 125 # increase plot sizes!

input_size = 22
hidden_size = 11
hidden_lyr_num = 2
output_size = 6

inf_subjects = []
no_inf_subjects = []
number_of_subjects = 32
epochs = 1
lr = 0.03
activation = torch.sigmoid
# n_train = 1000

log_file = open(log_file_name, 'w')
train_scenarios_set = [(training_scenarios, 'Training scenarios extended', [500, 1000]), (training_scenarios_ltd, 'Training scenarios ltd', [100, 200])]

for ts in train_scenarios_set:
    log_file.write(ts[1] + '\n')
    for n_train in ts[2]:
        preds = np.zeros((number_of_subjects, output_size))
        surprises = []
        for i in range(number_of_subjects):
            (w1, w2) = initialize_random_subject(input_size, hidden_size, output_size)

            print("************************ Subject {} **********************************".format(i))

            (epoch, accuracy, loss, pred, surprise) = train_and_test_at_intervals_clone_weights(w1, w2, epochs, lr, activation,
                                                                      testing_scenarios_inference_required,
                                                                      random.sample(ts[0], n_train), hidden_size, 2)
            inf_subjects.append((epoch, accuracy, loss))

            print(pred.data.numpy().ravel())
            preds[i, :] = (pred.data.numpy().ravel())
            surprises.append(surprise)

        x = inf_subjects[0][0]
        (inf_means, inf_stds, inf_losses) = get_mean_and_sems(inf_subjects)

    # plt.subplot(2, 1, 1)
    # plt.errorbar(x, inf_means, inf_stds, fmt='-o', label='Inference Required')
    # plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss (mse with S.E.M.)')
    #
    #
    # plt.subplot(2,1,2)
    # plt.plot(x, inf_losses, 'r', label='Loss')
    # plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss (mse with S.E.M.)')


        out_means = np.mean(preds, 0)

        so1_t = []
        so2_t = []
        so1_sem = []
        so2_sem = []
        for t in range(len(x)):
            mt1 = [k[t][1][0] for k in surprises]
            mt2 = [k[t][1][1] for k in surprises]
            so1_t.append(np.mean(mt1))
            so1_sem.append(scipy.stats.sem(mt1))
            so2_t.append(np.mean(mt2))
            so2_sem.append(scipy.stats.sem(mt2))
        print("DO mean = {}, DO sem = {}, SO mean = {}, SO sem = {}".format(so1_t, so1_sem, so2_t, so2_sem))
        log_file.write("N = {}, DO mean = {}, DO sem = {}, SO mean = {}, SO sem = {}\n".format(n_train, so1_t, so1_sem, so2_t, so2_sem))
        log_file.write(np.array2string(out_means))
        log_file.write("\n")

log_file.close()
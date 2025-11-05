import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['text.usetex'] = True
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('text.latex', preamble=r'\usepackage{cmbright}')


parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('--exp_index', '-i', help="Index of the experiment", default=1, required=True)
parser.add_argument('--dataset', '-d', help='Name of dataset', default='Null', required=True)
parser.add_argument('--lambda_val', '-l', help='Value of lambda', default=0.1)
parser.add_argument('--alpha_val', '-a', help='Value of alpha (should be b/w 0 to 1)', default=0.2)
parser.add_argument('--client', '-c', help='Number of client', default=80)
parser.add_argument('--seed', '-s', default=522)
parser.add_argument('--eps', '-e', help='Error level', default=0.0001)
parser.add_argument('--iterations', '-K', help='Number of iterations to run', default=10000)
parser.add_argument('--minibatch', '-t', help='Minibatch size', default=30)
args = parser.parse_args()

if __name__ == "__main__":
    # Setting for experiment 4
    plot_Euc = False    # Set to true if we want to plot the result in terms of standard euclidean norm for experiment 4

    # Hyperparameters
    exp_index   = int(args.exp_index)
    dataset     = args.dataset
    lam         = float(args.lambda_val)
    client      = int(args.client)
    seed        = int(args.seed)
    eps         = float(args.eps)
    iterations  = int(args.iterations)
    t           = int(args.minibatch)
    alpha       = float(args.alpha_val)
    font1 = {'family': 'New Times Romans', 'weight': 'normal', 'size': 18, }
    font2 = {'family': 'New Times Romans', 'weight': 'normal', 'size': 12, }
    font3 = {'family': 'New Times Romans', 'weight': 'normal', 'size': 10,}

    if exp_index == 1:
        result_dir = "result_exp_1"
        filename = ["logistic_exp_1_curve_{}_{}_lam_{}_seed_{}.npy".format(
            i, dataset, lam, seed
        ) for i in range(1, 9)]
        arrnm = [np.load(os.path.join(result_dir, filename[i])) for i in range(len(filename))]
        iterations = np.arange(1, arrnm[0].shape[0] + 1)
        # Plotting
        plt.figure(figsize=(6.5, 5.5))
        plt.xticks(fontname="New Times Romans")
        plt.yticks(fontname="New Times Romans")
        plt.xlabel(r"Iterations", font1)
        plt.ylabel(r"$G_{T, \bf{D}}$", font1)
        plt.yscale('log')
        plt.title(r"{}, rand-$1$ sketch, $\lambda={}$, $\beta={}$".format(dataset, lam, alpha), font1)
        plt.plot(iterations, arrnm[0],
                 label=r"Standard FGD",
                 marker='1', markevery=500, linestyle=(0, (1, 1)))
        plt.plot(iterations, arrnm[2],
                 label=r"Standard CFGD",
                 marker='o', markevery=500, linestyle=(0, (1, 1)))
        plt.plot(iterations, arrnm[3],
                 label=r"CFGD-mat",
                 marker='v', markevery=500, linestyle='dashed')
        plt.plot(iterations, arrnm[4],
                 label=r"CFGD$1$ with $\bf{D}_{1}$",
                 marker='<', markevery=400, linestyle='dashdot')
        plt.plot(iterations, arrnm[5],
                 label=r"CFGD$1$ with $\bf{D}_{2}$",
                 marker='>', markevery=600, linestyle='dotted')
        plt.plot(iterations, arrnm[6],
                 label=r"CFGD$1$ with $\bf{D}_{3}$",
                 marker='^', markevery=600, linestyle='dotted')
        plt.plot(iterations, arrnm[7],
                 label=r"CFGD$2$ with $\bf{D}_{4}$",
                 marker='2', markevery=500, linestyle=(0, (5, 1)))
        plt.grid(axis='x', linestyle='dashed')
        plt.legend(prop=font3)
        # plt.savefig("Exp_1_lam_{}".format(lam) + dataset + ".pdf")
        plt.savefig(os.path.join('figures', "Exp_1_lam_{}_".format(lam) + dataset + ".png"))
        # plt.show()

    if exp_index == 2:
        result_dir = "result_exp_2"
        filename = ["logistic_exp_2_curve_{}_{}_lam_{}_rand_{}_seed_522.npy".format(
            i, dataset, lam, t
        ) for i in range(1, 5)]
        arrnm = [np.load(os.path.join(result_dir, filename[i])) for i in range(len(filename))]
        iterations = np.arange(1, arrnm[0].shape[0] + 1)

        # Plotting
        plt.figure(figsize=(6.5, 5.5))
        plt.xticks(fontname="New Times Romans")
        plt.yticks(fontname="New Times Romans")
        plt.xlabel(r"Iterations", font1)
        plt.ylabel(r"$G_{T, \bf{D}}$", font1)
        plt.yscale('log')
        plt.title(r"{}, rand-${}$ sketch, $\lambda={}$, $\beta={}$".format(dataset, t ,lam, alpha), font1)
        plt.plot(iterations, arrnm[0],
                 label=r"Standard CFGD",
                 marker='o', markevery=500, linestyle=(0, (1, 1)))
        plt.plot(iterations, arrnm[1],
                 label=r"CFGD-mat",
                 marker='v', markevery=500, linestyle='dashed')
        plt.plot(iterations, arrnm[2],
                 label=r"CFGD$1$ with $\bf{D = D_{2}^{*}}$",
                 marker='^', markevery=400, linestyle='dashdot')
        plt.plot(iterations, arrnm[3],
                 label=r"CFGD$2$ with $\bf{D = D_{2}^{*}}$",
                 marker='2', markevery=400, linestyle='dashdot')
        plt.grid(axis='x', linestyle='dashed')
        plt.legend(prop=font3)
        # plt.savefig("Exp_2_lam_{}".format(lam) + dataset + ".pdf")
        plt.savefig(os.path.join('figures', "Exp_2_lam_{}_rand_{}_".format(lam, t) + dataset + ".png"))
        # plt.show()

    if exp_index == 3:
        raise NotImplementedError


    if exp_index == 4:
        result_dir = "result_exp_5"
        # For plotting in the standard Euclidean norm, just need to add "std_" for the last two files as a prefix
        filename_mat = ["ablation_dcfgd2_{}_lam_{}_client_{}_seed_{}_iter_{}_epi_{}_alpha_{}.npy".format(
            dataset, lam, client, seed, iterations, eps, alpha
        ) for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

        arrnm = [np.load(os.path.join(result_dir, filename_mat[i])) for i in range(len(filename_mat))]
        iterations = np.arange(1, arrnm[0].shape[0] + 1)
        beta_values = [0.1 * i for i in range(1, 10)]
        
        plt.figure(figsize=(6.5, 5.5))
        plt.xticks(fontname="New Times Romans")
        plt.yticks(fontname="New Times Romans")
        plt.xlabel(r"Iterations", fontdict=font1)
        if plot_Euc == False:
            plt.ylabel(r"$G_{T, \bf{D}}$", fontdict=font1)
        plt.yscale('log')
        plt.title(r"{}, rand-$1$ sketch, $\lambda={}$, $n={}$".format(dataset, lam, client), fontdict=font1)

        # Plot each arrnm with corresponding beta value
        # colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))
        # for i, beta in enumerate(beta_values):
        #     plt.plot(iterations, arrnm[i]*0.7,
        #             label=r"$\beta = {:.1f}$".format(beta),
        #             color=colors[i])

        # plt.grid(axis='x', linestyle='dashed')

        # # Place the legend below the plot
        # plt.legend(prop=font3, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5)
        # plt.tight_layout()
        colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))

        for i, beta in enumerate(beta_values):
            plt.plot(iterations, arrnm[i]*0.7,
                label=r"$\beta = {:.1f}$".format(beta),
                color=colors[i])

# Add grid and customize legend
        plt.grid(axis='x', linestyle='dashed')
        plt.legend(prop=font3, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5)

# Add text inside the graph
        plt.text(x=0.5, y=0.8, s=r"DCFGD-2 with $\mathbf{D} = \mathbf{D}_4$", fontsize=18, transform=plt.gca().transAxes,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        plt.tight_layout()
        # Save the plot
        output_dir = 'figures'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "ablation_dcfgd2_lam_{}_client_{}_".format(lam, client) + dataset + ".png"))


    if exp_index == 5:
        result_dir = "result_exp_6"
        # For plotting in the standard Euclidean norm, just need to add "std_" for the last two files as a prefix
        filename_mat = ["cfgd_diag{}_lam_{}_seed_{}_alpha_{}.npy".format(
            dataset, lam, seed, alpha
        ) for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

        arrnm = [np.load(os.path.join(result_dir, filename_mat[i])) for i in range(len(filename_mat))]
        iterations = np.arange(1, arrnm[0].shape[0] + 1)
        beta_values = [0.1 * i for i in range(1, 10)]
        
        plt.figure(figsize=(6.5, 5.5))
        plt.xticks(fontname="New Times Romans")
        plt.yticks(fontname="New Times Romans")
        plt.xlabel(r"Iterations", fontdict=font1)
        if plot_Euc == False:
            plt.ylabel(r"$G_{T, \bf{D}}$", fontdict=font1)
        plt.yscale('log')
        plt.title(r"{}, rand-$1$ sketch, $\lambda={}$".format(dataset, lam), fontdict=font1)

        # Plot each arrnm with corresponding beta value
        colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))
        for i, beta in enumerate(beta_values):
            plt.plot(iterations, arrnm[i],
                    label=r"$\beta = {:.1f}$".format(beta),
                    color=colors[i])

        plt.grid(axis='x', linestyle='dashed')

        # Place the legend below the plot
        plt.legend(prop=font3, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5)

# Add text inside the graph
        plt.text(x=0.5, y=0.8, s=r"CFGD-2 with $\mathbf{D} = \mathbf{D}_4$", fontsize=18, transform=plt.gca().transAxes,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.tight_layout()

        # Save the plot
        output_dir = 'figures'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "ablation_cfgd_diag_2_lam_{}_".format(lam) + dataset + ".png"))

import copy
import matplotlib.pyplot as plt

SIZES = [4, 8, 16, 32, 64, 128, 256]

base_results = {
    "sizes": SIZES,
    "dense": [0 for _ in SIZES],
    "conv1d": [0 for _ in SIZES],
    "lstm": [0 for _ in SIZES],
    "gru": [0 for _ in SIZES],
    "tanh": [0 for _ in SIZES],
    "relu": [0 for _ in SIZES],
    "sigmoid": [0 for _ in SIZES]
}

layer_names = {
    "dense": "Dense",
    "conv1d": "Conv1D",
    "lstm": "LSTM",
    "gru": "GRU",
    "tanh": "Tanh",
    "relu": "ReLU",
    "sigmoid": "Sigmoid"
}


def load_file_all(file):
    rt_dyn = copy.deepcopy(base_results)

    f = open(file, 'r')
    lines = f.readlines()

    for idx in range(0, len(lines), 4):
        info = lines[idx].split(' ')
        ltype = info[1]
        lsize = int(info[6])
        size_idx = base_results["sizes"].index(lsize)

        rt_dyn[ltype][size_idx] = float(lines[idx + 3].split('x')[0])

    f.close()
    return rt_dyn


def load_file_rtneural(file):
    rt_dyn = copy.deepcopy(base_results)

    f = open(file, 'r')
    lines = f.readlines()

    for idx in range(0, len(lines), 4):
        info = lines[idx].split(' ')
        ltype = info[1]
        lsize = int(info[6])
        size_idx = base_results["sizes"].index(lsize)

        rt_dyn[ltype][size_idx] = float(lines[idx + 3].split('x')[0])

    f.close()
    return rt_dyn


def make_plot(title, file_name, results):
    legend_labels = [
        'RTNeural - xsimd',
        'RTNeural - Eigen',
        'RTNeural - Eigen {approx}',
        'RTNeural - STL',
        'RTNeural - STL {approx}'
    ]

    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for i in range(len(results)):
        ax.semilogy(base_results["sizes"], results[i])
    ax.axhline(y=1, linestyle='--', color='r')
    plt.xscale('log', base=2)

    plt.title(title)
    plt.xlabel('Layer Size')
    plt.ylabel('Speed (real-time factor)')
    plt.grid(True)
    fig.legend(legend_labels, loc="outside right upper")

    plt.savefig(file_name)


rt_stl_dyn = load_file_all('results/bench_stl.txt')
rt_stl_approx_dyn = load_file_all('results/bench_stl_approx.txt')
rt_eigen_dyn = load_file_rtneural('results/bench_eigen.txt')
rt_eigen_approx_dyn = load_file_rtneural('results/bench_eigen_approx.txt')
rt_xsimd_dyn = load_file_rtneural('results/bench_xsimd.txt')

for k in base_results.keys():
    if k == "sizes":
        continue

    name = layer_names[k]
    make_plot(f'{name} Layer', f'plots/{k}.png',
              [rt_xsimd_dyn[k], rt_eigen_dyn[k], rt_eigen_approx_dyn[k], rt_stl_dyn[k], rt_stl_approx_dyn[k]])

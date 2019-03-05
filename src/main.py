import dataset, model, train, torch, util

@util.main
def main(disable=0, device="cuda", cycles=100, D=32, N=128):

    disable = int(disable)
    cycles= int(cycles)

    print("Using device: %s" % device)

    N = int(N)
    D = int(D)

    data = dataset.create(N, D)
    test = dataset.test(N, D)

    net = model.EvolutionaryModel(D, disable=disable).to(device)

    for i in range(cycles):
        net.do_cycle(*data, *test)

    best = net.select_best()
    print(best.net[0].weight[0])

    train.visualize(net.select_best(), outf="results.png", D=D)

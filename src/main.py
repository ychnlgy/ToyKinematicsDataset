import dataset, model, train, torch, util

@util.main
def main(disable=0, device="cpu", cycles=100, D=32, N=128, name="evo"):

    disable = int(disable)
    cycles= int(cycles)

    print("Using device: %s" % device)

    N = int(N)
    D = int(D)

    data = dataset.create(N, D)
    test = dataset.test(N, D)

    if name == "evo":
        net = model.EvolutionaryModel(D, disable=disable).to(device)

        try:
            for i in range(cycles):
                net.do_cycle(*data, *test)
        except KeyboardInterrupt:
            pass

        best = net.select_best()
        print(best.net[0].weight.data)

        train.visualize(net.select_best(), outf="results.png", D=D)

    else:
        net = model.Model(D)
        train.train(*data, *test, net)
        print(net.net[0].weight)

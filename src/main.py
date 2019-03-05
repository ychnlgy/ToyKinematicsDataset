import dataset, model, train, torch, util

@util.main
def main(disable=0):

    disable = int(disable)

    device = ["cpu", "cuda"][torch.cuda.is_available()]
    print("Using device: %s" % device)

    N = 512
    D = 32
    CYCLES = 100

    data = dataset.create(N, D)
    test = dataset.test(N, D)

    net = model.EvolutionaryModel(D, disable=disable).to(device)

    for i in range(CYCLES):
        net.do_cycle(*data, *test)

    train.visualize(net.select_best(), outf="results.png", D=D)
